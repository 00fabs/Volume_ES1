import websocket
import json
import time
import os
import threading
import uuid
from datetime import datetime, timezone
from collections import deque
from huggingface_hub import HfApi

# ── Config ───────────────────────────────────────────────────
API_KEY      = os.environ["DATA_API_KEY"]
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_DATASET   = os.environ["HF_DATASET_REPO"]
SYM          = "ES1!"
RUN_DURATION = 6 * 3600

WS_URL = f"wss://data.infoway.io/ws?business=common&apikey={API_KEY}"
api    = HfApi(token=HF_TOKEN)

# ── Shared state ─────────────────────────────────────────────
lock             = threading.Lock()
m1_closed        = []
live_bar         = {}
last_closed_ts   = 0
start_time       = time.time()
depth_history    = deque(maxlen=5)
bar_price_levels = {}

# HF throttle
last_hf_commit   = 0
MIN_COMMIT_GAP   = 45
pending_commit   = False

# ── Helpers ──────────────────────────────────────────────────
def now_utc():
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

def ts_to_eat(ts):
    return datetime.fromtimestamp(
        int(ts) + 3*3600, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S EAT")

def aggregate_poc(bars_in_group):
    """
    Re-aggregates price_levels across a group of M1 bars
    to find the true POC for M5/M15.
    """
    combined = {}
    for b in bars_in_group:
        pl = b.get("price_levels", {})
        for price_str, vol in pl.items():
            combined[price_str] = combined.get(price_str, 0) + vol
    if not combined:
        return None, None
    poc_price = max(combined, key=combined.get)
    poc_vol   = combined[poc_price]
    return float(poc_price), round(poc_vol, 0)

def build_tf(m1_bars, bucket_seconds, label):
    """Generic builder for M5 (300s) and M15 (900s)."""
    buckets = {}
    for b in m1_bars:
        t      = int(b["t"])
        bucket = (t // bucket_seconds) * bucket_seconds
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(b)

    result = []
    for bts in sorted(buckets):
        g         = buckets[bts]
        poc_price, poc_vol = aggregate_poc(g)
        result.append({
            "t":          str(bts),
            "o":          g[0]["o"],
            "h":          str(max(float(x["h"]) for x in g)),
            "l":          str(min(float(x["l"]) for x in g)),
            "c":          g[-1]["c"],
            "v":          str(sum(float(x["v"]) for x in g)),
            "vw":         str(sum(float(x["vw"]) for x in g)),
            "poc_price":  poc_price,
            "poc_vol":    poc_vol,
        })
    return result

# ── HF upload ────────────────────────────────────────────────
def write_to_hf(reason="bar_close"):
    global last_hf_commit, pending_commit

    now = time.time()
    if now - last_hf_commit < MIN_COMMIT_GAP:
        pending_commit = True
        wait = int(MIN_COMMIT_GAP - (now - last_hf_commit))
        print(f"  [{now_utc()}] ⏳ Throttled — retry in {wait}s")
        return

    with lock:
        bars = list(m1_closed)
        live = dict(live_bar)
        snap = list(depth_history)[-1] if depth_history else {}

    if not bars:
        return

    m5  = build_tf(bars, 300, "M5")
    m15 = build_tf(bars, 900, "M15")

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol":     SYM,
        "m1_bars":    bars,
        "m5_bars":    m5,
        "m15_bars":   m15,
        "live_bar":   live,
        "live_depth": snap,
    }

    try:
        data = json.dumps(payload, indent=2).encode("utf-8")
        api.upload_file(
            path_or_fileobj=data,
            path_in_repo="data.json",
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=f"{reason} {ts_to_eat(bars[-1]['t'])}",
        )
        last_hf_commit = time.time()
        pending_commit = False
        print(f"  [{now_utc()}] ✅ data.json — "
              f"{len(bars)} M1 | {len(m5)} M5 | {len(m15)} M15")
    except Exception as e:
        print(f"  [{now_utc()}] ❌ HF failed: {e}")
        last_hf_commit = time.time()

def throttle_watchdog():
    global pending_commit
    while True:
        time.sleep(10)
        if pending_commit:
            if time.time() - last_hf_commit >= MIN_COMMIT_GAP:
                print(f"  [{now_utc()}] 🔄 Retry pending commit...")
                threading.Thread(
                    target=write_to_hf,
                    args=("retry",),
                    daemon=True
                ).start()

# ── Heartbeat ────────────────────────────────────────────────
def send_heartbeat(ws):
    while True:
        time.sleep(20)
        try:
            ws.send(json.dumps({
                "code":  0,
                "trace": uuid.uuid4().hex,
                "data":  {}
            }))
        except Exception:
            break

# ── Load existing ────────────────────────────────────────────
def load_existing_bars():
    try:
        import urllib.request
        url = (f"https://huggingface.co/datasets/"
               f"{HF_DATASET}/resolve/main/data.json")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        bars = data.get("m1_bars", [])
        print(f"  [{now_utc()}] Loaded {len(bars)} bars from HF.")
        return bars
    except Exception:
        print(f"  [{now_utc()}] Starting fresh.")
        return []

# ── WebSocket callbacks ──────────────────────────────────────
def on_open(ws):
    print(f"[{now_utc()}] ✅ Connected")

    threading.Thread(
        target=send_heartbeat, args=(ws,), daemon=True
    ).start()

    ws.send(json.dumps({
        "code":  10006,
        "trace": uuid.uuid4().hex,
        "data":  {"arr": [{"type": 1, "codes": SYM}]}
    }))
    time.sleep(1)

    ws.send(json.dumps({
        "code":  10003,
        "trace": uuid.uuid4().hex,
        "data":  {"codes": SYM}
    }))
    print(f"[{now_utc()}] → Subscribed M1 kline + depth + heartbeat")

def on_message(ws, raw):
    global last_closed_ts

    data = json.loads(raw)
    code = data.get("code")

    # ── Depth ────────────────────────────────────────────────
    if code == 10005:
        d = data["data"]
        try:
            ask_px = float(d["a"][0][0]) if d.get("a") else 0
            ask_sz = float(d["a"][1][0]) if d.get("a") else 0
            bid_px = float(d["b"][0][0]) if d.get("b") else 0
            bid_sz = float(d["b"][1][0]) if d.get("b") else 0
            total  = ask_sz + bid_sz
            imb    = ((bid_sz - ask_sz) / total * 100) if total > 0 else 0
            with lock:
                depth_history.append({
                    "t":         int(time.time()),
                    "ask_price": ask_px,
                    "ask_size":  ask_sz,
                    "bid_price": bid_px,
                    "bid_size":  bid_sz,
                    "imbalance": round(imb, 1),
                    "spread":    round(ask_px - bid_px, 2),
                })
        except Exception:
            pass

    # ── Trade — POC accumulation ─────────────────────────────
    elif code == 10002:
        d = data["data"]
        try:
            price = str(round(float(d["p"]) * 4) / 4)
            vol   = float(d["v"])
            with lock:
                bar_price_levels[price] = \
                    bar_price_levels.get(price, 0) + vol
        except Exception:
            pass

    # ── Kline ────────────────────────────────────────────────
    elif code == 10008:
        d = data["data"]
        if d.get("ty") != 1:
            return

        bar_ts = int(d["t"])
        bar = {
            "t":  str(bar_ts),
            "o":  d["o"],
            "h":  d["h"],
            "l":  d["l"],
            "c":  d["c"],
            "v":  d["v"],
            "vw": d["vw"],
        }

        with lock:
            current_live_ts = int(live_bar.get("t", 0))

            if bar_ts > current_live_ts:
                if current_live_ts > 0 and \
                   current_live_ts > last_closed_ts:

                    closed      = dict(live_bar)
                    existing_ts = {int(b["t"]) for b in m1_closed}

                    if current_live_ts not in existing_ts:
                        # Store full price levels for M5/M15 POC aggregation
                        closed["price_levels"] = dict(bar_price_levels)

                        # POC for M1 display
                        if bar_price_levels:
                            poc_p = max(
                                bar_price_levels,
                                key=bar_price_levels.get
                            )
                            closed["poc_price"] = float(poc_p)
                            closed["poc_vol"]   = round(
                                bar_price_levels[poc_p], 0
                            )
                        else:
                            closed["poc_price"] = None
                            closed["poc_vol"]   = None

                        bar_price_levels.clear()
                        m1_closed.append(closed)
                        if len(m1_closed) > 500:
                            m1_closed.pop(0)

                    last_closed_ts = current_live_ts
                    poc = closed.get("poc_price", "?")
                    print(f"  [{now_utc()}] 🔒 CLOSED "
                          f"{ts_to_eat(current_live_ts)} "
                          f"v={closed['v']} poc={poc}")

                    threading.Thread(
                        target=write_to_hf,
                        args=("bar_close",),
                        daemon=True
                    ).start()

                live_bar.clear()
                live_bar.update(bar)
            else:
                live_bar.update(bar)

    elif code in (10007, 10004):
        label = "Kline" if code == 10007 else "Depth"
        print(f"  [{now_utc()}] ✅ {label} sub confirmed")
    elif code == 200:
        print(f"  [{now_utc()}] 🟢 {data.get('msg')}")
    elif code == 0:
        pass  # heartbeat ack

def on_error(ws, error):
    print(f"[{now_utc()}] ❌ {error}")

def on_close(ws, c, m):
    print(f"[{now_utc()}] 🔌 Closed")

# ── Reconnect loop ───────────────────────────────────────────
def run_with_reconnect():
    while time.time() - start_time < RUN_DURATION:
        print(f"[{now_utc()}] Connecting...")
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)
        if time.time() - start_time < RUN_DURATION:
            print(f"[{now_utc()}] Reconnecting in 5s...")
            time.sleep(5)
    print(f"[{now_utc()}] 6h complete.")

# ── Main ─────────────────────────────────────────────────────
def main():
    global m1_closed
    print(f"ES Collector — WebSocket + POC")
    print(f"Symbol: {SYM} | {RUN_DURATION//3600}h\n")
    existing = load_existing_bars()
    with lock:
        m1_closed.extend(existing)
    threading.Thread(
        target=throttle_watchdog, daemon=True
    ).start()
    run_with_reconnect()
    print(f"[{now_utc()}] Final write...")
    write_to_hf("session_end")

if __name__ == "__main__":
    main()
