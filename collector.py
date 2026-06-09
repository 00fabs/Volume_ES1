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
live_levels      = {}   # price_levels for the building bar
last_closed_ts   = 0
start_time       = time.time()
depth_history    = deque(maxlen=5)

# Per-bar trade accumulator — resets on bar close
# Structure: { "7447.50": {"b": 137, "s": 23} }
bar_levels       = {}
prev_price       = 0.0
prev_dir         = 0

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

def tick_dir(price):
    global prev_price, prev_dir
    if price > prev_price:   prev_dir =  1
    elif price < prev_price: prev_dir = -1
    prev_price = price
    return prev_dir

def accum_level(levels_dict, price, vol, dirn):
    key = str(round(price * 4) / 4)
    if key not in levels_dict:
        levels_dict[key] = {"b": 0.0, "s": 0.0}
    if dirn == 1:
        levels_dict[key]["b"] += vol
    elif dirn == -1:
        levels_dict[key]["s"] += vol
    else:
        # Unknown direction — split evenly
        levels_dict[key]["b"] += vol / 2
        levels_dict[key]["s"] += vol / 2

def levels_to_compact(levels_dict):
    """Round values to 1dp to keep JSON small."""
    return {
        k: {"b": round(v["b"], 1), "s": round(v["s"], 1)}
        for k, v in levels_dict.items()
        if v["b"] + v["s"] > 0
    }

def build_tf(m1_bars, bucket_secs):
    buckets = {}
    for b in m1_bars:
        t      = int(b["t"])
        bucket = (t // bucket_secs) * bucket_secs
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(b)

    result = []
    for bts in sorted(buckets):
        g = buckets[bts]

        # Aggregate price levels across all M1 bars in group
        combined = {}
        for bar in g:
            for px, vs in bar.get("price_levels", {}).items():
                if px not in combined:
                    combined[px] = {"b": 0.0, "s": 0.0}
                combined[px]["b"] += vs.get("b", 0)
                combined[px]["s"] += vs.get("s", 0)

        # POC — highest total volume level
        poc_price = poc_vol = None
        if combined:
            poc_key   = max(combined,
                            key=lambda k: combined[k]["b"] +
                                          combined[k]["s"])
            poc_price = float(poc_key)
            poc_vol   = round(
                combined[poc_key]["b"] + combined[poc_key]["s"], 0
            )

        result.append({
            "t":            str(bts),
            "o":            g[0]["o"],
            "h":            str(max(float(x["h"]) for x in g)),
            "l":            str(min(float(x["l"]) for x in g)),
            "c":            g[-1]["c"],
            "v":            str(sum(float(x["v"]) for x in g)),
            "vw":           str(sum(float(x["vw"]) for x in g)),
            "price_levels": levels_to_compact(combined),
            "poc_price":    poc_price,
            "poc_vol":      poc_vol,
        })
    return result

# ── HF upload ────────────────────────────────────────────────
def write_to_hf(reason="bar_close"):
    global last_hf_commit, pending_commit

    if time.time() - last_hf_commit < MIN_COMMIT_GAP:
        pending_commit = True
        return

    with lock:
        bars  = list(m1_closed)
        live  = dict(live_bar)
        snap  = list(depth_history)[-1] if depth_history else {}
        lvls  = levels_to_compact(dict(live_levels))

    if not bars:
        return

    m5  = build_tf(bars, 300)
    m15 = build_tf(bars, 900)

    # Attach live price levels to live bar
    live_with_levels = dict(live)
    live_with_levels["price_levels"] = lvls

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol":     SYM,
        "m1_bars":    bars,
        "m5_bars":    m5,
        "m15_bars":   m15,
        "live_bar":   live_with_levels,
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
        print(f"  [{now_utc()}] ✅ {len(bars)} M1 | "
              f"{len(m5)} M5 | {len(m15)} M15")
    except Exception as e:
        print(f"  [{now_utc()}] ❌ HF: {e}")
        last_hf_commit = time.time()

def throttle_watchdog():
    global pending_commit
    while True:
        time.sleep(10)
        if pending_commit:
            if time.time() - last_hf_commit >= MIN_COMMIT_GAP:
                threading.Thread(
                    target=write_to_hf, args=("retry",),
                    daemon=True
                ).start()

# ── WebSocket callbacks ──────────────────────────────────────
def subscribe(ws):
    """Send all subscriptions. Called on open and reconnect."""
    # Kline M1
    ws.send(json.dumps({
        "code":  10006,
        "trace": uuid.uuid4().hex,
        "data":  {"arr": [{"type": 1, "codes": SYM}]}
    }))
    time.sleep(0.5)
    # Trade feed
    ws.send(json.dumps({
        "code":  10000,
        "trace": uuid.uuid4().hex,
        "data":  {"codes": SYM, "includeTy": True}
    }))
    time.sleep(0.5)
    # Depth
    ws.send(json.dumps({
        "code":  10003,
        "trace": uuid.uuid4().hex,
        "data":  {"codes": SYM}
    }))

def on_open(ws):
    print(f"[{now_utc()}] ✅ Connected")
    subscribe(ws)

def on_message(ws, raw):
    global last_closed_ts

    data = json.loads(raw)
    code = data.get("code")

    # ── Trade — accumulate footprint levels ──────────────────
    if code == 10002:
        d     = data["data"]
        price = float(d["p"])
        vol   = float(d["v"])
        dirn  = tick_dir(price)
        with lock:
            accum_level(bar_levels, price, vol, dirn)
            accum_level(live_levels, price, vol, dirn)

    # ── Depth ────────────────────────────────────────────────
    elif code == 10005:
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
                # Previous bar just closed
                if current_live_ts > 0 and \
                   current_live_ts > last_closed_ts:

                    closed      = dict(live_bar)
                    existing_ts = {int(b["t"]) for b in m1_closed}

                    if current_live_ts not in existing_ts:
                        # Attach accumulated levels to closed bar
                        closed["price_levels"] = \
                            levels_to_compact(dict(bar_levels))

                        # POC
                        if bar_levels:
                            poc_key = max(
                                bar_levels,
                                key=lambda k: bar_levels[k]["b"] +
                                              bar_levels[k]["s"]
                            )
                            closed["poc_price"] = float(poc_key)
                            closed["poc_vol"]   = round(
                                bar_levels[poc_key]["b"] +
                                bar_levels[poc_key]["s"], 0
                            )

                        m1_closed.append(closed)
                        if len(m1_closed) > 500:
                            m1_closed.pop(0)

                    # Reset accumulators for new bar
                    bar_levels.clear()
                    live_levels.clear()
                    last_closed_ts = current_live_ts

                    print(f"  [{now_utc()}] 🔒 "
                          f"{ts_to_eat(current_live_ts)} "
                          f"v={closed['v']} "
                          f"levels={len(closed.get('price_levels',{}))}")

                    threading.Thread(
                        target=write_to_hf,
                        args=("bar_close",),
                        daemon=True
                    ).start()

                live_bar.clear()
                live_bar.update(bar)
            else:
                live_bar.update(bar)

    elif code == 10007:
        print(f"  [{now_utc()}] ✅ Kline sub confirmed")
    elif code == 10001:
        print(f"  [{now_utc()}] ✅ Trade sub confirmed")
    elif code == 10004:
        print(f"  [{now_utc()}] ✅ Depth sub confirmed")
    elif code == 200:
        print(f"  [{now_utc()}] 🟢 {data.get('msg')}")
    elif code == 0:
        pass

def on_error(ws, e):
    print(f"[{now_utc()}] ❌ {e}")

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
        ws.run_forever(ping_interval=15, ping_timeout=8)
        if time.time() - start_time < RUN_DURATION:
            print(f"[{now_utc()}] Reconnecting in 3s...")
            time.sleep(3)
    print(f"[{now_utc()}] 6h complete.")

# ── Load existing ────────────────────────────────────────────
def load_existing():
    try:
        import urllib.request
        url = (f"https://huggingface.co/datasets/"
               f"{HF_DATASET}/resolve/main/data.json")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        bars = data.get("m1_bars", [])
        print(f"  [{now_utc()}] Loaded {len(bars)} bars.")
        return bars
    except Exception:
        print(f"  [{now_utc()}] Starting fresh.")
        return []

# ── Main ─────────────────────────────────────────────────────
def main():
    global m1_closed
    print(f"ES Collector — Footprint + WebSocket")
    print(f"Symbol: {SYM} | {RUN_DURATION//3600}h\n")

    existing = load_existing()
    with lock:
        m1_closed.extend(existing)

    threading.Thread(
        target=throttle_watchdog, daemon=True
    ).start()

    run_with_reconnect()
    write_to_hf("session_end")

if __name__ == "__main__":
    main()
