import websocket
import json
import time
import os
import threading
import uuid
from datetime import datetime, timezone
from huggingface_hub import HfApi

# ── Config ───────────────────────────────────────────────────
API_KEY      = os.environ["DATA_API_KEY"]
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_DATASET   = os.environ["HF_DATASET_REPO"]
SYM          = "ES1!"
RUN_DURATION = 6 * 3600   # 6 hours

WS_URL = f"wss://data.infoway.io/ws?business=common&apikey={API_KEY}"

api = HfApi(token=HF_TOKEN)

# ── Shared state (thread-safe via lock) ──────────────────────
lock             = threading.Lock()
m1_closed        = []       # list of fully closed M1 bar dicts
live_bar         = {}       # current building bar
last_closed_ts   = 0        # timestamp of last bar we wrote to HF
last_live_write  = 0        # time of last live.json write
is_connected     = False
start_time       = time.time()

# ── Helpers ──────────────────────────────────────────────────
def now_utc():
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

def ts_to_eat(ts):
    return datetime.fromtimestamp(
        int(ts) + 3 * 3600, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S EAT")

def build_m5(m1_bars):
    buckets = {}
    for b in m1_bars:
        t      = int(b["t"])
        bucket = (t // 300) * 300
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(b)
    m5 = []
    for bts in sorted(buckets):
        g = buckets[bts]
        m5.append({
            "t":  str(bts),
            "o":  g[0]["o"],
            "h":  str(max(float(x["h"]) for x in g)),
            "l":  str(min(float(x["l"]) for x in g)),
            "c":  g[-1]["c"],
            "v":  str(sum(float(x["v"]) for x in g)),
            "vw": str(sum(float(x["vw"]) for x in g)),
        })
    return m5

def write_closed_bars():
    """Write latest.json — all closed M1 + M5 bars."""
    with lock:
        bars = list(m1_closed)

    if not bars:
        return

    m5 = build_m5(bars)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol":     SYM,
        "m1_bars":    bars,
        "m5_bars":    m5,
    }
    _upload("latest.json", payload,
            f"close {ts_to_eat(bars[-1]['t'])}")
    print(f"  [{now_utc()}] ✅ latest.json — {len(bars)} M1, {len(m5)} M5 bars")

def write_live_bar():
    """Write live.json — current building bar only."""
    with lock:
        bar = dict(live_bar)

    if not bar:
        return

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol":     SYM,
        "live_bar":   bar,
    }
    _upload("live.json", payload, "live")

def _upload(filename, payload, msg):
    try:
        data = json.dumps(payload, indent=2).encode("utf-8")
        api.upload_file(
            path_or_fileobj=data,
            path_in_repo=filename,
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=msg,
        )
    except Exception as e:
        print(f"  [{now_utc()}] ❌ HF upload {filename} failed: {e}")

def load_existing_bars():
    """On startup load whatever closed bars are already in HF."""
    try:
        url  = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/latest.json"
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        bars = data.get("m1_bars", [])
        print(f"  [{now_utc()}] Loaded {len(bars)} existing bars from HF.")
        return bars
    except Exception:
        print(f"  [{now_utc()}] No existing bars found. Starting fresh.")
        return []

# ── WebSocket callbacks ──────────────────────────────────────
def on_open(ws):
    global is_connected
    is_connected = True
    print(f"[{now_utc()}] ✅ Connected\n")

    # Subscribe kline M1
    ws.send(json.dumps({
        "code":  10006,
        "trace": uuid.uuid4().hex,
        "data":  {"arr": [{"type": 1, "codes": SYM}]}
    }))
    time.sleep(1)

    # Subscribe kline M5 as well (type 2)
    ws.send(json.dumps({
        "code":  10006,
        "trace": uuid.uuid4().hex,
        "data":  {"arr": [{"type": 2, "codes": SYM}]}
    }))
    print(f"[{now_utc()}] → Subscribed to M1 + M5 kline for {SYM}")

def on_message(ws, raw):
    global last_closed_ts, last_live_write

    data = json.loads(raw)
    code = data.get("code")

    # ── Kline push ───────────────────────────────────────────
    if code == 10008:
        d  = data["data"]
        ty = d.get("ty", 0)

        # Only process M1 (ty=1) — M5 we rebuild ourselves
        if ty != 1:
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
                # New bar started — previous live bar is now closed
                if current_live_ts > 0 and current_live_ts > last_closed_ts:
                    closed = dict(live_bar)
                    # Avoid duplicates
                    existing_ts = {int(b["t"]) for b in m1_closed}
                    if current_live_ts not in existing_ts:
                        m1_closed.append(closed)
                        # Keep last 500 bars
                        if len(m1_closed) > 500:
                            m1_closed.pop(0)
                    last_closed_ts = current_live_ts

                    print(f"  [{now_utc()}] 🔒 BAR CLOSED"
                          f" | {ts_to_eat(current_live_ts)}"
                          f" | v={closed['v']} ct")

                    # Write closed bars to HF on every bar close
                    threading.Thread(
                        target=write_closed_bars, daemon=True
                    ).start()

                # Update live bar to new bar
                live_bar.clear()
                live_bar.update(bar)

            else:
                # Same bar — update live bar with latest values
                live_bar.update(bar)

        # Write live.json every 10 seconds
        now = time.time()
        if now - last_live_write >= 10:
            last_live_write = now
            threading.Thread(
                target=write_live_bar, daemon=True
            ).start()

    # ── Subscription ack ─────────────────────────────────────
    elif code in (10007,):
        print(f"  [{now_utc()}] ✅ Kline subscription confirmed")

    elif code == 200:
        print(f"  [{now_utc()}] 🟢 {data.get('msg')}")

    # ── Errors ───────────────────────────────────────────────
    elif code in (506, 507):
        print(f"  [{now_utc()}] ⚠️  WS error {code}: {data.get('msg')}")

def on_error(ws, error):
    print(f"[{now_utc()}] ❌ WS error: {error}")

def on_close(ws, close_code, msg):
    global is_connected
    is_connected = False
    print(f"[{now_utc()}] 🔌 WS closed | code={close_code}")

# ── Reconnect loop ───────────────────────────────────────────
def run_with_reconnect():
    global is_connected
    while time.time() - start_time < RUN_DURATION:
        print(f"[{now_utc()}] Connecting to WebSocket...")
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)

        # If we get here the connection dropped
        if time.time() - start_time < RUN_DURATION:
            print(f"[{now_utc()}] Reconnecting in 5s...")
            time.sleep(5)

    print(f"[{now_utc()}] 6-hour run complete.")

# ── Main ─────────────────────────────────────────────────────
def main():
    global m1_closed

    print(f"ES Volume Collector — WebSocket mode")
    print(f"Symbol: {SYM} | Duration: {RUN_DURATION//3600}h\n")

    # Load existing bars so session history is preserved on restart
    existing = load_existing_bars()
    with lock:
        m1_closed.extend(existing)

    run_with_reconnect()

    # Final write on exit
    print(f"[{now_utc()}] Final write...")
    write_closed_bars()
    write_live_bar()
    print(f"[{now_utc()}] Done.")

if __name__ == "__main__":
    main()
