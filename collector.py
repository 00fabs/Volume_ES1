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
lock           = threading.Lock()
m1_closed      = []
live_bar       = {}
last_closed_ts = 0
last_live_write= 0
start_time     = time.time()

# Depth history — keep last 5 snapshots for signal computation
depth_history  = deque(maxlen=5)

# Per-bar volume accumulator (reset on bar close)
bar_trade_vol  = 0.0

# ── Helpers ──────────────────────────────────────────────────
def now_utc():
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

def ts_to_eat(ts):
    return datetime.fromtimestamp(
        int(ts) + 3*3600, tz=timezone.utc
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
            "t":      str(bts),
            "o":      g[0]["o"],
            "h":      str(max(float(x["h"]) for x in g)),
            "l":      str(min(float(x["l"]) for x in g)),
            "c":      g[-1]["c"],
            "v":      str(sum(float(x["v"]) for x in g)),
            "vw":     str(sum(float(x["vw"]) for x in g)),
            "signal": _merge_m5_signal([x.get("signal","none") for x in g]),
            "signal_detail": g[-1].get("signal_detail", {}),
        })
    return m5

def _merge_m5_signal(signals):
    # If any bar in M5 group has a signal, propagate it
    if "absorption" in signals:
        return "absorption"
    if "continuation_bull" in signals:
        return "continuation_bull"
    if "continuation_bear" in signals:
        return "continuation_bear"
    return "none"

# ── Signal computation ───────────────────────────────────────
def compute_signal(closed_bar, m1_bars_so_far, depth_hist):
    """
    Runs on every bar close.
    Returns signal string and detail dict.
    """
    if len(m1_bars_so_far) < 5 or len(depth_hist) < 3:
        return "none", {}

    o = float(closed_bar["o"])
    h = float(closed_bar["h"])
    l = float(closed_bar["l"])
    c = float(closed_bar["c"])
    v = float(closed_bar["v"])

    bar_range = h - l
    if bar_range == 0:
        return "none", {}

    body_pct  = abs(c - o) / bar_range          # 0=doji 1=full body
    close_pct = (c - l) / bar_range             # 0=closed at low 1=at high

    # Average volume of last 20 closed bars
    lookback  = m1_bars_so_far[-20:]
    avg_vol   = sum(float(b["v"]) for b in lookback) / len(lookback)
    vol_ratio = v / avg_vol if avg_vol > 0 else 0

    # Depth snapshot analysis
    snaps     = list(depth_hist)[-3:]            # last 3 snapshots
    imbalances= [s["imbalance"] for s in snaps]
    ask_sizes = [s["ask_size"]  for s in snaps]
    bid_sizes = [s["bid_size"]  for s in snaps]

    # Imbalance trend
    imb_now   = imbalances[-1]
    imb_prev  = imbalances[0]
    imb_flipped_bull = imb_prev < 0 and imb_now > 0
    imb_consistently_bull = all(i >= 15 for i in imbalances)
    imb_consistently_bear = all(i <= -15 for i in imbalances)

    # Ask consumption (bull absorption / continuation)
    ask_depleted = False
    ask_refilled = False
    if len(ask_sizes) >= 3 and ask_sizes[0] > 0:
        ask_drop_pct = (ask_sizes[0] - ask_sizes[1]) / ask_sizes[0]
        ask_depleted = ask_drop_pct >= 0.40
        ask_refilled = ask_sizes[2] > ask_sizes[1]

    # Bid consumption (bear continuation)
    bid_depleted = False
    if len(bid_sizes) >= 3 and bid_sizes[0] > 0:
        bid_drop_pct = (bid_sizes[0] - bid_sizes[1]) / bid_sizes[0]
        bid_depleted = bid_drop_pct >= 0.40

    detail = {
        "vol_ratio":    round(vol_ratio, 2),
        "body_pct":     round(body_pct,  2),
        "close_pct":    round(close_pct, 2),
        "imb_now":      round(imb_now,   1),
        "imb_prev":     round(imb_prev,  1),
        "ask_depleted": ask_depleted,
        "ask_refilled": ask_refilled,
        "bid_depleted": bid_depleted,
        "avg_vol":      round(avg_vol,   0),
    }

    # ── Absorption reversal ──────────────────────────────────
    # High volume + doji/small body + imbalance flipped bull
    # + ask absorbed and refilled (passive seller was overwhelmed)
    if (vol_ratio  >= 1.5
    and body_pct   <= 0.30
    and imb_flipped_bull
    and ask_depleted
    and ask_refilled):
        return "absorption", detail

    # ── Bullish continuation ─────────────────────────────────
    # High volume + closed near high + persistent bull imbalance
    # + ask being consumed (no refill = price moving up through it)
    if (vol_ratio  >= 1.3
    and close_pct  >= 0.70
    and imb_consistently_bull
    and ask_depleted
    and not ask_refilled):
        return "continuation_bull", detail

    # ── Bearish continuation ─────────────────────────────────
    # High volume + closed near low + persistent bear imbalance
    # + bid being consumed
    if (vol_ratio  >= 1.3
    and close_pct  <= 0.30
    and imb_consistently_bear
    and bid_depleted):
        return "continuation_bear", detail

    return "none", detail

# ── HF upload ────────────────────────────────────────────────
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
        print(f"  [{now_utc()}] ❌ HF {filename}: {e}")

def write_closed_bars():
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
    sig = bars[-1].get("signal", "none")
    print(f"  [{now_utc()}] ✅ latest.json — "
          f"{len(bars)} M1 bars | last signal={sig}")

def write_live_bar():
    with lock:
        bar  = dict(live_bar)
        snap = list(depth_history)[-1] if depth_history else {}
    if not bar:
        return
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol":     SYM,
        "live_bar":   bar,
        "live_depth": snap,
    }
    _upload("live.json", payload, "live")

def load_existing_bars():
    try:
        import urllib.request
        url = (f"https://huggingface.co/datasets/"
               f"{HF_DATASET}/resolve/main/latest.json")
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

    # Kline M1
    ws.send(json.dumps({
        "code":  10006,
        "trace": uuid.uuid4().hex,
        "data":  {"arr": [{"type": 1, "codes": SYM}]}
    }))
    time.sleep(1)

    # Depth
    ws.send(json.dumps({
        "code":  10003,
        "trace": uuid.uuid4().hex,
        "data":  {"codes": SYM}
    }))
    print(f"[{now_utc()}] → Subscribed kline M1 + depth")

def on_message(ws, raw):
    global last_closed_ts, last_live_write

    data = json.loads(raw)
    code = data.get("code")

    # ── Depth push ───────────────────────────────────────────
    if code == 10005:
        d = data["data"]
        try:
            ask_px = float(d["a"][0][0]) if d.get("a") else 0
            ask_sz = float(d["a"][1][0]) if d.get("a") else 0
            bid_px = float(d["b"][0][0]) if d.get("b") else 0
            bid_sz = float(d["b"][1][0]) if d.get("b") else 0
            total  = ask_sz + bid_sz
            imb    = ((bid_sz - ask_sz) / total * 100) if total > 0 else 0
            snap = {
                "t":         int(time.time()),
                "ask_price": ask_px,
                "ask_size":  ask_sz,
                "bid_price": bid_px,
                "bid_size":  bid_sz,
                "imbalance": round(imb, 1),
                "spread":    round(ask_px - bid_px, 2),
            }
            with lock:
                depth_history.append(snap)
        except Exception:
            pass

    # ── Kline push ───────────────────────────────────────────
    elif code == 10008:
        d  = data["data"]
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
                if current_live_ts > 0 and current_live_ts > last_closed_ts:
                    closed        = dict(live_bar)
                    existing_ts   = {int(b["t"]) for b in m1_closed}

                    if current_live_ts not in existing_ts:
                        # Compute signal on close
                        sig, detail = compute_signal(
                            closed, m1_closed, depth_history
                        )
                        closed["signal"]        = sig
                        closed["signal_detail"] = detail
                        m1_closed.append(closed)
                        if len(m1_closed) > 500:
                            m1_closed.pop(0)

                    last_closed_ts = current_live_ts
                    s = closed.get("signal", "none")
                    print(f"  [{now_utc()}] 🔒 CLOSED "
                          f"{ts_to_eat(current_live_ts)} "
                          f"v={closed['v']} signal={s}")

                    threading.Thread(
                        target=write_closed_bars, daemon=True
                    ).start()

                live_bar.clear()
                live_bar.update(bar)

            else:
                live_bar.update(bar)

        now = time.time()
        if now - last_live_write >= 10:
            last_live_write = now
            threading.Thread(
                target=write_live_bar, daemon=True
            ).start()

    elif code == 10007:
        print(f"  [{now_utc()}] ✅ Kline sub confirmed")
    elif code == 10004:
        print(f"  [{now_utc()}] ✅ Depth sub confirmed")
    elif code == 200:
        print(f"  [{now_utc()}] 🟢 {data.get('msg')}")

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
    print(f"[{now_utc()}] Run complete.")

# ── Main ─────────────────────────────────────────────────────
def main():
    global m1_closed
    print(f"ES Collector — WebSocket + Signals")
    print(f"Symbol: {SYM} | {RUN_DURATION//3600}h\n")
    existing = load_existing_bars()
    with lock:
        m1_closed.extend(existing)
    run_with_reconnect()
    print(f"[{now_utc()}] Final write...")
    write_closed_bars()
    write_live_bar()

if __name__ == "__main__":
    main()
