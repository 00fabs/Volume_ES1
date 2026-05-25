import requests
import json
import time
import os
from datetime import datetime, timezone
from huggingface_hub import HfApi

# ── Config ───────────────────────────────────────────────────
API_KEY       = os.environ["DATA_API_KEY"]
HF_TOKEN      = os.environ["HF_TOKEN"]
HF_DATASET    = os.environ["HF_DATASET_REPO"]   # e.g. "yourname/es-volume-data"
BASE_URL      = "https://data.infoway.io"
SYM           = "ES1!"
POLL_INTERVAL = 10        # seconds between each time-check loop
RUN_DURATION  = 6 * 3600  # 6 hours

HEADERS_JSON = {
    "apiKey": API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}
HEADERS_GET = {
    "apiKey": API_KEY,
    "Accept": "application/json"
}

api = HfApi(token=HF_TOKEN)

# ── Time helpers ─────────────────────────────────────────────
def now_utc():
    return datetime.now(timezone.utc)

def last_closed_bar_ts():
    """
    Returns the unix timestamp of the last fully closed M1 bar.
    If current time is 12:58:47 → returns timestamp for 12:57:00.
    A bar closes when the next minute starts.
    We subtract one extra second as safety buffer.
    """
    now  = int(time.time())
    current_minute = (now // 60) * 60
    return current_minute - 60

def ts_to_str(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ── Fetch exactly one closed M1 bar by timestamp ─────────────
def fetch_closed_bar(bar_ts):
    """
    Fetches the M1 bar that closed at bar_ts.
    Uses timestamp param so Infoway anchors to that exact point.
    Returns the bar dict or None.
    """
    try:
        r = requests.post(
            f"{BASE_URL}/common/v2/batch_kline",
            headers=HEADERS_JSON,
            json={
                "klineType": 1,          # M1
                "klineNum":  3,          # fetch 3, pick the one we want
                "codes":     SYM,
                "timestamp": bar_ts + 59 # anchor just before bar close
            },
            timeout=10
        )
        data = r.json()
        if data.get("ret") != 200:
            print(f"  ⚠️  kline ret={data.get('ret')} msg={data.get('msg')}")
            return None

        bars = data["data"][0]["respList"]
        # Find the bar whose timestamp matches exactly
        for b in bars:
            if int(b["t"]) == bar_ts:
                return b

        # If exact match not found, take the closest bar <= bar_ts
        candidates = [b for b in bars if int(b["t"]) <= bar_ts]
        if candidates:
            return max(candidates, key=lambda x: int(x["t"]))

        return None

    except Exception as e:
        print(f"  ❌ fetch_closed_bar error: {e}")
        return None

# ── Load existing bars from HF dataset ───────────────────────
def load_existing():
    """
    Loads latest.json from HF dataset.
    Returns the m1_bars list and last_ts we already have.
    """
    try:
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/latest.json"
        r   = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            bars = data.get("m1_bars", [])
            print(f"  Loaded {len(bars)} existing bars from HF dataset.")
            return bars
        else:
            print(f"  No existing data found (HTTP {r.status_code}). Starting fresh.")
            return []
    except Exception as e:
        print(f"  Could not load existing data: {e}. Starting fresh.")
        return []

# ── Build M5 bars from M1 bars ────────────────────────────────
def build_m5(m1_bars):
    """
    Aggregates M1 bars into M5 bars.
    Groups by floor(timestamp / 300) * 300.
    """
    buckets = {}
    for b in m1_bars:
        t      = int(b["t"])
        bucket = (t // 300) * 300
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(b)

    m5_bars = []
    for bucket_ts in sorted(buckets.keys()):
        group = buckets[bucket_ts]
        m5_bars.append({
            "t":  str(bucket_ts),
            "o":  group[0]["o"],
            "h":  str(max(float(b["h"]) for b in group)),
            "l":  str(min(float(b["l"]) for b in group)),
            "c":  group[-1]["c"],
            "v":  str(sum(float(b["v"]) for b in group)),
            "vw": str(sum(float(b["vw"]) for b in group)),
        })

    return m5_bars

# ── Write to HF dataset ───────────────────────────────────────
def write_to_hf(m1_bars):
    m5_bars = build_m5(m1_bars)
    payload = {
        "updated_at": now_utc().isoformat(),
        "symbol":     SYM,
        "m1_bars":    m1_bars,
        "m5_bars":    m5_bars,
    }
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")
    try:
        api.upload_file(
            path_or_fileobj=json_bytes,
            path_in_repo="latest.json",
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=f"bar {ts_to_str(int(m1_bars[-1]['t']))}"
        )
        print(f"  ✅ HF updated — {len(m1_bars)} M1 bars, {len(m5_bars)} M5 bars")
    except Exception as e:
        print(f"  ❌ HF write failed: {e}")

# ── Main loop ─────────────────────────────────────────────────
def main():
    start = time.time()
    print(f"Collector started at {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Symbol: {SYM} | Poll every {POLL_INTERVAL}s | Run for {RUN_DURATION//3600}h\n")

    # Load whatever bars we already have so we don't lose history on restart
    m1_bars = load_existing()

    # Track the timestamp of the last bar we successfully stored
    last_stored_ts = int(m1_bars[-1]["t"]) if m1_bars else 0

    while True:
        elapsed = time.time() - start
        if elapsed >= RUN_DURATION:
            print("6-hour limit reached. Final write...")
            if m1_bars:
                write_to_hf(m1_bars)
            print("Done.")
            break

        now_str    = now_utc().strftime("%H:%M:%S UTC")
        target_ts  = last_closed_bar_ts()

        print(f"[{now_str}] Last closed bar should be: {ts_to_str(target_ts)}")

        if target_ts <= last_stored_ts:
            # No new closed bar yet — current minute still forming
            print(f"  → Already have this bar. Waiting for next close.")
        else:
            # New bar has closed — fetch it
            print(f"  → New bar detected! Fetching...")
            time.sleep(1.2)   # rate limit buffer
            bar = fetch_closed_bar(target_ts)

            if bar:
                actual_ts = int(bar["t"])
                print(f"  → Got bar: t={ts_to_str(actual_ts)} "
                      f"o={bar['o']} h={bar['h']} l={bar['l']} "
                      f"c={bar['c']} v={bar['v']}")

                # Only append if this timestamp is truly new
                if actual_ts > last_stored_ts:
                    m1_bars.append(bar)
                    last_stored_ts = actual_ts

                    # Keep last 500 bars in memory to avoid huge JSON
                    if len(m1_bars) > 500:
                        m1_bars = m1_bars[-500:]

                    write_to_hf(m1_bars)
                else:
                    print(f"  → Bar timestamp {actual_ts} not newer than {last_stored_ts}. Skipping.")
            else:
                print(f"  → Bar not available yet. Will retry next poll.")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
