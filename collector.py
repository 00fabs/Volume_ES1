import requests
import json
import time
import os
from datetime import datetime, timezone
from huggingface_hub import HfApi

# ── Config ──────────────────────────────────────────────────
API_KEY        = os.environ["DATA_API_KEY"]
HF_TOKEN       = os.environ["HF_TOKEN"]
HF_DATASET     = os.environ["HF_DATASET_REPO"]   # e.g. "yourname/es-volume-data"
BASE_URL       = "https://data.infoway.io"
SYM            = "ES1!"
POLL_INTERVAL  = 20          # seconds between each Infoway poll
WRITE_INTERVAL = 30         # seconds between each HF dataset write (5 min)
RUN_DURATION   = 6 * 3600    # 6 hours total runtime in seconds

HEADERS = {
    "apiKey": API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

api = HfApi(token=HF_TOKEN)

# ── Helpers ──────────────────────────────────────────────────
def fetch_kline(kline_type, kline_num):
    try:
        r = requests.post(
            f"{BASE_URL}/common/v2/batch_kline",
            headers=HEADERS,
            json={"klineType": kline_type, "klineNum": kline_num, "codes": SYM},
            timeout=10
        )
        data = r.json()
        if data.get("ret") == 200:
            return data["data"][0]["respList"]
    except Exception as e:
        print(f"  kline error: {e}")
    return []

def fetch_depth():
    try:
        r = requests.get(
            f"{BASE_URL}/common/batch_depth/{SYM}",
            headers={"apiKey": API_KEY, "Accept": "application/json"},
            timeout=10
        )
        data = r.json()
        if data.get("ret") == 200:
            return data["data"][0]
    except Exception as e:
        print(f"  depth error: {e}")
    return None

def fetch_last_trade():
    try:
        r = requests.get(
            f"{BASE_URL}/common/batch_trade/{SYM}",
            headers={"apiKey": API_KEY, "Accept": "application/json"},
            timeout=10
        )
        data = r.json()
        if data.get("ret") == 200:
            return data["data"][0]
    except Exception as e:
        print(f"  trade error: {e}")
    return None

def write_to_hf(m1_bars, m5_bars, depth_snap, last_trade):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"  Writing to HF dataset at {ts}...")

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": SYM,
        "m1": m1_bars,
        "m5": m5_bars,
        "depth": depth_snap,
        "last_trade": last_trade
    }

    # Write as a single JSON file — the HF space reads this
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    try:
        api.upload_file(
            path_or_fileobj=json_bytes,
            path_in_repo="latest.json",
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=f"update {ts}"
        )
        print(f"  ✅ Written latest.json ({len(json_bytes)//1024}KB)")
    except Exception as e:
        print(f"  ❌ HF write failed: {e}")

# ── Main loop ────────────────────────────────────────────────
def main():
    start      = time.time()
    last_write = 0

    print(f"Collector started. Will run for {RUN_DURATION//3600}h.")
    print(f"Polling every {POLL_INTERVAL}s, writing to HF every {WRITE_INTERVAL}s.")

    # Accumulators — keep last 500 bars of each in memory
    m1_bars    = []
    m5_bars    = []
    depth_snap = None
    last_trade = None

    while True:
        elapsed = time.time() - start
        if elapsed >= RUN_DURATION:
            print("6-hour limit reached. Exiting.")
            break

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{now}] Polling Infoway...")

        # Fetch M1 — last 500 bars
        time.sleep(1.1)
        raw_m1 = fetch_kline(kline_type=1, kline_num=500)
        if raw_m1:
            m1_bars = raw_m1
            print(f"  M1: {len(m1_bars)} bars, last v={m1_bars[-1].get('v')}")

        # Fetch M5 — last 500 bars
        time.sleep(1.1)
        raw_m5 = fetch_kline(kline_type=2, kline_num=500)
        if raw_m5:
            m5_bars = raw_m5
            print(f"  M5: {len(m5_bars)} bars, last v={m5_bars[-1].get('v')}")

        # Fetch depth
        time.sleep(1.1)
        d = fetch_depth()
        if d:
            depth_snap = d
            print(f"  Depth: bid={d['b'][0][0] if d.get('b') else '?'}")

        # Fetch last trade
        time.sleep(1.1)
        lt = fetch_last_trade()
        if lt:
            last_trade = lt
            print(f"  Last trade: p={lt.get('p')} v={lt.get('v')}")

        # Write to HF every WRITE_INTERVAL seconds
        if (time.time() - last_write) >= WRITE_INTERVAL:
            write_to_hf(m1_bars, m5_bars, depth_snap, last_trade)
            last_write = time.time()

        # Sleep until next poll (minus the ~4.4s already spent on 4 requests)
        sleep_remaining = max(0, POLL_INTERVAL - 4.4)
        print(f"  Sleeping {sleep_remaining:.0f}s...")
        time.sleep(sleep_remaining)

if __name__ == "__main__":
    main()
