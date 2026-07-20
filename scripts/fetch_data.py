"""Build BTC OHLCV history from Binance, reusing whatever is already cached.

Data comes from data.binance.vision bulk archives (monthly zips for closed
months, daily zips for the current month). Archives already present under
data/cache/ are never re-downloaded, so a normal run only pulls the days that
have appeared since the last run. The recent-tail bars that the archive lag
has not published yet come from the REST API.

Funding is pulled from the futures REST API (the repo's cached funding file
was a 20-row stub, not usable for carry costs).

Usage:
    python3 scripts/fetch_data.py                 # 1h, 4h, 1d + funding
    python3 scripts/fetch_data.py --tf 4h         # one timeframe
    python3 scripts/fetch_data.py --no-funding
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache" / "binance_klines"

VISION = "https://data.binance.vision/data/spot"
SPOT_API = "https://api.binance.com/api/v3/klines"
FAPI_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"

# BTCUSDT spot listed 2017-08-17; the vision archives start 2017-08.
GENESIS = datetime(2017, 8, 1, tzinfo=timezone.utc)

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


# --------------------------------------------------------------------------
# archive download (cached)
# --------------------------------------------------------------------------

def _fetch_zip(url: str, dest: Path) -> bytes | None:
    """Return zip bytes, downloading only if not already cached.

    A 404 means that shard is not published (future month, or a day the
    archive has not caught up to yet) — that is expected, not an error.
    """
    if dest.exists():
        return dest.read_bytes()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return blob


# Binance switched the archive timestamp unit from milliseconds to
# MICROSECONDS partway through 2025. Mixing the two silently produces bars
# dated in the year 58518, so every timestamp is normalised on the way in.
# Milliseconds "now" are ~1.8e12; microseconds "now" are ~1.8e15.
_US_THRESHOLD = 1e14


def _to_ms(raw: str) -> int:
    v = int(raw)
    return v // 1000 if v > _US_THRESHOLD else v


def _rows_from_zip(blob: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode()
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    # Newer archives ship a header line; older ones do not.
    if rows and not rows[0][0].lstrip("-").isdigit():
        rows = rows[1:]
    for r in rows:
        r[0] = str(_to_ms(r[0]))   # open_time
        r[6] = str(_to_ms(r[6]))   # close_time
    return rows


def _months(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur < datetime(end.year, end.month, 1, tzinfo=timezone.utc):
        yield cur
        cur = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=timezone.utc)


def _days(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def collect_archive(symbol: str, tf: str) -> dict[int, list[str]]:
    """All bars available from bulk archives, keyed by open_time (dedup)."""
    now = datetime.now(timezone.utc)
    bars: dict[int, list[str]] = {}
    downloaded = 0

    for m in _months(GENESIS, now):
        tag = m.strftime("%Y-%m")
        dest = CACHE / symbol / tf / "monthly" / f"{tag}.zip"
        cached = dest.exists()
        blob = _fetch_zip(f"{VISION}/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{tag}.zip", dest)
        if blob is None:
            continue
        if not cached:
            downloaded += 1
        for r in _rows_from_zip(blob):
            bars[int(r[0])] = r

    # Current (and previous, if the monthly shard has not landed) month: dailies.
    for d in _days(now.replace(day=1) - timedelta(days=31), now):
        tag = d.strftime("%Y-%m-%d")
        dest = CACHE / symbol / tf / "daily" / f"{tag}.zip"
        cached = dest.exists()
        blob = _fetch_zip(f"{VISION}/daily/klines/{symbol}/{tf}/{symbol}-{tf}-{tag}.zip", dest)
        if blob is None:
            continue
        if not cached:
            downloaded += 1
        for r in _rows_from_zip(blob):
            bars[int(r[0])] = r

    print(f"  archive: {len(bars)} bars ({downloaded} new shards downloaded)")
    return bars


# --------------------------------------------------------------------------
# REST tail (the last day or two the archive has not published yet)
# --------------------------------------------------------------------------

def _api(url: str, params: dict, retries: int = 5):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{url}?{qs}", timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 - transient network/ratelimit
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"    retry {attempt + 1}: {exc}")
    return []


def fill_tail(symbol: str, tf: str, bars: dict[int, list[str]]) -> int:
    start = (max(bars) + TF_MS[tf]) if bars else int(GENESIS.timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    added = 0
    while start < now_ms:
        batch = _api(SPOT_API, {"symbol": symbol, "interval": tf, "startTime": start, "limit": 1000})
        if not batch:
            break
        for r in batch:
            bars[int(r[0])] = [str(x) for x in r]
        added += len(batch)
        start = batch[-1][0] + TF_MS[tf]
        time.sleep(0.2)
    print(f"  rest tail: +{added} bars")
    return added


# --------------------------------------------------------------------------

def fetch_funding(symbol: str) -> list[list]:
    rows: list[list] = []
    start = 1568102400000  # BTCUSDT perp listing, 2019-09-10
    now_ms = int(time.time() * 1000)
    while start < now_ms:
        batch = _api(FAPI_FUNDING, {"symbol": symbol, "startTime": start, "limit": 1000})
        if not batch:
            break
        rows.extend([[r["fundingTime"], r["fundingRate"]] for r in batch])
        start = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    span = ""
    if rows:
        span = f" {_iso(rows[0][0])} -> {_iso(rows[-1][0])}"
    print(f"  wrote {path.name}: {len(rows)} rows{span}")


def _iso(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="*", default=["1h", "4h", "1d"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--no-funding", action="store_true")
    args = ap.parse_args()

    for tf in args.tf:
        print(f"{args.symbol} {tf}:")
        bars = collect_archive(args.symbol, tf)
        fill_tail(args.symbol, tf, bars)
        ordered = [bars[k] for k in sorted(bars)]
        write_csv(DATA / f"{args.symbol}_{tf}.csv", COLUMNS, ordered)

    if not args.no_funding:
        print("funding:")
        write_csv(DATA / f"{args.symbol}_funding.csv",
                  ["fundingTime", "fundingRate"], fetch_funding(args.symbol))


if __name__ == "__main__":
    main()
