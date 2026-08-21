"""Build BTC OHLCV history, reusing whatever is already cached.

History comes from data.binance.vision bulk archives (monthly zips for closed
months, daily zips for the current month). Archives already present under
data/cache/ are never re-downloaded, so a normal run only pulls the days that
have appeared since the last run.

The recent-tail bars the archive lag has not published yet, and funding, come
from a REST source chain (`btcbot/sources.py`): Binance's own mirror first,
then OKX, then Kraken. That chain exists because api.binance.com answers 451
from some networks — and when it did, the old code let the exception kill the
whole run, archive bars included, leaving the monitor to publish 27-day-old
prices without ever raising an error.

Whichever source served each timeframe is written to data/fetch_provenance.json
and surfaced on the dashboard. Backup bars are close enough to be usable
(OKX BTC-USDT tracks Binance BTCUSDT within ~2.5bp) but "we quietly switched
exchanges" must never be something you have to infer from a chart.

Usage:
    python3 scripts/fetch_data.py                 # 1h, 4h, 1d + funding
    python3 scripts/fetch_data.py --tf 4h         # one timeframe
    python3 scripts/fetch_data.py --no-funding
    python3 scripts/fetch_data.py --check-sources # reachability probe, no writes
    python3 scripts/fetch_data.py --source okx-spot        # force one source
    python3 scripts/fetch_data.py --no-archive    # REST chain only (drill)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btcbot import sources as SRC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache" / "binance_klines"
PROVENANCE = DATA / "fetch_provenance.json"

VISION = "https://data.binance.vision/data/spot"

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

def fill_tail(symbol: str, tf: str, bars: dict[int, list[str]],
              only: str | None = None) -> dict:
    """Fill everything the archives have not published, via the source chain.

    Archive bars always win: the backup only supplies open_times the archives
    did not have, so a run that falls back to OKX cannot rewrite history that
    Binance already settled.
    """
    start = (max(bars) + TF_MS[tf]) if bars else int(GENESIS.timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    if start >= now_ms:
        print("  rest tail: nothing to fill")
        return {"source": "archive-only", "added": 0, "notes": []}

    try:
        fresh, source, notes = SRC.fetch_klines(symbol, tf, start, now_ms + TF_MS[tf], only)
    except Exception as exc:  # noqa: BLE001 - archives are still worth writing
        print(f"  rest tail: ALL SOURCES FAILED ({exc})")
        return {"source": "none", "added": 0, "notes": [str(exc)], "failed": True}

    added = 0
    for ms, row in fresh.items():
        if ms not in bars:
            bars[ms] = row
            added += 1
    for n in notes:
        print(f"    fell through — {n}")
    print(f"  rest tail: +{added} bars from {source}")
    return {"source": source, "added": added, "notes": notes}


# --------------------------------------------------------------------------

FUNDING_GENESIS = 1568102400000  # BTCUSDT perp listing, 2019-09-10


def fetch_funding(symbol: str, only: str | None = None) -> tuple[list[list], str, list[str]]:
    """Full funding history from whichever source answers.

    OKX's BTC-USDT-SWAP is the backup. Its funding is a different contract on a
    different venue, so the carry it implies is similar but not identical —
    which is exactly why the source gets recorded rather than assumed.
    """
    return SRC.fetch_funding(symbol, FUNDING_GENESIS, only)


def _read_csv(path: Path) -> dict[int, list[str]]:
    """Existing file keyed by its timestamp column, or {} if there is none."""
    if not path.exists():
        return {}
    out: dict[int, list[str]] = {}
    with path.open() as fh:
        for row in csv.reader(fh):
            if not row or not row[0].lstrip("-").isdigit():
                continue          # header
            out[int(row[0])] = row
    return out


def write_csv(path: Path, header: list[str], rows: list[list],
              overwrite_from: int | None = None) -> dict:
    """Merge `rows` into whatever the file already holds, then write.

    Overwriting outright was safe while every run rebuilt the full history from
    the Binance archives. It stopped being safe the moment a backup source could
    answer instead: OKX's spot history starts 2018-01-11 and its funding endpoint
    only reaches back about three months, so one fallback run would have silently
    amputated 2017 out of the klines and six years out of the funding curve — and
    a backtest with no funding before 2026 does not crash, it just quietly
    reports a strategy that never paid carry.

    Two rules, both about not letting a degraded run damage a good file:

    * **Settled history is never repainted.** Rows older than `overwrite_from`
      only fill gaps; a bar already on disk keeps the value the better source
      gave it. Otherwise a single fallback run would quietly restate eight years
      of Binance bars as OKX bars — a 0.5bp move per bar that no chart would
      ever show you, on data the backtest treats as ground truth.
    * **The live tail is always refreshed.** Rows at or after `overwrite_from`
      do get replaced, because the newest bar is still forming and last run's
      copy of it is genuinely stale. `overwrite_from=None` protects everything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv(path)
    merged = dict(existing)
    repainted = 0
    for r in rows:
        ms = int(r[0])
        if ms in merged:
            if overwrite_from is None or ms < overwrite_from:
                continue                    # settled — leave it alone
            repainted += 1
        merged[ms] = [str(x) for x in r]
    ordered = [merged[k] for k in sorted(merged)]

    if existing and (len(ordered) < len(existing) or min(merged) > min(existing)):
        raise RuntimeError(
            f"{path.name}: merge would lose history "
            f"({len(existing)} rows from {_iso(min(existing))} -> "
            f"{len(ordered)} rows from {_iso(min(merged))}) — refusing to write")

    with path.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for r in ordered:
            fh.write(",".join(str(x) for x in r) + "\n")

    span = f" {_iso(ordered[0][0])} -> {_iso(ordered[-1][0])}" if ordered else ""
    detail = ""
    if existing:
        detail = (f"  (+{len(ordered) - len(existing)} new, "
                  f"{len(existing)} kept, {repainted} tail bars refreshed)")
    print(f"  wrote {path.name}: {len(ordered)} rows{span}{detail}")
    return {"rows": len(ordered), "added": len(ordered) - len(existing),
            "tail_refreshed": repainted,
            "first": _iso(ordered[0][0]) if ordered else None}


def _iso(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def _iso_min(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="*", default=["1h", "4h", "1d"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--no-archive", action="store_true",
                    help="skip the Binance archives — drills the REST chain end to end")
    ap.add_argument("--source", default=None,
                    help="force one source (binance-vision-rest | okx-spot | kraken-spot)")
    ap.add_argument("--check-sources", action="store_true",
                    help="probe every source and exit without writing anything")
    args = ap.parse_args()

    if args.check_sources:
        for entry in SRC.check():
            newest = entry.pop("newest_bar", None)
            if newest:
                entry["newest_bar"] = _iso_min(newest)
            mark = "ok " if entry.pop("ok", False) else "DOWN"
            print(f"  [{mark}] {entry.pop('kind'):8s} {entry.pop('source'):22s} "
                  + " ".join(f"{k}={v}" for k, v in entry.items()))
        return

    prov = {"fetched_at": datetime.now(timezone.utc).isoformat(), "timeframes": {}}

    for tf in args.tf:
        print(f"{args.symbol} {tf}:")
        bars = {} if args.no_archive else collect_archive(args.symbol, tf)
        archive_bars = len(bars)
        tail = fill_tail(args.symbol, tf, bars, args.source)
        ordered = [bars[k] for k in sorted(bars)]
        if not ordered:
            # Refuse to replace a good CSV with nothing. A zero-row file makes
            # every downstream indicator NaN, which reads as a code bug for a
            # long time before anyone suspects the network.
            print(f"  no bars at all for {tf} — existing CSV left untouched")
            prov["timeframes"][tf] = {**tail, "archive_bars": 0, "written": False}
            continue
        # Only the last two bars are still in motion; everything older is settled.
        written = write_csv(DATA / f"{args.symbol}_{tf}.csv", COLUMNS, ordered,
                            overwrite_from=max(bars) - 2 * TF_MS[tf])
        prov["timeframes"][tf] = {
            **tail, **written, "archive_bars": archive_bars,
            "last_bar": _iso_min(max(bars)), "written": True,
        }

    if not args.no_funding:
        print("funding:")
        # An empty result must never overwrite the existing history — a zero-row
        # funding file reads as "carry is free", which is a lie the backtest
        # would happily price in.
        rows, source, notes = fetch_funding(args.symbol, args.source)
        for n in notes:
            print(f"    fell through — {n}")
        if rows:
            # A funding payment is settled the moment it is published, so nothing
            # here is ever repainted — overwrite_from stays None.
            write_csv(DATA / f"{args.symbol}_funding.csv",
                      ["fundingTime", "fundingRate"], rows)
            print(f"  funding source: {source}")
        else:
            print("  no funding rows returned — existing file left untouched")
        prov["funding"] = {"source": source, "rows": len(rows), "notes": notes,
                           "written": bool(rows)}

    PROVENANCE.write_text(json.dumps(prov, indent=2))
    print(f"wrote {PROVENANCE.name}")


if __name__ == "__main__":
    main()
