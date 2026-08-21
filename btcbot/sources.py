"""Market-data source chain: Binance first, then non-Binance backups.

Why this exists: `api.binance.com` / `fapi.binance.com` answer HTTP 451
(geo-block) from some networks. The bulk archives on data.binance.vision and
the `data-api.binance.vision` REST mirror still serve, but they are the same
operator — if Binance is unreachable for a different reason, both go with it.
So the fetcher needs somewhere else to go.

Design rules, in order of how badly violating them would hurt:

1. **A backup bar must be interchangeable with a Binance bar.** BTC-USDT on OKX
   closes within ~0.02% of BTCUSDT on Binance and shares the exact same UTC bar
   boundaries, so a gap filled from OKX does not move an EMA100 or a 30-bar
   Donchian level. Kraken quotes XBT/**USD**, not USDT, so it carries a stablecoin
   basis and sits last in the chain.
2. **Never mix venues silently.** Every fetch reports which source served it and
   the caller records that, because "the dashboard is quietly running on a
   different exchange" is exactly the kind of thing that looks like alpha decay.
3. **A source that answers wrongly is worse than one that fails.** OKX's `1D`
   bar is Hong-Kong-aligned (16:00 UTC boundaries); the UTC daily bar the macro
   gate needs is `1Dutc`. Getting this wrong shifts every daily close by 8h and
   the strategy would never notice.

All functions here are stdlib-only and return rows in Binance kline shape, so
callers cannot tell the difference apart from the returned source label.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

# OKX rejects the default urllib User-Agent with 403. This is not a rate limit
# and retrying with the same headers never succeeds.
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) smart_btc/1.0"}

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

BINANCE_MIRROR = "https://data-api.binance.vision/api/v3/klines"
OKX = "https://www.okx.com/api/v5"
KRAKEN = "https://api.kraken.com/0/public"

# Binance kline row shape every source is normalised into.
_N_COLS = 12


def _get(url: str, timeout: int = 30, retries: int = 3):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # 451/403 are verdicts, not weather — retrying cannot change them.
            if exc.code in (403, 451):
                raise
            last = exc
        except Exception as exc:  # noqa: BLE001 - transient network/ratelimit
            last = exc
        time.sleep(2 ** attempt)
    raise last if last else RuntimeError("unreachable")


def _row(open_ms: int, o, h, l, c, vol, tf: str, quote=0) -> list[str]:
    """Normalise to the Binance kline row the CSVs and loader expect."""
    close_ms = open_ms + TF_MS[tf] - 1
    return [str(open_ms), str(o), str(h), str(l), str(c), str(vol),
            str(close_ms), str(quote), "0", "0", "0", "0"]


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def from_binance(symbol: str, tf: str, start_ms: int, end_ms: int) -> dict[int, list[str]]:
    """Binance's own REST mirror on data.binance.vision (not geo-blocked)."""
    out: dict[int, list[str]] = {}
    cursor = start_ms
    while cursor < end_ms:
        qs = f"symbol={symbol}&interval={tf}&startTime={cursor}&limit=1000"
        batch = _get(f"{BINANCE_MIRROR}?{qs}")
        if not batch:
            break
        for r in batch:
            out[int(r[0])] = [str(x) for x in r][:_N_COLS]
        cursor = int(batch[-1][0]) + TF_MS[tf]
        time.sleep(0.2)
    return out


# OKX names the UTC-aligned daily bar `1Dutc`; plain `1D` is Hong Kong time.
_OKX_BAR = {"1h": "1H", "4h": "4H", "1d": "1Dutc"}


def from_okx(symbol: str, tf: str, start_ms: int, end_ms: int) -> dict[int, list[str]]:
    """OKX spot BTC-USDT — the primary backup.

    Same quote asset and the same UTC bar grid as Binance; measured 2026-08-20,
    closes track BTCUSDT within 0.5bp median / 2.5bp max across 1h, 4h and 1d.
    Depth stops at 2018-01-11, so it can carry the tail but not replace the
    2017 archives — callers must merge, never overwrite.
    """
    inst = symbol.replace("USDT", "-USDT")
    bar = _OKX_BAR[tf]
    out: dict[int, list[str]] = {}

    def absorb(rows) -> int:
        oldest = end_ms
        for r in rows:
            ms = int(r[0])
            oldest = min(oldest, ms)
            if start_ms <= ms < end_ms:
                # r = [ts, o, h, l, c, volBase, volQuote, volQuoteAlt, confirm]
                out[ms] = _row(ms, r[1], r[2], r[3], r[4], r[5], tf, r[6])
        return oldest

    # Newest window first (this endpoint includes the in-progress bar, which the
    # Binance path also returns — keeping them consistent matters more than
    # dropping it here).
    oldest = absorb(_get(f"{OKX}/market/candles?instId={inst}&bar={bar}&limit=300")["data"])

    # Then page backwards. `after` means "strictly older than this timestamp".
    while oldest > start_ms:
        data = _get(f"{OKX}/market/history-candles?instId={inst}&bar={bar}"
                    f"&after={oldest}&limit=100")["data"]
        if not data:
            break
        prev = oldest
        oldest = absorb(data)
        if oldest >= prev:      # cursor stopped moving — stop rather than spin
            break
        time.sleep(0.2)
    return out


_KRAKEN_PAIR = {"BTCUSDT": "XBTUSD"}
_KRAKEN_INTERVAL = {"1h": 60, "4h": 240, "1d": 1440}


def from_kraken(symbol: str, tf: str, start_ms: int, end_ms: int) -> dict[int, list[str]]:
    """Kraken XBT/USD — last resort.

    USD-quoted, so it carries a stablecoin basis vs BTCUSDT: measured 9.3bp
    median / 19bp max, roughly 20x OKX's error. Serves only the most recent
    ~720 bars. Good enough to keep the monitor alive, not good enough to
    backfill research data.
    """
    pair = _KRAKEN_PAIR.get(symbol, symbol)
    # Kraken rejects a negative `since` with EGeneral:Invalid arguments rather
    # than treating it as "from the beginning".
    since = max(0, start_ms // 1000 - 1)
    payload = _get(f"{KRAKEN}/OHLC?pair={pair}&interval={_KRAKEN_INTERVAL[tf]}"
                   f"&since={since}")
    if payload.get("error"):
        raise RuntimeError(f"kraken: {payload['error']}")
    series = next(v for k, v in payload["result"].items() if k != "last")
    out: dict[int, list[str]] = {}
    for r in series:
        ms = int(r[0]) * 1000
        if start_ms <= ms < end_ms:
            out[ms] = _row(ms, r[1], r[2], r[3], r[4], r[6], tf)
    return out


KLINE_SOURCES = [
    ("binance-vision-rest", from_binance),
    ("okx-spot", from_okx),
    ("kraken-spot", from_kraken),
]


def fetch_klines(symbol: str, tf: str, start_ms: int, end_ms: int,
                 only: str | None = None) -> tuple[dict[int, list[str]], str, list[str]]:
    """Walk the chain until one source returns bars.

    Returns (bars_by_open_time, source_used, notes). Raises only if every
    source failed — a bad primary must not take the whole run down with it,
    which is the failure mode this module was written for.
    """
    notes: list[str] = []
    for name, fn in KLINE_SOURCES:
        if only and name != only:
            continue
        try:
            bars = fn(symbol, tf, start_ms, end_ms)
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            notes.append(f"{name}: {type(exc).__name__} {exc}")
            continue
        if bars:
            return bars, name, notes
        notes.append(f"{name}: no bars in range")
    raise RuntimeError("all kline sources failed: " + " | ".join(notes))


# --------------------------------------------------------------------------
# funding
# --------------------------------------------------------------------------

FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"


def funding_from_binance(symbol: str, start_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    now_ms = int(time.time() * 1000)
    while cursor < now_ms:
        batch = _get(f"{FAPI}?symbol={symbol}&startTime={cursor}&limit=1000")
        if not batch:
            break
        rows.extend([[r["fundingTime"], r["fundingRate"]] for r in batch])
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    return rows


def funding_from_okx(symbol: str, start_ms: int) -> list[list]:
    """OKX BTC-USDT-SWAP funding history, paged backwards to start_ms.

    The endpoint only reaches back about three months (~283 rows measured),
    against Binance's full history from 2019. It is a gap-filler, not a
    replacement — writing it over the existing funding file would tell the
    backtest that carry was free for six years.
    """
    inst = symbol.replace("USDT", "-USDT") + "-SWAP"
    rows: dict[int, str] = {}
    cursor = int(time.time() * 1000)
    while cursor > start_ms:
        data = _get(f"{OKX}/public/funding-rate-history?instId={inst}"
                    f"&after={cursor}&limit=100")["data"]
        if not data:
            break
        prev = cursor
        for r in data:
            ms = int(r["fundingTime"])
            cursor = min(cursor, ms)
            if ms >= start_ms:
                rows[ms] = r["realizedRate"] or r["fundingRate"]
        if cursor >= prev:
            break
        time.sleep(0.2)
    return [[ms, rows[ms]] for ms in sorted(rows)]


FUNDING_SOURCES = [
    ("binance-fapi", funding_from_binance),
    ("okx-swap", funding_from_okx),
]


def fetch_funding(symbol: str, start_ms: int,
                  only: str | None = None) -> tuple[list[list], str, list[str]]:
    notes: list[str] = []
    for name, fn in FUNDING_SOURCES:
        if only and name != only:
            continue
        try:
            rows = fn(symbol, start_ms)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name}: {type(exc).__name__} {exc}")
            continue
        if rows:
            return rows, name, notes
        notes.append(f"{name}: no rows")
    return [], "none", notes


# --------------------------------------------------------------------------

def check() -> list[dict]:
    """Reachability probe for every source. Used by --check-sources."""
    now = int(time.time() * 1000)
    report = []
    for name, fn in KLINE_SOURCES:
        entry = {"source": name, "kind": "klines"}
        t0 = time.time()
        try:
            bars = fn("BTCUSDT", "4h", now - 3 * TF_MS["4h"], now + TF_MS["4h"])
            newest = max(bars) if bars else None
            entry.update(ok=True, bars=len(bars),
                         newest_bar=newest, ms=int((time.time() - t0) * 1000))
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        report.append(entry)
    for name, fn in FUNDING_SOURCES:
        entry = {"source": name, "kind": "funding"}
        try:
            rows = fn("BTCUSDT", now - 3 * 86_400_000)
            entry.update(ok=bool(rows), rows=len(rows))
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        report.append(entry)
    return report
