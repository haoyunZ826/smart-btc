"""Loading and aligning the raw Binance CSVs produced by scripts/fetch_data.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

_NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]


def load_ohlcv(tf: str = "4h", symbol: str = "BTCUSDT") -> pd.DataFrame:
    """OHLCV indexed by bar OPEN time (UTC), ascending, deduplicated.

    Indexing by open time keeps the no-lookahead rule easy to state: a bar
    stamped T is only knowable once T+tf has elapsed.
    """
    path = DATA / f"{symbol}_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/fetch_data.py first")

    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.drop(columns=["ignore"], errors="ignore")
        .drop_duplicates(subset="open_time", keep="last")
        .sort_values("open_time")
        .set_index("open_time")
    )

    _assert_sane(df, tf)
    return df


def _assert_sane(df: pd.DataFrame, tf: str) -> None:
    """Fail loudly on the data problems that silently corrupt a backtest."""
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(f"{tf}: NaNs in OHLC")
    bad = df[(df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])]
    if len(bad):
        raise ValueError(f"{tf}: {len(bad)} bars violate OHLC ordering, first at {bad.index[0]}")


def gap_report(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Missing-bar report. Binance has real outage gaps; we want them visible."""
    step = pd.Timedelta(tf)
    delta = df.index.to_series().diff()
    gaps = delta[delta > step]
    return pd.DataFrame({
        "gap_start": gaps.index - gaps.values,
        "gap_end": gaps.index,
        "missing_bars": (gaps / step).astype(int) - 1,
    }).reset_index(drop=True)


def load_funding(symbol: str = "BTCUSDT") -> pd.Series:
    """Perp funding rate series (8h cadence), indexed by funding timestamp."""
    path = DATA / f"{symbol}_funding.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return (
        df.drop_duplicates(subset="fundingTime")
        .sort_values("fundingTime")
        .set_index("fundingTime")["fundingRate"]
        .astype(float)
    )


def align_intrabar(coarse: pd.DataFrame, fine: pd.DataFrame, tf: str) -> dict:
    """Map each coarse bar to the fine bars inside it.

    Used by the engine to resolve stop-vs-target ordering within a bar instead
    of guessing. Returns {coarse_open_time: fine_slice_positions}.
    """
    step = pd.Timedelta(tf)
    bucket = fine.index.to_series().apply(lambda t: t.floor(step))
    groups = {}
    fine_pos = {t: i for i, t in enumerate(fine.index)}
    for coarse_t, times in bucket.groupby(bucket).groups.items():
        groups[coarse_t] = np.array([fine_pos[t] for t in times], dtype=np.int64)
    return {t: groups[t] for t in coarse.index if t in groups}
