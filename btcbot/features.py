"""Indicator layer.

Hand-rolled rather than ta-lib so the exact lookback semantics are visible —
every function here is causal: the value at bar T uses only bars <= T. The
backtest engine additionally shifts signals by one bar before acting on them,
so an off-by-one here cannot leak the future into a fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's ATR (the EMA-with-alpha=1/n form, not a simple mean)."""
    return true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Trend-strength. Used to separate trending regimes from chop."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def donchian(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Channel EXCLUDING the current bar.

    Shifting by one is what makes "close breaks the N-bar high" a real
    breakout test — without it the current bar's own high is part of the
    channel and the condition can never trigger cleanly.
    """
    upper = df["high"].shift(1).rolling(window, min_periods=window).max()
    lower = df["low"].shift(1).rolling(window, min_periods=window).min()
    return upper, lower


def realized_vol(s: pd.Series, window: int, bars_per_year: float) -> pd.Series:
    """Annualized realized volatility of log returns."""
    return np.log(s / s.shift(1)).rolling(window, min_periods=window).std() * np.sqrt(bars_per_year)


def zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    return (s - mean) / std.replace(0, np.nan)


BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def build(df: pd.DataFrame, tf: str = "4h") -> pd.DataFrame:
    """Standard feature set shared by every strategy in this project."""
    bpy = BARS_PER_YEAR[tf]
    out = df.copy()

    out["ema20"] = ema(df["close"], 20)
    out["ema50"] = ema(df["close"], 50)
    out["ema100"] = ema(df["close"], 100)
    out["ema200"] = ema(df["close"], 200)

    out["atr14"] = atr(df, 14)
    out["atr_pct"] = out["atr14"] / df["close"]
    out["rsi14"] = rsi(df["close"], 14)
    out["adx14"] = adx(df, 14)

    for w in (10, 20, 30, 55):
        up, lo = donchian(df, w)
        out[f"dc_up{w}"], out[f"dc_lo{w}"] = up, lo

    out["rv20"] = realized_vol(df["close"], 20, bpy)
    out["rv60"] = realized_vol(df["close"], 60, bpy)
    out["vol_ratio"] = out["rv20"] / out["rv60"]

    out["vol_z"] = zscore(df["volume"], 50)
    out["ret1"] = df["close"].pct_change()
    out["mom20"] = df["close"] / df["close"].shift(20) - 1
    out["mom50"] = df["close"] / df["close"].shift(50) - 1

    # Trend stack: the classic fast>slow>slowest alignment, as a 0/1 gate.
    out["trend_up"] = ((out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema100"])).astype(int)
    out["above_200"] = (df["close"] > out["ema200"]).astype(int)

    return out


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow — where the close sits in the bar, volume-weighted."""
    span = (df["high"] - df["low"]).replace(0, np.nan)
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / span * df["volume"]
    return mfv.rolling(window, min_periods=window).sum() / \
        df["volume"].rolling(window, min_periods=window).sum()


def dmi(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
    """+DI / -DI, the directional pair behind ADX."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / tr
    return plus_di, minus_di


def merge_daily(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach daily context to intraday bars without leaking the future.

    Every daily column is shifted one day before the merge, so a 4h bar inside
    day D only ever sees day D-1's completed daily values. Without that shift
    the 4h bars early in day D would be reading a daily close that has not
    happened yet — the single most common lookahead bug in multi-timeframe
    backtests.
    """
    d = pd.DataFrame(index=daily.index)
    d["d_close"] = daily["close"]
    d["d_ema50"] = ema(daily["close"], 50)
    d["d_ema200"] = ema(daily["close"], 200)
    d["d_rsi14"] = rsi(daily["close"], 14)
    d["d_ema50_prev3"] = d["d_ema50"].shift(3)
    d = d.shift(1).dropna(how="all")

    merged = pd.merge_asof(
        intraday.sort_index(),
        d.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


def daily_regime(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The inherited champion's `bull` / `bear` daily state flags."""
    bull = (
        (df["d_close"] > df["d_ema200"])
        & (df["d_ema50"] > df["d_ema200"])
        & (df["d_ema50"] > df["d_ema50_prev3"])
        & (df["d_rsi14"] > 52)
    ).fillna(False)
    bear = (
        (df["d_close"] < df["d_ema200"])
        & (df["d_ema50"] < df["d_ema200"])
        & (df["d_ema50"] < df["d_ema50_prev3"])
        & (df["d_rsi14"] < 48)
    ).fillna(False)
    return bull, bear


def regime(df: pd.DataFrame, adx_trend: float = 22.0) -> pd.Series:
    """Coarse regime label used for routing and for per-regime reporting.

    bull  — price above the 200EMA with a real trend reading
    bear  — price below the 200EMA with a real trend reading
    range — everything else (chop); this is where breakout systems bleed
    """
    trending = df["adx14"] >= adx_trend
    out = pd.Series("range", index=df.index, dtype=object)
    out[trending & (df["close"] > df["ema200"])] = "bull"
    out[trending & (df["close"] < df["ema200"])] = "bear"
    return out
