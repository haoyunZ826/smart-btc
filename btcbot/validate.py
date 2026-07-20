"""Validation harness.

The inherited repo searched ~150k configurations against one BTC history and
never held anything out, so every number it reports is in-sample. This module
exists to make that failure mode impossible here:

* `walk_forward`   — fit on a rolling window, score only on the untouched
                     window that follows, then step forward.
* `split_report`   — a single train/test cut for quick sanity checks.
* `monte_carlo`    — reshuffle trade order to see the drawdown distribution
                     the one realized path happened to avoid.
* `per_period`     — yearly and per-regime breakdown, to expose the case where
                     all the profit lives in two quarters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .backtest import Backtester, Costs, RiskConfig


@dataclass
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"train {self.train_start.date()}..{self.train_end.date()} "
                f"test {self.test_start.date()}..{self.test_end.date()}")


def make_windows(index: pd.DatetimeIndex, train_years: float = 3.0,
                 test_years: float = 1.0, step_years: float = 1.0) -> list[Window]:
    """Rolling train/test windows across the sample."""
    start, end = index[0], index[-1]
    train_td = pd.Timedelta(days=365 * train_years)
    test_td = pd.Timedelta(days=365 * test_years)
    step_td = pd.Timedelta(days=365 * step_years)

    windows = []
    cursor = start
    while cursor + train_td + test_td <= end:
        windows.append(Window(cursor, cursor + train_td,
                              cursor + train_td, cursor + train_td + test_td))
        cursor += step_td
    return windows


def _run(df, strategy, fine=None, funding=None, costs=None, risk=None, tf="4h"):
    return Backtester(df, strategy, costs=costs, risk=risk,
                      fine=fine, funding=funding, tf=tf).run()


def split_report(df, strategy, split: str, **kw) -> dict:
    """One train/test cut. `split` is a date string, e.g. '2023-01-01'."""
    cut = pd.Timestamp(split, tz="UTC")
    return {
        "in_sample": _run(df[df.index < cut], copy.deepcopy(strategy), **kw).stats(),
        "out_of_sample": _run(df[df.index >= cut], copy.deepcopy(strategy), **kw).stats(),
    }


def walk_forward(
    df: pd.DataFrame,
    build: Callable[[dict], object],
    grid: Iterable[dict],
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    select: str = "sharpe",
    warmup_bars: int = 250,
    **kw,
) -> dict:
    """Walk-forward optimization.

    For each window: score every config in `grid` on the TRAIN slice, take the
    winner by `select`, and run only that one on the TEST slice. Test results
    are stitched into a single out-of-sample record.

    `warmup_bars` of history is prepended to each slice so indicators are warm
    at the slice boundary — without it the first ~200 bars of every window are
    dead and the comparison is biased toward configs with short lookbacks.
    """
    windows = make_windows(df.index, train_years, test_years, step_years)
    if not windows:
        raise ValueError("sample too short for the requested window sizes")

    rows = []
    for w in windows:
        train = df[(df.index >= w.train_start) & (df.index < w.train_end)]
        test_from = df.index.searchsorted(w.test_start)
        test = df.iloc[max(0, test_from - warmup_bars):df.index.searchsorted(w.test_end)]

        best, best_score, best_stats = None, -np.inf, None
        for params in grid:
            stats = _run(train, build(params), **kw).stats()
            score = stats.get(select, -np.inf)
            # A config with almost no trades can post a flattering ratio.
            if stats.get("trades", 0) < 5 or not np.isfinite(score):
                continue
            if score > best_score:
                best, best_score, best_stats = params, score, stats

        if best is None:
            rows.append({"window": repr(w), "chosen": None, "note": "no viable config"})
            continue

        oos = _run(test, build(best), **kw)
        oos_stats = oos.stats()
        rows.append({
            "window": repr(w),
            "test_start": w.test_start,
            "test_end": w.test_end,
            "chosen": best,
            "is_score": best_score,
            "is_return": best_stats.get("total_return"),
            "oos_return": oos_stats.get("total_return"),
            "oos_sharpe": oos_stats.get("sharpe"),
            "oos_maxdd": oos_stats.get("max_drawdown"),
            "oos_trades": oos_stats.get("trades"),
            "oos_pf": oos_stats.get("profit_factor"),
        })

    frame = pd.DataFrame(rows)
    valid = frame.dropna(subset=["oos_return"]) if "oos_return" in frame else frame

    summary = {}
    if len(valid):
        compounded = float(np.prod(1 + valid["oos_return"].to_numpy()) - 1)
        summary = {
            "windows": int(len(valid)),
            "oos_compounded_return": compounded,
            "oos_mean_return": float(valid["oos_return"].mean()),
            "oos_median_return": float(valid["oos_return"].median()),
            "oos_positive_windows": int((valid["oos_return"] > 0).sum()),
            "oos_mean_sharpe": float(valid["oos_sharpe"].mean()),
            "oos_worst_dd": float(valid["oos_maxdd"].min()),
            # The gap between fitted and realized performance IS the overfit.
            "is_oos_gap": float(valid["is_return"].mean() - valid["oos_return"].mean()),
        }
    return {"windows": frame, "summary": summary}


def monte_carlo(result, n: int = 2000, seed: int = 7) -> dict:
    """Bootstrap the trade sequence to bound drawdown and ruin risk.

    The realized equity curve is one draw from the distribution of orderings.
    Resampling R-multiples with replacement answers "how bad could the same
    edge have looked with different luck?"
    """
    tf = result.trade_frame
    if tf.empty:
        return {}

    rng = np.random.default_rng(seed)
    returns = (tf.pnl / tf.equity_after.shift(1).fillna(result.initial)).to_numpy()
    returns = returns[np.isfinite(returns)]
    if not len(returns):
        return {}

    finals, dds = [], []
    for _ in range(n):
        path = rng.choice(returns, size=len(returns), replace=True)
        curve = result.initial * np.cumprod(1 + path)
        peak = np.maximum.accumulate(curve)
        finals.append(curve[-1])
        dds.append(float(((curve - peak) / peak).min()))

    finals, dds = np.array(finals), np.array(dds)
    return {
        "median_final": float(np.median(finals)),
        "p05_final": float(np.percentile(finals, 5)),
        "p95_final": float(np.percentile(finals, 95)),
        "median_maxdd": float(np.median(dds)),
        "p95_maxdd": float(np.percentile(dds, 5)),   # 5th pct = worst tail
        "prob_loss": float((finals < result.initial).mean()),
        "prob_ruin_50pct": float((dds < -0.5).mean()),
    }


def per_period(result, freq: str = "YE") -> pd.DataFrame:
    """PnL by calendar period, plus how concentrated the profit is."""
    tf = result.trade_frame
    if tf.empty:
        return pd.DataFrame()
    tf = tf.copy()
    tf["period"] = pd.to_datetime(tf.exit_time).dt.to_period(
        {"YE": "Y", "QE": "Q", "ME": "M"}[freq])
    grouped = tf.groupby("period").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        avg_r=("r_multiple", "mean"),
    )
    return grouped


def concentration(result) -> dict:
    """How much of the edge rides on a handful of trades.

    Measured in R-multiples, NOT dollars. On a compounding curve the late
    trades are mechanically the largest in cash terms, so a dollar-weighted
    concentration figure says more about when a trade happened than about how
    much it mattered. R is scale-free and comparable across the sample.
    """
    tf = result.trade_frame
    if tf.empty:
        return {}
    r = tf.r_multiple.sort_values(ascending=False).to_numpy()
    total = r.sum()
    if total <= 0:
        return {"total_r": float(total), "n_trades": int(len(r)),
                "note": "net negative in R"}
    n = len(r)
    return {
        "top1_share": float(r[0] / total),
        "top3_share": float(r[:3].sum() / total),
        "top5_share": float(r[:5].sum() / total),
        "top10pct_share": float(r[:max(1, n // 10)].sum() / total),
        "n_trades": int(n),
        "total_r": float(total),
    }
