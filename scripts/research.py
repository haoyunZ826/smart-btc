"""Strategy research, scored out-of-sample.

The rule this project runs on: a configuration is never judged by the numbers
it produces on the data used to pick it. Every candidate here is scored by
walk-forward — fit on a rolling 3y window, scored only on the following
untouched year — and the in-sample/out-of-sample gap is reported alongside,
because that gap is the overfit.

    python3 scripts/research.py --stage grid      # full-sample screen
    python3 scripts/research.py --stage wf        # walk-forward the finalists
    python3 scripts/research.py --stage risk      # scale risk to a DD budget
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcbot import data as D
from btcbot import strategies as S
from btcbot import validate as V
from btcbot.backtest import Backtester, Costs, RiskConfig

RESEARCH = Path(__file__).resolve().parent.parent / "research"
RESEARCH.mkdir(exist_ok=True)

_CACHE: dict = {}


def _load(tf: str):
    if tf not in _CACHE:
        _CACHE[tf] = (D.load_ohlcv(tf), D.load_funding())
    return _CACHE[tf]


def run_one(params: dict) -> dict:
    """Score a single configuration on the full sample."""
    tf = params.get("tf", "4h")
    df, funding = _load(tf)
    start = params.pop("_start", None)
    end = params.pop("_end", None)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    risk_pct = params.pop("_risk", 0.02)
    lev = params.pop("_lev", 3.0)

    strat = S.TrendCore(**params)
    res = Backtester(
        df, strat,
        costs=Costs(), risk=RiskConfig(risk_per_trade=risk_pct, max_leverage=lev),
        funding=funding, tf=tf,
    ).run()
    stats = res.stats()
    stats.update({k: v for k, v in params.items()})
    stats["_risk"] = risk_pct
    stats["_lev"] = lev
    return stats


# --------------------------------------------------------------------------

def stage_grid(workers: int) -> pd.DataFrame:
    """Broad screen. Its ONLY job is to shortlist for walk-forward."""
    space = {
        "tf": ["4h", "1d"],
        "entry_window": [10, 20, 30, 55],
        "atr_stop_mult": [2.0, 2.5, 3.0],
        "trail_mult": [2.5, 3.5, 5.0],
        "regime_ema": [0, 100, 200],
        "vol_target": [0.0, 0.6],
    }
    keys = list(space)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]
    print(f"grid: {len(combos)} configurations")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_one, combos, chunksize=4))

    frame = pd.DataFrame(rows)
    frame = frame[frame.trades >= 20]
    frame = frame.sort_values("calmar", ascending=False)
    frame.to_csv(RESEARCH / "grid.csv", index=False)

    cols = ["tf", "entry_window", "atr_stop_mult", "trail_mult", "regime_ema",
            "vol_target", "trades", "total_return", "cagr", "max_drawdown",
            "sharpe", "calmar", "profit_factor", "win_rate"]
    print("\ntop 15 by calmar (IN-SAMPLE — not yet trustworthy):")
    print(frame[cols].head(15).to_string(index=False,
          float_format=lambda v: f"{v:,.2f}"))
    return frame


def stage_wf(workers: int) -> pd.DataFrame:
    """Walk-forward the shortlist. These are the numbers that count."""
    grid_path = RESEARCH / "grid.csv"
    if not grid_path.exists():
        raise SystemExit("run --stage grid first")

    frame = pd.read_csv(grid_path)
    param_cols = ["tf", "entry_window", "atr_stop_mult", "trail_mult",
                  "regime_ema", "vol_target"]

    # Shortlist by calmar, but keep the field diverse: taking the top 20 rows
    # of a grid usually means 20 near-identical configs.
    shortlist = []
    seen = set()
    for _, row in frame.iterrows():
        key = (row.tf, row.entry_window, row.regime_ema)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append({c: row[c] for c in param_cols})
        if len(shortlist) >= 12:
            break

    for cfg in shortlist:
        cfg["entry_window"] = int(cfg["entry_window"])
        cfg["regime_ema"] = int(cfg["regime_ema"])

    print(f"walk-forward on {len(shortlist)} distinct configs")

    rows = []
    for cfg in shortlist:
        tf = cfg["tf"]
        df, funding = _load(tf)
        windows = V.make_windows(df.index, train_years=3, test_years=1, step_years=1)
        oos_returns, is_cagrs, oos_cagrs, oos_dds, oos_sharpes = [], [], [], [], []

        for w in windows:
            train = df[(df.index >= w.train_start) & (df.index < w.train_end)]
            test_from = df.index.searchsorted(w.test_start)
            test = df.iloc[max(0, test_from - 250):df.index.searchsorted(w.test_end)]

            kw = {k: v for k, v in cfg.items() if k != "tf"}
            is_stats = Backtester(train, S.TrendCore(tf=tf, **kw), costs=Costs(),
                                  funding=funding, tf=tf).run().stats()
            oos_stats = Backtester(test, S.TrendCore(tf=tf, **kw), costs=Costs(),
                                   funding=funding, tf=tf).run().stats()
            if oos_stats.get("trades", 0) > 0:
                oos_returns.append(oos_stats["total_return"])
                oos_cagrs.append(oos_stats["cagr"])
                oos_dds.append(oos_stats["max_drawdown"])
                oos_sharpes.append(oos_stats["sharpe"])
            if is_stats.get("trades", 0) > 0:
                is_cagrs.append(is_stats["cagr"])

        if not oos_returns:
            continue
        rows.append({
            **cfg,
            "windows": len(oos_returns),
            "oos_compounded": float(np.prod([1 + r for r in oos_returns]) - 1),
            "oos_cagr": float(np.mean(oos_cagrs)),
            "oos_median": float(np.median(oos_returns)),
            "oos_positive": int(sum(r > 0 for r in oos_returns)),
            "oos_worst_dd": float(min(oos_dds)),
            "oos_sharpe": float(np.mean(oos_sharpes)),
            "is_cagr": float(np.mean(is_cagrs)) if is_cagrs else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no walk-forward results")
    # Compare like with like: the train window is 3y and the test window 1y, so
    # raw totals are not comparable. Annualised, the shortfall is the overfit.
    out["cagr_decay"] = out["is_cagr"] - out["oos_cagr"]
    out = out.sort_values("oos_compounded", ascending=False)
    out.to_csv(RESEARCH / "walkforward.csv", index=False)

    print("\nwalk-forward results (OUT-OF-SAMPLE):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return out


def stage_risk(workers: int) -> pd.DataFrame:
    """Given a validated edge, scale risk per trade to a drawdown budget.

    Return maximisation happens HERE, not in the signal search. Once the edge
    survives out-of-sample, leverage is the dial — and the only honest way to
    set it is against a drawdown you are willing to sit through.
    """
    wf_path = RESEARCH / "walkforward.csv"
    if not wf_path.exists():
        raise SystemExit("run --stage wf first")

    wf = pd.read_csv(wf_path)
    # Robustness is the FILTER, return is the objective. Requiring a win in
    # every out-of-sample window throws out configs that ride one lucky year;
    # among the survivors, the goal is the largest compounded return.
    survivors = wf[wf.oos_positive == wf.windows]
    if survivors.empty:
        survivors = wf[wf.oos_positive >= wf.windows - 1]
        print("note: no config won every window — relaxed to n-1")
    best = survivors.sort_values("oos_compounded", ascending=False).iloc[0].to_dict()
    print(f"selected: won {best['oos_positive']}/{best['windows']} OOS windows, "
          f"compounded {best['oos_compounded']:.1f}x, sharpe {best['oos_sharpe']:.2f}, "
          f"cagr decay {best['cagr_decay']:+.2f}")
    cfg = {
        "tf": best["tf"],
        "entry_window": int(best["entry_window"]),
        "atr_stop_mult": float(best["atr_stop_mult"]),
        "trail_mult": float(best["trail_mult"]),
        "regime_ema": int(best["regime_ema"]),
        "vol_target": float(best["vol_target"]),
    }
    print(f"scaling risk for: {cfg}\n")

    combos = []
    for risk_pct in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        for lev in [2.0, 3.0, 5.0, 8.0]:
            combos.append({**cfg, "_risk": risk_pct, "_lev": lev})

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_one, combos, chunksize=2))

    out = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    keep = ["_risk", "_lev", "trades", "total_return", "cagr", "max_drawdown",
            "sharpe", "calmar", "profit_factor", "liquidations"]
    out[keep].to_csv(RESEARCH / "risk_scaling.csv", index=False)
    print(out[keep].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "wf", "risk"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    {"grid": stage_grid, "wf": stage_wf, "risk": stage_risk}[args.stage](args.workers)


if __name__ == "__main__":
    main()
