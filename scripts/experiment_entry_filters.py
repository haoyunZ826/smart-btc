"""Do the 2026 entry-quality differences survive nine years of history?

Four paper trades in 2026 separated on three things: breakout thrust in ATR,
prior-day volume vs its 20d mean, and whether price was above the daily SMA200.
Four observations is not evidence — it is a hypothesis. This script is the test.

Method, chosen so a filter cannot flatter itself:

* Every variant is a FIXED config. Nothing is grid-selected per window, so there
  is no selection bias to launder — the out-of-sample slices are scored by the
  same rule that was written down in advance.
* The out-of-sample slices are the walk-forward test windows from
  validate.make_windows(), with warmup prepended so indicators are live at each
  boundary. Baseline and variant see byte-identical slices.
* Thresholds are round numbers (0.25/0.5/0.75 ATR, 1.0/1.25/1.5x volume). No
  decimals were tuned. If a filter only works at 0.37 ATR it is noise.
* A filter that improves the full sample but not the out-of-sample windows is
  reported as REJECTED, not as an improvement.

    python3 scripts/experiment_entry_filters.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcbot import data as D
from btcbot import strategies as S
from btcbot import validate as V
from btcbot.backtest import Backtester, Costs, RiskConfig
from btcbot.config import CHAMPION, RISK_PROFILES

ROOT = Path(__file__).resolve().parent.parent
PROFILE = "aggressive"


def build(daily, **overrides):
    params = {k: v for k, v in CHAMPION.items()
              if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    params["daily"] = daily
    params.update(overrides)
    return S.TrendCore(tf=CHAMPION["tf"], **params)


def run(df, strat, funding):
    cfg = RISK_PROFILES[PROFILE]
    return Backtester(
        df, copy.deepcopy(strat),
        costs=Costs(taker_fee=0.0005, slippage_bps=3.0,
                    stop_slippage_bps=8.0, funding=True),
        risk=RiskConfig(initial_equity=10_000.0,
                        risk_per_trade=cfg["risk_per_trade"],
                        max_leverage=cfg["max_leverage"]),
        funding=funding, tf=CHAMPION["tf"],
    ).run()


VARIANTS = {
    "baseline (aggressive)":      {},
    "+thrust 0.25 ATR":           {"min_thrust_atr": 0.25},
    "+thrust 0.50 ATR":           {"min_thrust_atr": 0.50},
    "+thrust 0.75 ATR":           {"min_thrust_atr": 0.75},
    "+dvol 1.00x":                {"min_dvol_ratio": 1.00},
    "+dvol 1.25x":                {"min_dvol_ratio": 1.25},
    "+dvol 1.50x":                {"min_dvol_ratio": 1.50},
    "+daily SMA200":              {"macro_daily_sma": 200},
    "+thrust 0.50 +dvol 1.25":    {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25},
    "+thrust 0.50 +SMA200":       {"min_thrust_atr": 0.50, "macro_daily_sma": 200},
    "all three (0.50/1.25/200)":  {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25,
                                   "macro_daily_sma": 200},
}

KEYS = ("trades", "total_return", "cagr", "max_drawdown", "sharpe",
        "calmar", "profit_factor", "win_rate", "expectancy_r", "exposure")


def main() -> None:
    df = D.load_ohlcv(CHAMPION["tf"])
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()
    print(f"sample: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df):,} bars)\n")

    # ---------------- full sample ----------------
    full = {}
    for name, ov in VARIANTS.items():
        st = run(df, build(daily, **ov), funding).stats()
        full[name] = {k: st.get(k) for k in KEYS}
    frame = pd.DataFrame(full).T
    print("=== FULL SAMPLE (in-sample — treat as a sanity check, not evidence) ===")
    show = frame.copy()
    for c in ("total_return", "cagr", "max_drawdown", "win_rate", "exposure"):
        show[c] = (show[c] * 100).round(1)
    print(show.round(2).to_string())

    # ---------------- out-of-sample windows ----------------
    windows = V.make_windows(df.index, train_years=3.0, test_years=1.0, step_years=1.0)
    print(f"\n=== OUT-OF-SAMPLE: {len(windows)} rolling 1-year test windows ===")
    print("(fixed configs, identical slices, no per-window selection)\n")

    oos = {}
    per_window = {}
    for name, ov in VARIANTS.items():
        rets, dds, trades, sharpes = [], [], [], []
        for w in windows:
            lo = max(0, df.index.searchsorted(w.test_start) - 250)
            test = df.iloc[lo:df.index.searchsorted(w.test_end)]
            st = run(test, build(daily, **ov), funding).stats()
            rets.append(st.get("total_return", 0.0))
            dds.append(st.get("max_drawdown", 0.0))
            trades.append(st.get("trades", 0))
            sharpes.append(st.get("sharpe", np.nan))
        per_window[name] = rets
        oos[name] = {
            "oos_compounded": float(np.prod([1 + r for r in rets]) - 1),
            "oos_mean": float(np.mean(rets)),
            "oos_median": float(np.median(rets)),
            "positive_windows": f"{int(sum(r > 0 for r in rets))}/{len(rets)}",
            "worst_window_dd": float(np.min(dds)),
            "mean_sharpe": float(np.nanmean(sharpes)),
            "total_trades": int(sum(trades)),
        }
    o = pd.DataFrame(oos).T
    for c in ("oos_compounded", "oos_mean", "oos_median", "worst_window_dd"):
        o[c] = (o[c].astype(float) * 100).round(1)
    print(o.round(2).to_string())

    print("\n=== PER-WINDOW OOS RETURN % (baseline vs each variant) ===")
    labels = [f"{w.test_start.date()}" for w in windows]
    pw = pd.DataFrame({k: [round(r * 100, 1) for r in v]
                       for k, v in per_window.items()}, index=labels)
    print(pw.T.to_string())

    out = ROOT / "research" / "entry_filter_experiment.json"
    out.write_text(json.dumps({
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "sample": {"start": str(df.index[0]), "end": str(df.index[-1]), "bars": len(df)},
        "profile": PROFILE,
        "variants": {k: v for k, v in VARIANTS.items()},
        "full_sample": full,
        "out_of_sample": oos,
        "per_window": {k: v for k, v in per_window.items()},
        "windows": labels,
    }, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
