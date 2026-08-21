"""Third pass: is the risk-matched winner robust, or one lucky window?

A compounded out-of-sample number is a product, so a single 700% window can
carry a variant that loses everywhere else. This script breaks the OOS result
per window, then reshuffles trade order (Monte Carlo) to ask what the drawdown
distribution looks like — the check the README calls the most important number
in the project, and the one the inherited repo never ran.

    python3 scripts/experiment_filters_robustness.py
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
from btcbot.config import CHAMPION

ROOT = Path(__file__).resolve().parent.parent

# risk/trade taken from the risk-matched pass (same full-sample drawdown)
CONFIGS = {
    "baseline (aggressive)":     (0.0500, {}),
    "+dvol 1.00x":               (0.0561, {"min_dvol_ratio": 1.00}),
    "+daily SMA200":             (0.0471, {"macro_daily_sma": 200}),
    "+thrust.50 +dvol1.25":      (0.0604, {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25}),
    "all three":                 (0.1125, {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25,
                                           "macro_daily_sma": 200}),
    "all three @5%":             (0.0500, {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25,
                                           "macro_daily_sma": 200}),
}


def build(daily, **ov):
    p = {k: v for k, v in CHAMPION.items()
         if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    p["daily"] = daily
    p.update(ov)
    return S.TrendCore(tf=CHAMPION["tf"], **p)


def run(df, strat, funding, rp):
    return Backtester(
        df, copy.deepcopy(strat),
        costs=Costs(taker_fee=0.0005, slippage_bps=3.0,
                    stop_slippage_bps=8.0, funding=True),
        risk=RiskConfig(initial_equity=10_000.0, risk_per_trade=rp, max_leverage=8.0),
        funding=funding, tf=CHAMPION["tf"],
    ).run()


def main() -> None:
    df = D.load_ohlcv(CHAMPION["tf"])
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()
    windows = V.make_windows(df.index, 3.0, 1.0, 1.0)
    labels = [str(w.test_start.date()) for w in windows]

    per, mc_rows = {}, {}
    for name, (rp, ov) in CONFIGS.items():
        rets, trades = [], []
        for w in windows:
            lo = max(0, df.index.searchsorted(w.test_start) - 250)
            test = df.iloc[lo:df.index.searchsorted(w.test_end)]
            s = run(test, build(daily, **ov), funding, rp).stats()
            rets.append(round(s.get("total_return", 0.0) * 100, 1))
            trades.append(s.get("trades", 0))
        per[name] = rets + [sum(trades)]

        full = run(df, build(daily, **ov), funding, rp)
        mc = V.monte_carlo(full, n=2000, seed=7)
        mc_rows[name] = {k: (round(v * 100, 1) if isinstance(v, float) and abs(v) <= 50 else
                             round(v, 3) if isinstance(v, float) else v)
                         for k, v in mc.items()}

    print("=== OOS RETURN % PER WINDOW (risk-matched) ===")
    print(pd.DataFrame(per, index=labels + ["trades"]).T.to_string())

    print("\n=== MONTE CARLO (2000 trade-order reshuffles, full sample) ===")
    print(pd.DataFrame(mc_rows).T.to_string())

    p = ROOT / "research" / "entry_filter_robustness.json"
    p.write_text(json.dumps({"per_window": per, "labels": labels,
                             "monte_carlo": mc_rows}, indent=2, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
