"""Second pass: compare the filters at MATCHED risk, and on 2026 itself.

The first pass compared every variant at the same 5% risk-per-trade, which is
not a fair fight: a filter that takes half as many trades also carries half the
exposure, so it "wins" on drawdown for a reason that has nothing to do with
signal quality. The honest question is: at the SAME drawdown, does the filter
deliver more return?

So each variant gets its risk-per-trade scaled until its full-sample max
drawdown lands on the baseline's, and only then are returns compared.

Part 2 replays 2026 to answer the question that started this: would the filters
have skipped the three losing trades and kept the current one?

    python3 scripts/experiment_filters_risk_matched.py
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


def build(daily, **ov):
    p = {k: v for k, v in CHAMPION.items()
         if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    p["daily"] = daily
    p.update(ov)
    return S.TrendCore(tf=CHAMPION["tf"], **p)


def run(df, strat, funding, risk_pct, lev=8.0):
    return Backtester(
        df, copy.deepcopy(strat),
        costs=Costs(taker_fee=0.0005, slippage_bps=3.0,
                    stop_slippage_bps=8.0, funding=True),
        risk=RiskConfig(initial_equity=10_000.0, risk_per_trade=risk_pct,
                        max_leverage=lev),
        funding=funding, tf=CHAMPION["tf"],
    ).run()


VARIANTS = {
    "baseline (aggressive)":      {},
    "+thrust 0.50 ATR":           {"min_thrust_atr": 0.50},
    "+dvol 1.00x":                {"min_dvol_ratio": 1.00},
    "+dvol 1.25x":                {"min_dvol_ratio": 1.25},
    "+daily SMA200":              {"macro_daily_sma": 200},
    "+thrust 0.50 +dvol 1.25":    {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25},
    "all three (0.50/1.25/200)":  {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25,
                                   "macro_daily_sma": 200},
}


def match_risk(df, daily, funding, ov, target_dd, lo=0.005, hi=0.40):
    """Bisect risk_per_trade until full-sample max drawdown hits target_dd."""
    strat = build(daily, **ov)
    for _ in range(22):
        mid = (lo + hi) / 2
        dd = run(df, strat, funding, mid).stats().get("max_drawdown", -1.0)
        if dd < target_dd:      # deeper than target -> too much risk
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main() -> None:
    df = D.load_ohlcv(CHAMPION["tf"])
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()

    base_dd = run(df, build(daily), funding, 0.05).stats()["max_drawdown"]
    print(f"baseline full-sample max drawdown = {base_dd*100:.1f}%  "
          f"(every variant is scaled to this)\n")

    windows = V.make_windows(df.index, 3.0, 1.0, 1.0)
    rows = {}
    for name, ov in VARIANTS.items():
        rp = 0.05 if not ov else match_risk(df, daily, funding, ov, base_dd)
        st = run(df, build(daily, **ov), funding, rp).stats()
        rets, dds = [], []
        for w in windows:
            lo = max(0, df.index.searchsorted(w.test_start) - 250)
            test = df.iloc[lo:df.index.searchsorted(w.test_end)]
            s = run(test, build(daily, **ov), funding, rp).stats()
            rets.append(s.get("total_return", 0.0))
            dds.append(s.get("max_drawdown", 0.0))
        rows[name] = {
            "risk/trade %": round(rp * 100, 2),
            "trades": st["trades"],
            "full return %": round(st["total_return"] * 100, 0),
            "full maxDD %": round(st["max_drawdown"] * 100, 1),
            "CAGR %": round(st["cagr"] * 100, 1),
            "Sharpe": round(st["sharpe"], 2),
            "Calmar": round(st["calmar"], 2),
            "OOS compounded %": round((np.prod([1 + r for r in rets]) - 1) * 100, 0),
            "OOS +win": f"{int(sum(r>0 for r in rets))}/{len(rets)}",
            "OOS worst DD %": round(min(dds) * 100, 1),
        }
    print("=== RISK-MATCHED COMPARISON (same full-sample drawdown) ===")
    print(pd.DataFrame(rows).T.to_string())

    # ---------------- part 2: what happens in 2026 ----------------
    print("\n=== 2026 PAPER PERIOD: which entries does each variant take? ===")
    cut = pd.Timestamp("2026-01-01", tz="UTC")
    lo = max(0, df.index.searchsorted(cut) - 400)
    win = df.iloc[lo:]
    out = {}
    for name, ov in VARIANTS.items():
        res = run(win, build(daily, **ov), funding, 0.05)
        tf_ = res.trade_frame
        tf_ = tf_[tf_.entry_time >= cut] if len(tf_) else tf_
        taken = [(str(t.entry_time)[:16], round(float(t.r_multiple), 2), t.reason)
                 for _, t in tf_.iterrows()]
        out[name] = taken
        st = res.stats()
        print(f"\n{name}:  {len(taken)} 笔")
        for e, r, why in taken:
            print(f"    {e}  {r:+.2f}R  {why}")

    p = ROOT / "research" / "entry_filter_risk_matched.json"
    p.write_text(json.dumps({"risk_matched": rows, "trades_2026": out},
                            indent=2, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
