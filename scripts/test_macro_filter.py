"""Does a daily macro trend filter earn its place?

Tested walk-forward across EVERY window, not just the recent bear. A filter
that only helps in the window that motivated it is a curve fit; a filter that
holds up across the whole sample is a real improvement.
"""

from __future__ import annotations

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

RISK, LEV = 0.03, 5.0


def evaluate(df, daily, funding, macro_ema, tf="4h"):
    params = {k: v for k, v in CHAMPION.items()
              if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    params["macro_daily_ema"] = macro_ema
    params["daily"] = daily

    windows = V.make_windows(df.index, 3, 1, 1)
    rows = []
    for w in windows:
        lo = max(0, df.index.searchsorted(w.test_start) - 250)
        test = df.iloc[lo:df.index.searchsorted(w.test_end)]
        s = Backtester(test, S.TrendCore(tf=tf, **params), costs=Costs(),
                       risk=RiskConfig(risk_per_trade=RISK, max_leverage=LEV),
                       funding=funding, tf=tf).run().stats()
        rows.append({
            "period": f"{w.test_start.date()}",
            "return": s.get("total_return", 0.0),
            "maxDD": s.get("max_drawdown", 0.0),
            "trades": s.get("trades", 0),
        })

    full = Backtester(df, S.TrendCore(tf=tf, **params), costs=Costs(),
                      risk=RiskConfig(risk_per_trade=RISK, max_leverage=LEV),
                      funding=funding, tf=tf).run()
    recent = df[df.index >= pd.Timestamp("2025-01-01", tz="UTC")]
    rec = Backtester(recent, S.TrendCore(tf=tf, **params), costs=Costs(),
                     risk=RiskConfig(risk_per_trade=RISK, max_leverage=LEV),
                     funding=funding, tf=tf).run().stats()
    return rows, full.stats(), rec


def main() -> None:
    df = D.load_ohlcv("4h")
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()

    print("Daily macro filter: require yesterday's daily close above its EMA(N)")
    print("(N=0 is the current champion, i.e. no macro filter)\n")

    summary = []
    for macro in [0, 50, 100, 150, 200]:
        rows, full, rec = evaluate(df, daily, funding, macro)
        oos = [r["return"] for r in rows]
        summary.append({
            "macro_ema": macro,
            "oos_compounded": float(np.prod([1 + r for r in oos]) - 1),
            "oos_median": float(np.median(oos)),
            "oos_won": int(sum(r > 0 for r in oos)),
            "oos_worst_dd": float(min(r["maxDD"] for r in rows)),
            "full_return": full.get("total_return", 0.0),
            "full_cagr": full.get("cagr", 0.0),
            "full_maxDD": full.get("max_drawdown", 0.0),
            "full_sharpe": full.get("sharpe", 0.0),
            "full_calmar": full.get("calmar", 0.0),
            "trades": full.get("trades", 0),
            "since2025_ret": rec.get("total_return", 0.0),
            "since2025_dd": rec.get("max_drawdown", 0.0),
        })

    out = pd.DataFrame(summary)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\nper-window out-of-sample returns:")
    for macro in [0, 200]:
        rows, _, _ = evaluate(df, daily, funding, macro)
        detail = "  ".join(f"{r['period']}:{r['return']:+.0%}" for r in rows)
        print(f"  macro={macro:3d}  {detail}")


if __name__ == "__main__":
    main()
