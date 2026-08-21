"""Where is the return-maximising position size, and what does it cost?

Written for one specific question: "I can tolerate a big drawdown for a bigger
return — how far up should I turn it?"

The answer is not "as far as you like". A leveraged system's return is not
monotonic in position size. Past a point, bigger bets make LESS money, because:

  * volatility drag — a -50% loss needs +100% to recover, so the arithmetic
    average bet size that maximises growth is finite (this is the Kelly point);
  * liquidation — the engine closes a position that touches its liquidation
    price and loses the margin, which is not a drawdown you recover from;
  * the leverage cap binds — past some risk-per-trade, notional is clamped to
    max_leverage x equity and extra "risk" does nothing but distort sizing.

So this sweeps position size for the baseline and for the filtered variant, and
reports return alongside the things that end accounts: liquidations, and the
Monte-Carlo probability of a drawdown you cannot trade your way out of.

Drawdown tolerance and ruin tolerance are different quantities. -48% is a bad
year. -85% means the remaining stake has to 7x just to get back to even, which
no amount of stoicism accomplishes.

    python3 scripts/experiment_risk_frontier.py
"""

from __future__ import annotations

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

CONFIGS = {
    "baseline": {},
    "filtered": {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25},
}
RISKS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
LEVS = [8.0, 15.0]


def build(daily, **ov):
    p = {k: v for k, v in CHAMPION.items()
         if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    p["daily"] = daily
    p.update(ov)
    return S.TrendCore(tf=CHAMPION["tf"], **p)


def run(df, daily, funding, ov, rp, lev):
    return Backtester(
        df, build(daily, **ov),
        costs=Costs(taker_fee=0.0005, slippage_bps=3.0,
                    stop_slippage_bps=8.0, funding=True),
        risk=RiskConfig(initial_equity=10_000.0, risk_per_trade=rp, max_leverage=lev),
        funding=funding, tf=CHAMPION["tf"],
    ).run()


def deep_dd_prob(result, threshold: float, n: int = 1500, seed: int = 11) -> float:
    """Share of reshuffled trade orderings whose max drawdown is worse than
    `threshold`. monte_carlo() only reports the 50% level."""
    rng = np.random.default_rng(seed)
    rs = np.array([t.r_multiple for t in result.trades], dtype=float)
    if len(rs) < 2:
        return float("nan")
    pnl = np.array([t.pnl for t in result.trades], dtype=float)
    eq0 = 10_000.0
    hits = 0
    for _ in range(n):
        order = rng.permutation(len(pnl))
        # Compound proportionally: each trade's return as a fraction of the
        # equity it was actually sized against.
        rel = pnl[order] / np.maximum(
            np.array([t.equity_after - t.pnl for t in result.trades])[order], 1e-9)
        curve = eq0 * np.cumprod(1 + rel)
        peak = np.maximum.accumulate(np.concatenate([[eq0], curve]))
        dd = (np.concatenate([[eq0], curve]) / peak - 1).min()
        if dd < threshold:
            hits += 1
    return hits / n


def main() -> None:
    df = D.load_ohlcv(CHAMPION["tf"])
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()
    windows = V.make_windows(df.index, 3.0, 1.0, 1.0)

    rows = []
    for lev in LEVS:
        for name, ov in CONFIGS.items():
            for rp in RISKS:
                res = run(df, daily, funding, ov, rp, lev)
                st = res.stats()
                mc = V.monte_carlo(res, n=1500, seed=11)
                oos = []
                for w in windows:
                    lo = max(0, df.index.searchsorted(w.test_start) - 250)
                    test = df.iloc[lo:df.index.searchsorted(w.test_end)]
                    oos.append(run(test, daily, funding, ov, rp, lev).stats()
                               .get("total_return", 0.0))
                rows.append({
                    "config": name, "lev": f"{lev:g}x", "risk%": rp * 100,
                    "full_return%": st["total_return"] * 100,
                    "CAGR%": st["cagr"] * 100,
                    "maxDD%": st["max_drawdown"] * 100,
                    "liquidations": st.get("liquidations", 0),
                    "OOS_compounded%": (np.prod([1 + r for r in oos]) - 1) * 100,
                    "MC_med_DD%": mc["median_maxdd"] * 100,
                    "MC_p(DD>50%)": mc["prob_ruin_50pct"],
                    "MC_p(DD>80%)": deep_dd_prob(res, -0.80),
                })

    f = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for lev in LEVS:
        print(f"\n=== 杠杆上限 {lev:g}x ===")
        sub = f[f.lev == f"{lev:g}x"].drop(columns=["lev"])
        print(sub.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    out = ROOT / "research" / "risk_frontier.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
