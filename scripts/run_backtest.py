"""Compare strategies on the full history with honest costs.

    python3 scripts/run_backtest.py
    python3 scripts/run_backtest.py --from 2023-01-01
    python3 scripts/run_backtest.py --no-fine     # skip intrabar resolution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcbot import data as D
from btcbot import strategies as S
from btcbot import validate as V
from btcbot.backtest import Backtester, Costs, RiskConfig


def build_all(daily):
    return [
        S.BuyHold(),
        S.DonchianTrend(),
        S.ChampionFaithful(daily=daily),
        S.DualEngineRegime(),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--no-fine", action="store_true")
    ap.add_argument("--no-funding", action="store_true")
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--leverage", type=float, default=3.0)
    args = ap.parse_args()

    df = D.load_ohlcv(args.tf)
    daily = D.load_ohlcv("1d")
    fine = None if args.no_fine else D.load_ohlcv("1h")
    funding = None if args.no_funding else D.load_funding()

    gaps = D.gap_report(df, args.tf)
    if len(gaps):
        print(f"data gaps: {len(gaps)} (largest {gaps.missing_bars.max()} bars)")

    if args.start:
        cut = pd.Timestamp(args.start, tz="UTC")
        df = df[df.index >= cut]
    if args.end:
        cut = pd.Timestamp(args.end, tz="UTC")
        df = df[df.index <= cut]

    print(f"\n{args.tf} bars: {len(df)}  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"costs: taker 5bps/side + 3bps slip (8bps on stops), funding "
          f"{'on' if funding is not None else 'off'}, "
          f"intrabar {'1h' if fine is not None else 'coarse'}\n")

    rows = []
    results = {}

    for strat in build_all(daily):
        spot = getattr(strat, "spot", False)
        risk = RiskConfig(initial_equity=args.equity,
                          max_leverage=1.0 if spot else args.leverage)
        bt = Backtester(df, strat,
                        costs=Costs(funding=(funding is not None) and not spot),
                        risk=risk, fine=fine,
                        funding=None if spot else funding, tf=args.tf)
        res = bt.run()
        results[strat.name] = res
        s = res.stats()
        rows.append({
            "strategy": strat.name,
            "trades": s.get("trades", 0),
            "return": s.get("total_return", 0),
            "cagr": s.get("cagr", 0),
            "maxDD": s.get("max_drawdown", 0),
            "sharpe": s.get("sharpe", 0),
            "calmar": s.get("calmar", 0),
            "PF": s.get("profit_factor", 0),
            "win": s.get("win_rate", 0),
            "liq": s.get("liquidations", 0),
        })

    frame = pd.DataFrame(rows).set_index("strategy")
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(frame.to_string())

    print("\nprofit concentration (share of net profit from the best trades):")
    for name, res in results.items():
        c = V.concentration(res)
        if c and "top1_share" in c:
            print(f"  {name:22s} top1={c['top1_share']:.0%} top3={c['top3_share']:.0%} "
                  f"top10%={c['top10pct_share']:.0%} n={c['n_trades']}")
        elif c:
            print(f"  {name:22s} {c.get('note', '')} (pnl={c.get('total_pnl', 0):,.0f})")

    return results


if __name__ == "__main__":
    main()
