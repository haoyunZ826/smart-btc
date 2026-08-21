"""Final validation of the selected strategy.

Produces research/final_report.json, which is also what the dashboard reads.
Everything here is a check the inherited repo never ran on its champion.
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
from btcbot.config import CHAMPION, RISK_PROFILES

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"


def run(df, strat, funding, risk_pct, lev, spot=False, tf="4h"):
    return Backtester(
        df, strat,
        costs=Costs(funding=not spot),
        risk=RiskConfig(risk_per_trade=risk_pct, max_leverage=1.0 if spot else lev),
        funding=None if spot else funding,
        tf=tf,
    ).run()


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def main() -> None:
    tf = CHAMPION["tf"]
    df = D.load_ohlcv(tf)
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()

    params = {k: v for k, v in CHAMPION.items() if k not in ("tf", "risk_per_trade",
                                                             "max_leverage", "name")}
    params["daily"] = daily
    risk_pct = CHAMPION["risk_per_trade"]
    lev = CHAMPION["max_leverage"]

    report: dict = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data": {
            "symbol": "BTCUSDT",
            "timeframe": tf,
            "bars": int(len(df)),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
        },
        "config": jsonable(CHAMPION),
        "costs": {"taker_fee_bps": 5, "slippage_bps": 3, "stop_slippage_bps": 8,
                  "funding": True},
    }

    # ---- headline comparison -------------------------------------------
    print("=" * 78)
    print("FULL SAMPLE", df.index[0].date(), "->", df.index[-1].date())
    print("=" * 78)

    contenders = {
        "smart_btc (this project)": (S.TrendCore(tf=tf, **params), risk_pct, lev, False),
        "inherited champion": (S.ChampionFaithful(tf=tf, daily=daily), 0.02, 3.0, False),
        "plain donchian": (S.DonchianTrend(tf=tf), 0.02, 3.0, False),
        "buy & hold (spot)": (S.BuyHold(tf=tf), 0.02, 1.0, True),
    }

    rows, results = [], {}
    for label, (strat, rp, lv, spot) in contenders.items():
        res = run(df, strat, funding, rp, lv, spot=spot, tf=tf)
        results[label] = res
        s = res.stats()
        rows.append({
            "strategy": label, "trades": s["trades"], "return": s["total_return"],
            "cagr": s["cagr"], "maxDD": s["max_drawdown"], "sharpe": s["sharpe"],
            "calmar": s["calmar"], "PF": s["profit_factor"], "win": s["win_rate"],
        })

    table = pd.DataFrame(rows).set_index("strategy")
    print(table.to_string(float_format=lambda v: f"{v:,.2f}"))
    report["comparison"] = jsonable(
        {r["strategy"]: {k: v for k, v in r.items() if k != "strategy"} for r in rows})

    main_res = results["smart_btc (this project)"]

    # ---- walk-forward ---------------------------------------------------
    print("\n" + "=" * 78)
    print("WALK-FORWARD (out-of-sample only)")
    print("=" * 78)
    windows = V.make_windows(df.index, 3, 1, 1)
    wf_rows = []
    for w in windows:
        lo = max(0, df.index.searchsorted(w.test_start) - 250)
        test = df.iloc[lo:df.index.searchsorted(w.test_end)]
        s = run(test, S.TrendCore(tf=tf, **params), funding, risk_pct, lev, tf=tf).stats()
        bh = run(test, S.BuyHold(tf=tf), funding, 0.02, 1.0, spot=True, tf=tf).stats()
        wf_rows.append({
            "test_period": f"{w.test_start.date()}..{w.test_end.date()}",
            "trades": s.get("trades", 0),
            "return": s.get("total_return", 0.0),
            "maxDD": s.get("max_drawdown", 0.0),
            "sharpe": s.get("sharpe", 0.0),
            "buy_hold": bh.get("total_return", 0.0),
            "beat_bh": s.get("total_return", 0) > bh.get("total_return", 0),
        })
    wf = pd.DataFrame(wf_rows)
    print(wf.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\nwindows won: {int((wf['return'] > 0).sum())}/{len(wf)}   "
          f"beat buy&hold: {int(wf['beat_bh'].sum())}/{len(wf)}")
    report["walk_forward"] = jsonable(wf.to_dict("records"))

    # ---- monte carlo ----------------------------------------------------
    print("\n" + "=" * 78)
    print("MONTE CARLO (2000 reshuffles of the trade sequence)")
    print("=" * 78)
    mc = V.monte_carlo(main_res, n=2000)
    for k, v in mc.items():
        print(f"  {k:20s} {v:,.4f}")
    report["monte_carlo"] = jsonable(mc)

    # ---- per-year -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PER YEAR")
    print("=" * 78)
    yearly = V.per_period(main_res, "YE")
    print(yearly.to_string(float_format=lambda v: f"{v:,.2f}"))
    report["per_year"] = jsonable(
        {str(k): jsonable(v) for k, v in yearly.to_dict("index").items()})

    # ---- concentration --------------------------------------------------
    print("\n" + "=" * 78)
    print("PROFIT CONCENTRATION")
    print("=" * 78)
    for label, res in results.items():
        c = V.concentration(res)
        if "top1_share" in c:
            print(f"  {label:28s} top1={c['top1_share']:.0%} top3={c['top3_share']:.0%} "
                  f"top10%={c['top10pct_share']:.0%}")
    report["concentration"] = jsonable(V.concentration(main_res))

    # ---- risk profiles --------------------------------------------------
    print("\n" + "=" * 78)
    print("RISK PROFILES (position size; `filtered` also changes the entry rule)")
    print("=" * 78)
    prof_rows = []
    for name, cfg in RISK_PROFILES.items():
        # A profile may override entry parameters, not just size. Ignoring that
        # here would print a table where `filtered` shows the unfiltered result.
        pp = {**params, **(cfg.get("strategy") or {})}
        s = run(df, S.TrendCore(tf=tf, **pp), funding,
                cfg["risk_per_trade"], cfg["max_leverage"], tf=tf).stats()
        prof_rows.append({
            "profile": name, "risk": cfg["risk_per_trade"], "lev": cfg["max_leverage"],
            "filters": ", ".join(f"{k}={v}" for k, v in (cfg.get("strategy") or {}).items()) or "-",
            "return": s["total_return"], "cagr": s["cagr"], "maxDD": s["max_drawdown"],
            "sharpe": s["sharpe"], "calmar": s["calmar"],
        })
    profiles = pd.DataFrame(prof_rows).set_index("profile")
    print(profiles.to_string(float_format=lambda v: f"{v:,.2f}"))
    report["risk_profiles"] = jsonable(
        {r["profile"]: {k: v for k, v in r.items() if k != "profile"} for r in prof_rows})

    # ---- equity curve for the dashboard ---------------------------------
    curve = main_res.equity.resample("1D").last().dropna()
    bh_curve = results["buy & hold (spot)"].equity.resample("1D").last().dropna()
    report["equity_curve"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in curve.index],
        "strategy": [round(float(v), 2) for v in curve.values],
        "buy_hold": [round(float(v), 2) for v in
                     bh_curve.reindex(curve.index).ffill().fillna(0).values],
    }
    report["trades"] = jsonable(
        main_res.trade_frame.assign(
            entry_time=lambda d: d.entry_time.astype(str),
            exit_time=lambda d: d.exit_time.astype(str),
        ).to_dict("records"))

    RESEARCH.mkdir(exist_ok=True)
    (RESEARCH / "final_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESEARCH / 'final_report.json'}")


if __name__ == "__main__":
    main()
