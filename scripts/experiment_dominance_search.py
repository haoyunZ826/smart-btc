"""Is there a configuration that beats the baseline on BOTH return and drawdown?

"Higher return at lower drawdown" is a dominance claim, and dominance is easy to
fake. Two ways this experiment refuses to fake it:

* **Position size is calibrated on TRAIN ONLY.** An earlier pass in this repo
  sized every variant to match the baseline's FULL-SAMPLE drawdown, then scored
  it out of sample. That leaks: a filter that removes a 2018 drawdown earns a
  bigger risk budget, then spends it on 2021-2026. Here the sizing bisection and
  the config ranking both see only 2017-08 -> 2020-08, and the score comes only
  from 2020-08 -> 2026-08.

* **Out-of-sample equity compounds CONTINUOUSLY.** The rolling-window metric
  restarts at $10,000 every year, so it structurally cannot represent ruin — the
  30%-risk config that liquidates to -$3.77 scores highest on it. One unbroken
  run from 2020 to 2026 shows both the return and the drawdown that actually
  happened to a single account.

Selection discipline: the grid is round numbers only, configs are ranked by TRAIN
Calmar, and only the top handful are scored out of sample. How many of them
dominate is reported alongside which — one lucky winner out of many is a
different claim from a broad region that works.

    python3 scripts/experiment_dominance_search.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcbot import data as D
from btcbot import strategies as S
from btcbot.backtest import Backtester, Costs, RiskConfig
from btcbot.config import CHAMPION

ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = "2020-08-16"          # first out-of-sample window start
WARMUP = 400


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


# Round numbers only. Every value here is a plain choice someone would make
# without a computer; nothing is tuned to a decimal.
GRID = {
    "entry_window":   [20, 30, 55],
    "atr_stop_mult":  [2.0, 3.0],
    "trail_mult":     [3.0, 5.0, 7.0],
    "vol_target":     [0.4, 0.6, 0.8],
    "min_thrust_atr": [0.0, 0.5],
    "min_dvol_ratio": [0.0, 1.25],
}


def main() -> None:
    df = D.load_ohlcv(CHAMPION["tf"])
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()
    cut = pd.Timestamp(TRAIN_END, tz="UTC")
    i_cut = df.index.searchsorted(cut)
    train = df.iloc[:i_cut]
    oos = df.iloc[max(0, i_cut - WARMUP):]

    base_ov, base_rp, base_lev = {}, 0.05, 8.0
    b_train = run(train, daily, funding, base_ov, base_rp, base_lev).stats()
    b_oos = run(oos, daily, funding, base_ov, base_rp, base_lev).stats()
    print(f"训练段 2017-08 → {TRAIN_END}   基线: 收益 {b_train['total_return']*100:,.0f}%  "
          f"回撤 {b_train['max_drawdown']*100:.1f}%  Calmar {b_train['calmar']:.2f}")
    print(f"样本外 {TRAIN_END} → 2026-08  基线: 收益 {b_oos['total_return']*100:,.0f}%  "
          f"回撤 {b_oos['max_drawdown']*100:.1f}%  交易 {b_oos['trades']}\n")

    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"网格 {len(combos)} 组，只在训练段打分…")

    scored = []
    for ov in combos:
        st = run(train, daily, funding, ov, base_rp, base_lev).stats()
        if st["trades"] < 10:            # too few to rank on
            continue
        scored.append((st["calmar"], st, ov))
    scored.sort(key=lambda x: -x[0])
    print(f"可评分 {len(scored)} 组，取训练段 Calmar 前 8 名进入样本外\n")

    target_dd = b_train["max_drawdown"]

    def size_to_train_dd(ov, lev):
        lo, hi = 0.002, 0.60
        for _ in range(20):
            mid = (lo + hi) / 2
            dd = run(train, daily, funding, ov, mid, lev).stats()["max_drawdown"]
            if dd < target_dd:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    rows = []
    for calmar, st, ov in scored[:8]:
        rp = size_to_train_dd(ov, base_lev)
        o = run(oos, daily, funding, ov, rp, base_lev).stats()
        rows.append({
            "config": ", ".join(f"{k}={v}" for k, v in ov.items() if v),
            "train_calmar": round(calmar, 2),
            "risk%": round(rp * 100, 2),
            "OOS_return%": round(o["total_return"] * 100, 0),
            "OOS_maxDD%": round(o["max_drawdown"] * 100, 1),
            "OOS_trades": o["trades"],
            "OOS_calmar": round(o["calmar"], 2),
            "liq": o.get("liquidations", 0),
            "碾压基线": (o["total_return"] > b_oos["total_return"]
                     and o["max_drawdown"] > b_oos["max_drawdown"]),
        })

    frame = pd.DataFrame(rows)
    print("=== 样本外（连续复利，一个账户从头跑到尾）===")
    print(f"基线对照: 收益 {b_oos['total_return']*100:,.0f}%  回撤 {b_oos['max_drawdown']*100:.1f}%  "
          f"Calmar {b_oos['calmar']:.2f}\n")
    print(frame.to_string(index=False))
    n = int(frame["碾压基线"].sum())
    print(f"\n同时做到「收益更高 + 回撤更浅」的：{n}/{len(frame)}")

    out = ROOT / "research" / "dominance_search.json"
    out.write_text(json.dumps({
        "baseline_train": b_train, "baseline_oos": b_oos,
        "grid_size": len(combos), "results": rows,
    }, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
