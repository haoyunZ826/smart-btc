"""如果从熊市底部开机，这套策略跑成什么样？

问题本身带着一个必须说破的前提：**底部只有事后才知道是底部**。在 2018-12-15 那天，
没有任何东西告诉你 $3,212 是终点而不是中继。所以这不是一个可执行的策略，是一个
「假如择时完美」的情景——它会同时抬高策略和买入持有的成绩，因此唯一有意义的读数是
**两者的差**，不是策略的绝对收益。

实现上有一个坑值得写下来：指标需要预热。如果窗口就从底部那天开始切，前 100~200 根
bar 的 EMA100/ATR 是 NaN，策略被迫空转半个月，这会把「策略反应慢」这个结论凭空做出来。
所以特征在**完整历史**上计算，再切片喂给引擎——features.py 里每个指标都是因果的
（bar T 只用 <=T 的数据），所以这样做不引入前视，只是恢复了「你在底部那天开机时，
指标本来就是热的」这个事实。

    python3 scripts/experiment_bear_bottom_start.py
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
from btcbot.backtest import Backtester, Costs, RiskConfig
from btcbot.config import CHAMPION

ROOT = Path(__file__).resolve().parent.parent


class PreWarmed(S.TrendCore):
    """在完整历史上算特征，再按窗口切片——预热不丢，前视不引入。"""

    full: pd.DataFrame | None = None
    _cache: pd.DataFrame | None = None

    def prepare(self, df):
        if self._cache is None:
            object.__setattr__(self, "_cache", super().prepare(self.full))
        return self._cache.loc[df.index]


def build(full, daily, **ov):
    p = {k: v for k, v in CHAMPION.items()
         if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    p["daily"] = daily
    p.update(ov)
    s = PreWarmed(tf=CHAMPION["tf"], **p)
    object.__setattr__(s, "full", full)
    object.__setattr__(s, "_cache", None)
    return s


def run(window, strat, funding, rp, lev):
    return Backtester(
        window, copy.deepcopy(strat),
        costs=Costs(taker_fee=0.0005, slippage_bps=3.0,
                    stop_slippage_bps=8.0, funding=True),
        risk=RiskConfig(initial_equity=10_000.0, risk_per_trade=rp, max_leverage=lev),
        funding=funding, tf=CHAMPION["tf"],
    ).run()


VARIANTS = {
    "基线突破 (aggressive)": ({}, 0.05, 8.0),
    "真突破 thr0.5+dv1.25":  ({"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25}, 0.05, 8.0),
    "真突破 @6%/8x":         ({"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25}, 0.06, 8.0),
    "真突破 @15%/15x":       ({"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25}, 0.15, 15.0),
}


def bottoms(daily):
    c = daily["close"]
    dd = c / c.cummax() - 1
    out, in_bear = [], False
    for t, v in dd.items():
        if not in_bear and v <= -0.50:
            in_bear, tt, tv = True, t, v
        elif in_bear:
            if v < tv:
                tv, tt = v, t
            if v > -0.10:
                out.append((tt, float(c[tt]), tv)); in_bear = False
    if in_bear:
        out.append((tt, float(c[tt]), tv))
    return out


def main() -> None:
    df = D.load_ohlcv(CHAMPION["tf"])
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()
    bots = bottoms(daily)

    report = {}
    for k, (bt_date, bt_px, depth) in enumerate(bots, 1):
        start = bt_date  # already tz-aware from the daily index
        win = df[df.index >= start]
        if len(win) < 200:
            continue
        end_px = float(win["close"].iloc[-1])
        bh = end_px / float(win["close"].iloc[0]) - 1
        # 买入持有的回撤，同区间
        eq = win["close"] / float(win["close"].iloc[0])
        bh_dd = float((eq / eq.cummax() - 1).min())

        print("=" * 96)
        print(f"熊市底部 #{k}: {bt_date.date()} @ ${bt_px:,.0f} "
              f"(自峰顶 {depth*100:.1f}%)   → 至 {win.index[-1].date()}，"
              f"{(win.index[-1]-start).days} 天")
        print(f"  买入持有: {bh*100:>12,.0f}%   最大回撤 {bh_dd*100:6.1f}%")
        rows = {}
        for lbl, (ov, rp, lev) in VARIANTS.items():
            res = run(win, build(df, daily, **ov), funding, rp, lev)
            st = res.stats()
            tf_ = res.trade_frame
            first = tf_.iloc[0] if len(tf_) else None
            lag = (first.entry_time - start).days if first is not None else None
            above = (first.entry_price / bt_px - 1) * 100 if first is not None else None
            rows[lbl] = {
                "收益%": st["total_return"] * 100, "回撤%": st["max_drawdown"] * 100,
                "年化%": st["cagr"] * 100, "Calmar": st["calmar"], "笔数": st["trades"],
                "胜率%": st["win_rate"] * 100, "爆仓": st.get("liquidations", 0),
                "首笔延迟(天)": lag, "首笔离底(%)": above,
                "跑赢买入持有": st["total_return"] > bh,
            }
        f = pd.DataFrame(rows).T
        print(f.to_string(float_format=lambda v: f"{v:,.1f}"))
        print()
        report[str(bt_date.date())] = {
            "bottom_price": bt_px, "depth": depth, "buy_hold": bh,
            "buy_hold_maxdd": bh_dd, "variants": rows,
        }

    out = ROOT / "research" / "bear_bottom_start.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
