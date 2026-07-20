"""Engine correctness tests.

These target the failure modes that make a backtest look good and trade badly:
lookahead, free fills, ignored costs, and impossible survival under leverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btcbot import features as F
from btcbot.backtest import Backtester, Costs, RiskConfig
from btcbot.strategies import Strategy


def make_bars(closes, highs=None, lows=None, tf="4h"):
    n = len(closes)
    idx = pd.date_range("2020-01-01", periods=n, freq=tf, tz="UTC")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes,
        "high": closes if highs is None else np.asarray(highs, dtype=float),
        "low": closes if lows is None else np.asarray(lows, dtype=float),
        "close": closes,
        "volume": np.full(n, 100.0),
    }, index=idx)


class EnterAt(Strategy):
    """Enters once at a chosen bar with an explicit stop."""

    def __init__(self, bar, stop, tf="4h", targets=None, notional_frac=None):
        self.bar, self.stop, self.tf = bar, stop, tf
        self.targets = targets or []
        self.notional_frac = notional_frac
        self.name = "test"

    def prepare(self, df):
        return df

    def entry(self, sig, i):
        if i != self.bar:
            return None
        spec = {"direction": 1, "stop": self.stop, "targets": list(self.targets)}
        if self.notional_frac:
            spec["notional_frac"] = self.notional_frac
        return spec


# --------------------------------------------------------------------------
# no lookahead
# --------------------------------------------------------------------------

def test_entry_fills_at_next_bar_open_not_signal_bar_close():
    """A signal seen at the close of bar i must fill at bar i+1's open."""
    df = make_bars([100, 100, 100, 200, 200])
    df.loc[df.index[3], "open"] = 150.0     # the fill price we expect

    bt = Backtester(df, EnterAt(bar=2, stop=50.0),
                    costs=Costs(slippage_bps=0, taker_fee=0, funding=False),
                    risk=RiskConfig(initial_equity=10_000))
    bt.run()
    trade = bt.trades[0] if bt.trades else None
    assert trade is not None
    # 150 (bar 3 open), never 100 (bar 2 close) and never 200 (bar 3 close).
    assert trade.entry_price == pytest.approx(150.0)


def test_future_bars_cannot_change_a_past_fill():
    """Truncating the sample must not alter trades that already closed."""
    closes = [100, 105, 110, 115, 120, 125, 130]
    strat = lambda: EnterAt(bar=1, stop=90.0)  # noqa: E731

    long_run = Backtester(make_bars(closes), strat(),
                          costs=Costs(funding=False)).run()
    short_run = Backtester(make_bars(closes[:5]), strat(),
                           costs=Costs(funding=False)).run()

    assert long_run.equity.iloc[:5].round(6).tolist() == \
        short_run.equity.round(6).tolist()


def test_indicators_are_causal():
    """Changing a future bar must not change any indicator value before it."""
    closes = list(np.linspace(100, 200, 300))
    a = F.build(make_bars(closes), "4h")

    tampered = closes.copy()
    tampered[-1] = 10_000.0
    b = F.build(make_bars(tampered), "4h")

    cols = ["ema20", "ema50", "atr14", "rsi14", "adx14", "dc_up20", "mom20"]
    for col in cols:
        pd.testing.assert_series_equal(
            a[col].iloc[:-1], b[col].iloc[:-1], check_names=False,
            obj=f"{col} leaked the final bar",
        )


def test_donchian_excludes_current_bar():
    """The channel must not contain the bar being tested against it."""
    df = make_bars([10, 20, 30, 40, 50], highs=[10, 20, 30, 40, 50])
    up, _ = F.donchian(df, 2)
    # At bar 3 the 2-bar channel covers bars 1..2 -> max(20, 30) = 30.
    assert up.iloc[3] == pytest.approx(30.0)
    assert df["high"].iloc[3] > up.iloc[3]     # a real breakout is detectable


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------

def test_fees_and_slippage_reduce_pnl():
    df = make_bars([100, 100, 100, 100, 100])

    free = Backtester(df, EnterAt(bar=1, stop=50.0, notional_frac=1.0),
                      costs=Costs(taker_fee=0, slippage_bps=0, funding=False)).run()
    costed = Backtester(df, EnterAt(bar=1, stop=50.0, notional_frac=1.0),
                        costs=Costs(taker_fee=0.0005, slippage_bps=3, funding=False)).run()

    # Flat price: costless round trip breaks even, a costed one must lose.
    assert free.equity.iloc[-1] == pytest.approx(10_000, rel=1e-9)
    assert costed.equity.iloc[-1] < 10_000


def test_funding_is_charged_to_longs():
    n = 30
    df = make_bars([100] * n)
    funding = pd.Series(
        0.001,
        index=pd.date_range("2020-01-01", periods=n * 2, freq="8h", tz="UTC"),
    )

    with_f = Backtester(df, EnterAt(bar=1, stop=50.0, notional_frac=1.0),
                        costs=Costs(taker_fee=0, slippage_bps=0, funding=True),
                        funding=funding).run()
    assert with_f.equity.iloc[-1] < 10_000
    assert with_f.trades[0].funding_paid > 0 if with_f.trades else True


# --------------------------------------------------------------------------
# exits
# --------------------------------------------------------------------------

def test_stop_wins_when_stop_and_target_share_a_bar():
    """Same-bar ambiguity must resolve pessimistically."""
    df = make_bars([100, 100, 100], highs=[100, 100, 130], lows=[100, 100, 80])

    bt = Backtester(df, EnterAt(bar=0, stop=90.0, targets=[(120.0, 1.0)]),
                    costs=Costs(taker_fee=0, slippage_bps=0,
                                stop_slippage_bps=0, funding=False))
    bt.run()
    assert bt.trades[0].reason == "stop"
    assert bt.trades[0].pnl < 0


def test_liquidation_closes_before_the_stop():
    """A stop below the liquidation price cannot save the position."""
    df = make_bars([100, 100, 100], highs=[100, 100, 100], lows=[100, 100, 40])

    bt = Backtester(df, EnterAt(bar=0, stop=30.0, notional_frac=3.0),
                    costs=Costs(taker_fee=0, slippage_bps=0,
                                stop_slippage_bps=0, funding=False),
                    risk=RiskConfig(initial_equity=10_000, max_leverage=3.0))
    bt.run()
    assert bt.trades[0].reason == "liquidation"


def test_equity_never_goes_negative_under_leverage():
    """A -60% gap with 3x leverage should liquidate, not owe money."""
    df = make_bars([100, 100, 40, 40], highs=[100, 100, 100, 40],
                   lows=[100, 100, 40, 40])
    bt = Backtester(df, EnterAt(bar=0, stop=1.0, notional_frac=3.0),
                    costs=Costs(funding=False),
                    risk=RiskConfig(initial_equity=10_000, max_leverage=3.0))
    res = bt.run()
    assert res.equity.min() >= 0


def test_partial_targets_scale_out():
    df = make_bars([100, 100, 100, 100], highs=[100, 100, 100, 150],
                   lows=[100, 100, 100, 100])
    bt = Backtester(df, EnterAt(bar=0, stop=90.0,
                                targets=[(110.0, 0.5), (120.0, 0.5)],
                                notional_frac=1.0),
                    costs=Costs(taker_fee=0, slippage_bps=0, funding=False))
    bt.run()
    assert bt.trades[0].reason == "target_final"
    assert bt.trades[0].pnl > 0


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

def test_stats_on_empty_result():
    df = make_bars([100] * 5)

    class Never(Strategy):
        tf = "4h"
        name = "never"

        def prepare(self, df):
            return df

    res = Backtester(df, Never(), costs=Costs(funding=False)).run()
    assert res.stats()["trades"] == 0


def test_r_multiple_matches_realized_loss_on_a_stop():
    """A clean stop-out should land at roughly -1R."""
    df = make_bars([100, 100, 100], highs=[100, 100, 100], lows=[100, 100, 89])
    bt = Backtester(df, EnterAt(bar=0, stop=90.0),
                    costs=Costs(taker_fee=0, slippage_bps=0,
                                stop_slippage_bps=0, funding=False),
                    risk=RiskConfig(initial_equity=10_000, risk_per_trade=0.02))
    bt.run()
    assert bt.trades[0].r_multiple == pytest.approx(-1.0, abs=0.02)
