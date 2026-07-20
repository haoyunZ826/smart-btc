"""Strategy definitions.

Contract with the engine (btcbot/backtest.py):

    prepare(df)          -> feature frame, computed once
    entry(sig, i)        -> dict(direction, stop, targets) or None, read at the
                            CLOSE of bar i, filled at the OPEN of bar i+1
    manage(pos, sig, i, bt) -> mutate the open position (trailing, breakeven)

Everything here is long-biased on purpose. The inherited repo tested six
short families across ten bear quarters and the best one still lost money,
so shorts are off by default rather than "disabled for now".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F


class Strategy:
    name = "base"

    # Venue shape. Spot strategies own the coin: no leverage cap beyond 1x and
    # no funding to pay. Perp strategies inherit the runner's leverage and are
    # charged funding for every 8h they stay exposed.
    spot = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return F.build(df, self.tf)

    def entry(self, sig: pd.DataFrame, i: int):
        return None

    def manage(self, pos, sig: pd.DataFrame, i: int, bt) -> None:
        return None


# --------------------------------------------------------------------------

@dataclass
class BuyHold(Strategy):
    """Spot buy-and-hold. The benchmark every other strategy must beat.

    Unleveraged and unfunded on purpose: this is what the user could have
    done with zero effort, so beating it has to mean beating it net of the
    costs a leveraged system actually pays.
    """

    tf: str = "4h"
    name: str = "buy_hold"
    spot: bool = True

    def entry(self, sig, i):
        if i < 1:
            return None
        # Full equity, one position, no stop — held to the end of the sample.
        return {"direction": 1, "stop": sig["close"].iloc[i] * 1e-6,
                "notional_frac": 1.0, "targets": []}


@dataclass
class DonchianTrend(Strategy):
    """Long-only Donchian breakout with an ATR chandelier trail.

    The plainest trend-following system that exists, included as the honest
    baseline the fancier variants have to beat. Parameters are round numbers,
    not search results.
    """

    tf: str = "4h"
    name: str = "donchian_trend"
    entry_window: int = 20
    atr_stop_mult: float = 2.5
    trail_mult: float = 3.0
    trend_filter: bool = True
    breakeven_at_r: float = 1.0

    def prepare(self, df):
        sig = F.build(df, self.tf)
        up, _ = F.donchian(df, self.entry_window)
        sig["entry_level"] = up
        return sig

    def entry(self, sig, i):
        row = sig.iloc[i]
        if np.isnan(row["entry_level"]) or np.isnan(row["atr14"]):
            return None
        if self.trend_filter and not (row["close"] > row["ema200"]):
            return None
        if row["close"] <= row["entry_level"]:
            return None
        stop = row["close"] - self.atr_stop_mult * row["atr14"]
        if stop <= 0:
            return None
        return {"direction": 1, "stop": stop, "targets": []}

    def manage(self, pos, sig, i, bt):
        row = sig.iloc[i]
        atr = row["atr14"]
        if np.isnan(atr):
            return
        risk = abs(pos.entry_price - pos.initial_stop)

        # Lift to breakeven once the trade has paid for itself.
        if self.breakeven_at_r and not pos.trail_active:
            if (row["close"] - pos.entry_price) >= self.breakeven_at_r * risk:
                pos.stop = max(pos.stop, pos.entry_price)
                pos.trail_active = True

        # Chandelier: trail from the highest close seen since entry.
        trail = row["close"] - self.trail_mult * atr
        pos.stop = max(pos.stop, trail)


@dataclass
class TrendCore(Strategy):
    """The research workhorse: a parameterised long-only trend system.

    Deliberately built from a small number of well-understood parts, each with
    a reason to exist rather than a backtest that liked it:

      entry   Donchian breakout, optionally requiring a trend stack
      filter  regime gate (price vs a slow EMA) to stay out of bear chop
      size    fixed fractional risk, optionally scaled by realised volatility
      exit    ATR chandelier trail, optional breakeven lift, optional TP

    Volatility targeting is the one piece that earns its complexity: BTC's
    volatility moves by a factor of 3-4 across regimes, so a fixed stop
    distance means wildly different real risk per trade in calm vs violent
    markets.
    """

    tf: str = "4h"
    name: str = "trend_core"

    entry_window: int = 20
    exit_window: int = 0            # 0 = trail only; >0 = Donchian exit too
    atr_stop_mult: float = 2.5
    trail_mult: float = 3.0
    regime_ema: int = 200           # 0 disables the regime gate
    require_stack: bool = False
    breakeven_at_r: float = 1.0
    tp_r: float = 0.0               # 0 = no target, ride the trail
    tp_frac: float = 0.3
    vol_target: float = 0.0         # 0 = fixed risk; else annualised target
    vol_cap: float = 2.0            # ceiling on the vol-scaling multiplier

    # Macro gate on the DAILY frame. A 4h regime EMA spans only a couple of
    # weeks, which is short enough to keep re-arming inside a bear market;
    # a daily trend filter is the standard remedy in trend-following and is
    # included as a hypothesis to be tested walk-forward, not as a patch.
    macro_daily_ema: int = 0
    daily: pd.DataFrame | None = None

    def prepare(self, df):
        sig = F.build(df, self.tf)
        sig["entry_level"], _ = F.donchian(df, self.entry_window)
        if self.exit_window:
            _, sig["exit_level"] = F.donchian(df, self.exit_window)

        if self.macro_daily_ema:
            if self.daily is None:
                raise ValueError("macro_daily_ema needs the daily frame")
            d = pd.DataFrame(index=self.daily.index)
            d["macro_close"] = self.daily["close"]
            d["macro_ema"] = F.ema(self.daily["close"], self.macro_daily_ema)
            d = d.shift(1)          # yesterday's completed daily bar only
            sig = pd.merge_asof(sig.sort_index(), d.sort_index(),
                                left_index=True, right_index=True,
                                direction="backward")
        return sig

    def _size_scale(self, row) -> float:
        """Scale risk by how calm the market is, relative to the target."""
        if not self.vol_target:
            return 1.0
        rv = row["rv20"]
        if not np.isfinite(rv) or rv <= 0:
            return 1.0
        return float(np.clip(self.vol_target / rv, 1.0 / self.vol_cap, self.vol_cap))

    def entry(self, sig, i):
        row = sig.iloc[i]
        if np.isnan(row["entry_level"]) or np.isnan(row["atr14"]):
            return None
        if self.regime_ema:
            ref = row[f"ema{self.regime_ema}"] if f"ema{self.regime_ema}" in row else np.nan
            if np.isnan(ref) or row["close"] <= ref:
                return None
        if self.require_stack and row["trend_up"] != 1:
            return None
        if self.macro_daily_ema:
            macro, ref = row.get("macro_close", np.nan), row.get("macro_ema", np.nan)
            if np.isnan(macro) or np.isnan(ref) or macro <= ref:
                return None
        if row["close"] <= row["entry_level"]:
            return None

        stop = row["close"] - self.atr_stop_mult * row["atr14"]
        if stop <= 0:
            return None

        spec = {"direction": 1, "stop": stop, "targets": [],
                "risk_scale": self._size_scale(row)}
        if self.tp_r > 0:
            risk = row["close"] - stop
            spec["targets"] = [(row["close"] + self.tp_r * risk, self.tp_frac)]
        return spec

    def manage(self, pos, sig, i, bt):
        row = sig.iloc[i]
        atr = row["atr14"]
        if np.isnan(atr):
            return None
        risk = abs(pos.entry_price - pos.initial_stop)

        if self.breakeven_at_r and not pos.trail_active:
            if (row["close"] - pos.entry_price) >= self.breakeven_at_r * risk:
                pos.stop = max(pos.stop, pos.entry_price)
                pos.trail_active = True

        pos.peak = max(pos.peak, row["high"])
        pos.stop = max(pos.stop, pos.peak - self.trail_mult * atr)

        if self.exit_window and not np.isnan(row.get("exit_level", np.nan)):
            if row["close"] < row["exit_level"]:
                return "exit_channel"
        return None


@dataclass
class ChampionFaithful(Strategy):
    """Faithful port of the inherited repo's champion, for re-validation.

    Ported rule-for-rule from `scripts/btc_auto_strategies.py` in the
    btc-auto-trade repo, with two deliberate departures:

    * The campaign/sleeve pyramid (up to 3 adds off shrinking free cash) is
      collapsed to one position sized at the same first-sleeve notional
      (0.58 x 3.0 leverage). Adds were a small share of exposure and the
      original's own audit attributes the returns to the trailing exits.
    * The market-context and microstructure gates are omitted. In the source
      repo those datasets are now empty and both gates fail OPEN, so they are
      already inert there — reproducing them would just be reproducing True.

    Everything that actually drives the equity curve — the breakout signal,
    the daily macro filter, the fixed stop, the TP ladder, the post-TP2 trail,
    the exit signal and fail-fast — is reproduced exactly.
    """

    tf: str = "4h"
    name: str = "champion_faithful"
    daily: pd.DataFrame | None = None

    # Signal thresholds (the source's hardcoded defaults, not its dead config).
    volume_mult: float = 1.05
    di_ratio_mult: float = 0.97
    daily_close_ema200_mult: float = 0.985
    daily_rsi_min: float = 46.0

    # Sizing: first-sleeve share x mid-regime leverage.
    notional_frac: float = 0.58 * 3.0

    # Exits.
    stop_price_mult: float = 0.952
    stop_ema50_mult: float = 0.992
    tp1_mult: float = 1.10
    tp1_frac: float = 0.15
    tp2_mult: float = 1.24
    tp2_frac: float = 0.15
    trail_pct: float = 0.966
    stop_lock_1: float = 1.001
    stop_lock_2: float = 1.008
    min_hold_bars: int = 2

    # Fail-fast.
    fail_fast_mtm_pct: float = -0.0925   # -$925 on the source's $10k base
    fail_fast_rsi: float = 49.0
    fail_fast_ema_mult: float = 0.996
    fail_fast_max_bars: int = 6
    fail_fast_cooldown: int = 8

    def prepare(self, df):
        sig = F.build(df, self.tf)
        if self.daily is None:
            raise ValueError("ChampionFaithful needs the 1d frame for its macro filter")
        sig = F.merge_daily(sig, self.daily)

        sig["cmf20"] = F.cmf(df, 20)
        sig["di_plus"], sig["di_minus"] = F.dmi(df, 14)
        sig["avg_vol20"] = df["volume"].rolling(20, min_periods=20).mean()
        sig["box_high"] = df["high"].shift(1).rolling(10, min_periods=10).max()
        bull, bear = F.daily_regime(sig)
        sig["bull"], sig["bear"] = bull, bear

        macro_ok = (
            ~sig["bear"]
            & (sig["d_close"] >= sig["d_ema200"] * self.daily_close_ema200_mult)
            & (sig["d_rsi14"] >= self.daily_rsi_min)
        )
        reclaim_ok = (
            (sig["close"] >= sig["ema50"])
            & (sig["ema20"] >= sig["ema50"] * 0.995)
            & (sig["close"] >= sig["ema20"])
        )
        structure_ok = sig["close"] > sig["box_high"] * 1.0005
        volume_ok = sig["volume"] > sig["avg_vol20"] * self.volume_mult
        quality_ok = (sig["cmf20"] > -0.02) & (sig["di_plus"] >= sig["di_minus"] * self.di_ratio_mult)

        sig["breakout"] = (macro_ok & reclaim_ok & structure_ok & volume_ok & quality_ok).fillna(False)

        sig["exit_long"] = (
            ~sig["bull"]
            | (sig["close"] < sig["ema50"] * 0.985)
            | ((sig["close"] < sig["ema20"] * 0.99) & (sig["rsi14"] < 46))
        ).fillna(False)
        return sig

    def entry(self, sig, i):
        row = sig.iloc[i]
        if not row["breakout"] or np.isnan(row["ema50"]):
            return None
        price = row["close"]
        stop = min(price * self.stop_price_mult, row["ema50"] * self.stop_ema50_mult)
        if stop <= 0 or stop >= price:
            return None
        return {
            "direction": 1,
            "stop": stop,
            "notional_frac": self.notional_frac,
            "targets": [
                (price * self.tp1_mult, self.tp1_frac),
                (price * self.tp2_mult, self.tp2_frac),
            ],
        }

    def manage(self, pos, sig, i, bt):
        row = sig.iloc[i]

        # Stop-locks ratchet as each target fills; the trail only arms after TP2.
        filled = 2 - len(pos.targets)
        if filled >= 1:
            pos.stop = max(pos.stop, pos.entry_price * self.stop_lock_1)
        if filled >= 2:
            pos.stop = max(pos.stop, pos.entry_price * self.stop_lock_2)
            pos.peak = max(pos.peak, row["high"])
            pos.stop = max(pos.stop, pos.peak * self.trail_pct)

        if pos.bars < self.min_hold_bars:
            return None

        # Fail-fast: cut early losers inside the first few bars, then stand down.
        if pos.bars <= self.fail_fast_max_bars:
            mtm = (row["close"] - pos.entry_price) * pos.qty
            broke = row["close"] < row["ema50"] * self.fail_fast_ema_mult and \
                row["rsi14"] < self.fail_fast_rsi
            if mtm <= self.fail_fast_mtm_pct * bt.risk.initial_equity or broke:
                bt.cooldown_until = i + self.fail_fast_cooldown
                return "fail_fast"

        if row["exit_long"]:
            return "exit_signal"
        return None


@dataclass
class DualEngineRegime(Strategy):
    """Reimplementation of the inherited repo's champion family.

    Two long engines share one position slot:
      breakout     — early, cheap probe into a fresh N-bar high
      acceleration — continuation add-on once trend expansion is confirmed

    Regime routing gates both engines and sets how hard the trail follows.
    Rebuilt here from the strategy description so it can be run inside an
    engine that charges real costs and refuses lookahead.
    """

    tf: str = "4h"
    name: str = "dual_engine_regime"
    breakout_window: int = 20
    accel_window: int = 10
    atr_stop_mult: float = 2.0
    trail_mult: float = 3.5
    adx_min: float = 20.0
    accel_vol_z: float = 0.5
    require_above_200: bool = True
    tp1_r: float = 2.0
    tp1_frac: float = 0.25
    breakeven_at_r: float = 1.0

    def prepare(self, df):
        sig = F.build(df, self.tf)
        sig["bo_level"], _ = F.donchian(df, self.breakout_window)
        sig["ac_level"], _ = F.donchian(df, self.accel_window)
        sig["regime"] = F.regime(sig)
        return sig

    def entry(self, sig, i):
        row = sig.iloc[i]
        if np.isnan(row["atr14"]) or np.isnan(row["bo_level"]) or np.isnan(row["ema200"]):
            return None
        if self.require_above_200 and row["close"] <= row["ema200"]:
            return None

        breakout = row["close"] > row["bo_level"] and row["adx14"] >= self.adx_min
        acceleration = (
            row["close"] > row["ac_level"]
            and row["trend_up"] == 1
            and row["vol_z"] >= self.accel_vol_z
            and row["adx14"] >= self.adx_min
        )
        if not (breakout or acceleration):
            return None

        stop = row["close"] - self.atr_stop_mult * row["atr14"]
        if stop <= 0:
            return None
        risk = row["close"] - stop
        targets = []
        if self.tp1_frac > 0:
            targets.append((row["close"] + self.tp1_r * risk, self.tp1_frac))
        return {"direction": 1, "stop": stop, "targets": targets}

    def manage(self, pos, sig, i, bt):
        row = sig.iloc[i]
        atr = row["atr14"]
        if np.isnan(atr):
            return
        risk = abs(pos.entry_price - pos.initial_stop)

        if self.breakeven_at_r and not pos.trail_active:
            if (row["close"] - pos.entry_price) >= self.breakeven_at_r * risk:
                pos.stop = max(pos.stop, pos.entry_price)
                pos.trail_active = True

        # Trail tighter when the trend reading decays — the repo's audit showed
        # the entire edge lives in trailing exits on the few long trends.
        mult = self.trail_mult if row["adx14"] >= self.adx_min else self.trail_mult * 0.6
        pos.stop = max(pos.stop, row["close"] - mult * atr)
