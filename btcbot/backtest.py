"""Event-driven backtest engine.

Design rules, all of them chosen because violating them is how backtests end
up lying:

1. No lookahead. A signal is computed from the CLOSE of bar i and can only be
   filled at the OPEN of bar i+1. The engine never reads bar i+1 while
   deciding on bar i.
2. Intrabar honesty. When both the stop and a target sit inside one bar, the
   engine walks the finer (1h) bars inside that bar to find which came first.
   If both land in the same fine bar, the STOP wins — the pessimistic read.
3. Costs are always charged: taker fee on entry and every exit, slippage on
   every fill, and 8h funding on notional for as long as the position is open.
4. Leverage is real. A position that touches its liquidation price is closed
   there and loses the margin, before any stop logic runs.

The engine is strategy-agnostic: a Strategy supplies entry intent and exit
parameters, the engine owns all accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class Costs:
    """OKX/Binance-like perp cost structure, deliberately on the harsh side."""

    taker_fee: float = 0.0005        # 5 bps per side
    slippage_bps: float = 3.0        # 3 bps adverse on every fill
    funding: bool = True             # charge 8h funding on notional
    # Stops execute as market orders in fast markets: extra adverse slip.
    stop_slippage_bps: float = 8.0


@dataclass
class RiskConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.02     # fraction of equity risked to the stop
    max_leverage: float = 3.0        # cap on notional / equity
    maint_margin_rate: float = 0.005  # exchange maintenance margin
    max_equity_fraction: float = 1.0  # margin cap per position


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: int                   # +1 long, -1 short
    qty: float
    notional: float
    stop: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: str = ""
    pnl: float = 0.0
    fees: float = 0.0
    funding_paid: float = 0.0
    r_multiple: float = 0.0
    equity_after: float = 0.0
    mae: float = 0.0                 # max adverse excursion, in R
    mfe: float = 0.0                 # max favorable excursion, in R
    bars_held: int = 0


@dataclass
class Position:
    direction: int
    entry_price: float
    qty: float
    stop: float
    initial_stop: float
    entry_time: pd.Timestamp
    liq_price: float
    entry_fee: float
    targets: list = field(default_factory=list)   # [(price, fraction), ...]
    trail_active: bool = False
    funding_paid: float = 0.0
    peak: float = 0.0                             # best price seen
    trough: float = 0.0                           # worst price seen
    bars: int = 0
    realized: float = 0.0                         # pnl banked by partial exits
    exit_fees: float = 0.0
    original_qty: float = 0.0


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy,
        costs: Costs | None = None,
        risk: RiskConfig | None = None,
        fine: pd.DataFrame | None = None,
        funding: pd.Series | None = None,
        tf: str = "4h",
    ):
        self.df = df
        self.strategy = strategy
        self.costs = costs or Costs()
        self.risk = risk or RiskConfig()
        self.fine = fine
        self.funding = funding if funding is not None else pd.Series(dtype=float)
        self.tf = tf
        self.step = pd.Timedelta(tf)

        self.equity = self.risk.initial_equity
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple] = []
        self._pending_entry = None
        self._pending_exit: Optional[str] = None
        # Bars during which the strategy refuses new entries (fail-fast cooldown).
        self.cooldown_until = -1

        self._fine_index = None
        if fine is not None:
            self._fine_index = {t: i for i, t in enumerate(fine.index)}
            self._fine_arr = fine[["high", "low", "close"]].to_numpy()
            self._fine_times = fine.index

    # -- fills ------------------------------------------------------------

    def _fill_price(self, price: float, direction: int, extra_bps: float = 0.0) -> float:
        """Slippage always moves against us."""
        bps = (self.costs.slippage_bps + extra_bps) / 10_000
        return price * (1 + direction * bps)

    def _liq_price(self, entry: float, direction: int, leverage: float) -> float:
        """Price at which margin is exhausted, given isolated-margin leverage."""
        if leverage <= 0:
            return 0.0 if direction > 0 else math.inf
        move = 1 / leverage - self.risk.maint_margin_rate
        return entry * (1 - direction * move)

    # -- intrabar path ----------------------------------------------------

    def _intrabar_path(self, bar_time: pd.Timestamp, bar) -> list[tuple]:
        """Chronological (high, low) segments inside a bar.

        With fine data we get the real sequence; without it we fall back to a
        single segment, which forces the pessimistic same-bar tie-break.
        """
        if self._fine_index is None:
            return [(bar["high"], bar["low"])]
        end = bar_time + self.step
        lo = self._fine_times.searchsorted(bar_time, side="left")
        hi = self._fine_times.searchsorted(end, side="left")
        if hi <= lo:
            return [(bar["high"], bar["low"])]
        return [(self._fine_arr[j, 0], self._fine_arr[j, 1]) for j in range(lo, hi)]

    # -- position lifecycle ----------------------------------------------

    def _open(self, time: pd.Timestamp, price: float, spec) -> None:
        direction = spec["direction"]
        stop = spec["stop"]
        entry = self._fill_price(price, direction)

        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return

        # Two sizing modes. Risk-based (default) targets a fixed loss at the
        # stop. Notional-based reproduces margin x leverage systems, where the
        # loss at the stop varies with how far away the stop happens to sit.
        if "notional_frac" in spec:
            notional = self.equity * spec["notional_frac"]
            qty = notional / entry
        else:
            risk_cash = self.equity * self.risk.risk_per_trade * spec.get("risk_scale", 1.0)
            qty = risk_cash / stop_dist
            notional = qty * entry

        max_notional = self.equity * self.risk.max_leverage
        if notional > max_notional:
            qty = max_notional / entry
            notional = qty * entry
        if qty <= 0:
            return

        leverage = notional / self.equity
        fee = notional * self.costs.taker_fee
        self.equity -= fee

        self.position = Position(
            direction=direction,
            entry_price=entry,
            qty=qty,
            stop=stop,
            initial_stop=stop,
            entry_time=time,
            liq_price=self._liq_price(entry, direction, leverage),
            entry_fee=fee,
            targets=list(spec.get("targets", [])),
            peak=entry,
            trough=entry,
            original_qty=qty,
        )

    def _close_qty(self, price: float, qty: float, extra_bps: float) -> float:
        """Close part (or all) of the position; returns realized pnl net of fee."""
        p = self.position
        exit_px = self._fill_price(price, -p.direction, extra_bps)
        pnl = (exit_px - p.entry_price) * p.direction * qty
        fee = exit_px * qty * self.costs.taker_fee
        self.equity += pnl - fee
        p.realized += pnl
        p.exit_fees += fee
        p.qty -= qty
        return pnl - fee

    def _finalize(self, time: pd.Timestamp, price: float, reason: str, extra_bps: float) -> None:
        p = self.position
        if p.qty > 0:
            self._close_qty(price, p.qty, extra_bps)

        risk_cash = abs(p.entry_price - p.initial_stop) * p.original_qty
        total = p.realized - p.entry_fee - p.exit_fees - p.funding_paid
        adverse = (p.trough - p.entry_price) * p.direction * p.original_qty
        favorable = (p.peak - p.entry_price) * p.direction * p.original_qty

        self.trades.append(Trade(
            entry_time=p.entry_time,
            entry_price=p.entry_price,
            direction=p.direction,
            qty=p.original_qty,
            notional=p.original_qty * p.entry_price,
            stop=p.initial_stop,
            exit_time=time,
            exit_price=price,
            reason=reason,
            pnl=total,
            fees=p.entry_fee + p.exit_fees,
            funding_paid=p.funding_paid,
            r_multiple=total / risk_cash if risk_cash > 0 else 0.0,
            equity_after=self.equity,
            mae=adverse / risk_cash if risk_cash > 0 else 0.0,
            mfe=favorable / risk_cash if risk_cash > 0 else 0.0,
            bars_held=p.bars,
        ))
        self.position = None

    # -- funding ----------------------------------------------------------

    def _charge_funding(self, bar_time: pd.Timestamp, price: float) -> None:
        """Longs pay positive funding; shorts receive it."""
        if not self.costs.funding or self.funding.empty or self.position is None:
            return
        window = self.funding.loc[
            (self.funding.index >= bar_time) & (self.funding.index < bar_time + self.step)
        ]
        if window.empty:
            return
        p = self.position
        cost = float(window.sum()) * p.qty * price * p.direction
        self.equity -= cost
        p.funding_paid += cost

    # -- main loop --------------------------------------------------------

    def run(self) -> "Result":
        sig = self.strategy.prepare(self.df)
        times = self.df.index
        opens = self.df["open"].to_numpy()
        closes = self.df["close"].to_numpy()

        for i, t in enumerate(times):
            bar = self.df.iloc[i]

            # (1) Act on decisions made at the PREVIOUS bar's close, filling at
            #     this bar's open. Exits are settled before new entries so the
            #     single position slot is free again.
            if self._pending_exit is not None and self.position is not None:
                self._finalize(t, self._fill_price(opens[i], -self.position.direction),
                               self._pending_exit, 0.0)
            self._pending_exit = None

            if self._pending_entry is not None and self.position is None:
                self._open(t, opens[i], self._pending_entry)
            self._pending_entry = None

            # (2) Manage an open position across this bar.
            if self.position is not None:
                self.position.bars += 1
                self._charge_funding(t, opens[i])
                self._walk_bar(t, bar)

            # (3) Decide at this bar's close; act next bar.
            if self.position is not None:
                reason = self.strategy.manage(self.position, sig, i, self)
                if reason:
                    self._pending_exit = reason
            elif i >= self.cooldown_until:
                spec = self.strategy.entry(sig, i)
                if spec is not None:
                    self._pending_entry = spec

            self.equity_curve.append((t, self._mark_to_market(closes[i])))

        # Close anything still open at the final close, so the curve is honest.
        if self.position is not None:
            self._finalize(times[-1], closes[-1], "end_of_data", 0.0)

        return Result(self.trades, pd.Series(
            [e for _, e in self.equity_curve],
            index=[t for t, _ in self.equity_curve],
        ), self.risk.initial_equity, self.tf)

    def _mark_to_market(self, price: float) -> float:
        if self.position is None:
            return self.equity
        p = self.position
        return self.equity + (price - p.entry_price) * p.direction * p.qty

    def _walk_bar(self, t: pd.Timestamp, bar) -> None:
        """Resolve liquidation / stop / targets in chronological order."""
        p = self.position
        for high, low in self._intrabar_path(t, bar):
            if p is None:
                return
            p.peak = max(p.peak, high) if p.direction > 0 else min(p.peak, low)
            p.trough = min(p.trough, low) if p.direction > 0 else max(p.trough, high)

            # Liquidation dominates everything else.
            if p.direction > 0 and low <= p.liq_price:
                self._finalize(t, p.liq_price, "liquidation", self.costs.stop_slippage_bps)
                return
            if p.direction < 0 and high >= p.liq_price:
                self._finalize(t, p.liq_price, "liquidation", self.costs.stop_slippage_bps)
                return

            # Stop before target: if both sit in one fine bar we take the loss.
            hit_stop = (low <= p.stop) if p.direction > 0 else (high >= p.stop)
            if hit_stop:
                self._finalize(t, p.stop, "stop", self.costs.stop_slippage_bps)
                return

            while p.targets:
                price, frac = p.targets[0]
                hit = (high >= price) if p.direction > 0 else (low <= price)
                if not hit:
                    break
                p.targets.pop(0)
                qty = min(p.original_qty * frac, p.qty)
                if qty > 0:
                    self._close_qty(price, qty, 0.0)
                if p.qty <= 1e-12:
                    self._finalize(t, price, "target_final", 0.0)
                    return


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

_BPY = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


class Result:
    def __init__(self, trades: list[Trade], equity: pd.Series, initial: float, tf: str):
        self.trades = trades
        self.equity = equity
        self.initial = initial
        self.tf = tf

    @property
    def trade_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def stats(self) -> dict:
        eq = self.equity
        final = float(eq.iloc[-1]) if len(eq) else self.initial
        tf_df = self.trade_frame

        if tf_df.empty:
            return {"trades": 0, "final_equity": final, "total_return": 0.0}

        wins = tf_df[tf_df.pnl > 0]
        losses = tf_df[tf_df.pnl <= 0]
        gross_win = float(wins.pnl.sum())
        gross_loss = float(-losses.pnl.sum())

        rets = eq.pct_change().dropna()
        bpy = _BPY[self.tf]
        years = len(eq) / bpy if len(eq) else 1.0

        peak = eq.cummax()
        dd = (eq - peak) / peak
        downside = rets[rets < 0].std()

        # Growth on a curve that touched zero is meaningless; guard for it.
        cagr = (final / self.initial) ** (1 / years) - 1 if final > 0 and years > 0 else -1.0

        return {
            "trades": int(len(tf_df)),
            "final_equity": final,
            "total_return": final / self.initial - 1,
            "cagr": cagr,
            "max_drawdown": float(dd.min()),
            "sharpe": float(rets.mean() / rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0.0,
            "sortino": float(rets.mean() / downside * np.sqrt(bpy)) if downside and downside > 0 else 0.0,
            "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else 0.0,
            "win_rate": float(len(wins) / len(tf_df)),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else math.inf,
            "avg_r": float(tf_df.r_multiple.mean()),
            "expectancy_r": float(tf_df.r_multiple.mean()),
            "best_r": float(tf_df.r_multiple.max()),
            "worst_r": float(tf_df.r_multiple.min()),
            "total_fees": float(tf_df.fees.sum()),
            "total_funding": float(tf_df.funding_paid.sum()),
            "avg_bars_held": float(tf_df.bars_held.mean()),
            "liquidations": int((tf_df.reason == "liquidation").sum()),
            "exposure": float((tf_df.bars_held.sum()) / len(eq)) if len(eq) else 0.0,
        }

    def summary(self) -> str:
        s = self.stats()
        if not s.get("trades"):
            return "no trades"
        return (
            f"trades={s['trades']} ret={s['total_return']:+.1%} cagr={s['cagr']:+.1%} "
            f"maxDD={s['max_drawdown']:.1%} sharpe={s['sharpe']:.2f} "
            f"PF={s['profit_factor']:.2f} win={s['win_rate']:.1%} "
            f"avgR={s['avg_r']:+.2f} liq={s['liquidations']}"
        )
