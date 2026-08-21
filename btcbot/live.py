"""Paper-trading monitor.

Design decision that matters more than any other file here: the live path runs
the SAME Backtester over the same Strategy class as the research path. It does
not reimplement the signal.

The inherited project got this wrong in a way worth remembering — its live
runner resolved signals through a code path that could never emit the
champion's signal names, so the deployed bot silently traded a different
strategy than the one that was backtested, and every cycle logged "no signal".

Here, state is *derived* rather than accumulated: each run replays the whole
history from PAPER_START and reports the final state. Replaying ~19k bars
takes about two seconds, and in exchange the live position can never drift
from what the strategy actually specifies. There is no incremental state file
to corrupt, and yesterday's bug cannot poison tomorrow's position.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from . import strategies as S
from .backtest import Backtester, Costs, RiskConfig
from .config import CHAMPION, DEFAULT_PROFILE, RISK_PROFILES

ROOT = Path(__file__).resolve().parent.parent

# Paper trading starts here. Chosen as a round date well after the end of the
# data used for selection, so the live record is a genuine forward test.
PAPER_START = "2026-01-01"
PAPER_EQUITY = 10_000.0


def build_strategy(daily: pd.DataFrame, overrides: dict | None = None) -> S.TrendCore:
    params = {k: v for k, v in CHAMPION.items()
              if k not in ("tf", "risk_per_trade", "max_leverage", "name")}
    params["daily"] = daily
    if overrides:
        params.update(overrides)
    return S.TrendCore(tf=CHAMPION["tf"], **params)


def run_paper(profile: str = DEFAULT_PROFILE, start: str = PAPER_START) -> dict:
    """Replay the strategy from `start` to now and return the current state."""
    tf = CHAMPION["tf"]
    df = D.load_ohlcv(tf)
    daily = D.load_ohlcv("1d")
    funding = D.load_funding()

    cfg = RISK_PROFILES[profile]
    cut = pd.Timestamp(start, tz="UTC")
    # Keep warmup history in front of the start so indicators are live on day 1.
    lo = max(0, df.index.searchsorted(cut) - 400)
    window = df.iloc[lo:]

    # Most profiles are pure position sizing. A profile may also override the
    # entry rule (see `filtered` in config), so the strategy is built per
    # profile rather than once — otherwise every profile would silently trade
    # the default signal while claiming to trade its own.
    strat = build_strategy(daily, cfg.get("strategy"))
    bt = Backtester(
        window, strat,
        costs=Costs(**{k: v for k, v in {
            "taker_fee": 0.0005, "slippage_bps": 3.0,
            "stop_slippage_bps": 8.0, "funding": True}.items()}),
        risk=RiskConfig(initial_equity=PAPER_EQUITY,
                        risk_per_trade=cfg["risk_per_trade"],
                        max_leverage=cfg["max_leverage"]),
        funding=funding, tf=tf,
    )
    result = bt.run()

    sig = strat.prepare(window)
    last = sig.iloc[-1]
    price = float(df["close"].iloc[-1])

    # Report the equity curve only from the paper start, not the warmup.
    curve = result.equity[result.equity.index >= cut]
    if curve.empty:
        curve = result.equity

    state = {
        "profile": profile,
        "entry_filters": cfg.get("strategy") or {},
        "as_of": df.index[-1].isoformat(),
        "price": price,
        "equity": float(curve.iloc[-1]),
        "initial_equity": PAPER_EQUITY,
        "return_pct": float(curve.iloc[-1] / PAPER_EQUITY - 1),
        "position": _position_view(bt, price),
        "signal": _signal_view(last, price, strat),
        "stats": result.stats(),
        "curve": {
            "dates": [d.strftime("%Y-%m-%d %H:%M") for d in curve.index],
            "equity": [round(float(v), 2) for v in curve.values],
        },
        "trades": _trade_view(result),
    }
    return state


def _position_view(bt: Backtester, price: float) -> dict | None:
    # `bt.position` is always None after run(): the engine closes whatever is
    # open at the last bar so the equity curve ends honestly. For research that
    # is correct and invisible. For the paper monitor it was a lie — the last
    # bar is *now*, and a position that is still open would be reported FLAT,
    # which is the one thing a position monitor exists to get right.
    p = bt.position or bt.open_at_end
    if p is None:
        return None
    unrealized = (price - p.entry_price) * p.direction * p.qty
    risk = abs(p.entry_price - p.initial_stop)
    return {
        "direction": "long" if p.direction > 0 else "short",
        "entry_time": p.entry_time.isoformat(),
        "entry_price": round(p.entry_price, 2),
        "qty": round(p.qty, 6),
        "notional": round(p.qty * price, 2),
        "stop": round(p.stop, 2),
        "initial_stop": round(p.initial_stop, 2),
        "liq_price": round(p.liq_price, 2),
        "unrealized": round(unrealized, 2),
        "unrealized_r": round(unrealized / (risk * p.original_qty), 2) if risk else 0.0,
        "bars_held": p.bars,
        "funding_paid": round(p.funding_paid, 2),
    }


def _signal_view(last: pd.Series, price: float, strat: S.TrendCore) -> dict:
    """What the strategy is waiting for, in plain terms."""
    level = float(last.get("entry_level", np.nan))
    regime_ref = float(last.get(f"ema{strat.regime_ema}", np.nan)) if strat.regime_ema else np.nan
    macro_close = float(last.get("macro_close", np.nan))
    macro_ema = float(last.get("macro_ema", np.nan))

    gates = {
        "breakout": {
            "label": f"close > {strat.entry_window}-bar high",
            "pass": bool(np.isfinite(level) and price > level),
            "detail": f"{price:,.0f} vs {level:,.0f}" if np.isfinite(level) else "n/a",
        },
        "regime": {
            "label": f"close > EMA{strat.regime_ema} ({strat.tf})",
            "pass": bool(np.isfinite(regime_ref) and price > regime_ref),
            "detail": f"{price:,.0f} vs {regime_ref:,.0f}" if np.isfinite(regime_ref) else "n/a",
        },
        "macro": {
            "label": f"daily close > EMA{strat.macro_daily_ema} (1d)",
            "pass": bool(np.isfinite(macro_close) and np.isfinite(macro_ema)
                         and macro_close > macro_ema),
            "detail": (f"{macro_close:,.0f} vs {macro_ema:,.0f}"
                       if np.isfinite(macro_ema) else "n/a"),
        },
    }

    # Profile-specific entry filters are real gates. Leaving them out would let
    # the `filtered` profile display "signal ready" on a bar its own rules
    # reject — the dashboard equivalent of trading a different strategy than the
    # one on the label.
    if strat.min_thrust_atr:
        thrust = float(last.get("thrust_atr", np.nan))
        gates["thrust"] = {
            "label": f"breakout clears channel by {strat.min_thrust_atr:g} x ATR",
            "pass": bool(np.isfinite(thrust) and thrust >= strat.min_thrust_atr),
            "detail": (f"{thrust:.2f} vs {strat.min_thrust_atr:g} ATR"
                       if np.isfinite(thrust) else "n/a"),
        }
    if strat.min_dvol_ratio:
        dvr = float(last.get("dvol_ratio", np.nan))
        gates["volume"] = {
            "label": f"prior-day volume >= {strat.min_dvol_ratio:g}x its 20d mean",
            "pass": bool(np.isfinite(dvr) and dvr >= strat.min_dvol_ratio),
            "detail": (f"{dvr:.2f}x vs {strat.min_dvol_ratio:g}x"
                       if np.isfinite(dvr) else "n/a"),
        }
    if strat.macro_daily_sma:
        mc = float(last.get("macro_close", np.nan))
        ms = float(last.get("macro_sma", np.nan))
        gates["sma"] = {
            "label": f"daily close > SMA{strat.macro_daily_sma} (1d)",
            "pass": bool(np.isfinite(mc) and np.isfinite(ms) and mc > ms),
            "detail": f"{mc:,.0f} vs {ms:,.0f}" if np.isfinite(ms) else "n/a",
        }

    atr = float(last.get("atr14", np.nan))
    return {
        "gates": gates,
        "ready": all(g["pass"] for g in gates.values()),
        "atr": round(atr, 2) if np.isfinite(atr) else None,
        "atr_pct": round(float(last.get("atr_pct", np.nan)) * 100, 2)
        if np.isfinite(last.get("atr_pct", np.nan)) else None,
        "rv20": round(float(last.get("rv20", np.nan)), 3)
        if np.isfinite(last.get("rv20", np.nan)) else None,
        "would_stop_at": round(price - strat.atr_stop_mult * atr, 2)
        if np.isfinite(atr) else None,
    }


def _trade_view(result, limit: int = 60) -> list[dict]:
    frame = result.trade_frame
    if frame.empty:
        return []
    frame = frame.tail(limit)
    out = []
    for _, t in frame.iterrows():
        # "end_of_data" is not an exit — it is the engine marking the still-open
        # position to market at the last bar. Listing it next to real stops would
        # read as a closed winner that nobody can actually book.
        still_open = t.reason == "end_of_data"
        out.append({
            "entry_time": str(t.entry_time),
            "exit_time": None if still_open else str(t.exit_time),
            "entry_price": round(float(t.entry_price), 2),
            "exit_price": None if still_open else round(float(t.exit_price), 2),
            "direction": "long" if t.direction > 0 else "short",
            "pnl": round(float(t.pnl), 2),
            "r": round(float(t.r_multiple), 2),
            "reason": "open" if still_open else t.reason,
            "open": still_open,
            "bars_held": int(t.bars_held),
        })
    return list(reversed(out))
