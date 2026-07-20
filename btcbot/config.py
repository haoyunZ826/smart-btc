"""The single source of truth for what this project actually trades.

CHAMPION was selected by walk-forward: fit on rolling 3-year windows, scored
only on the untouched year that followed. Among the configurations that made
money in EVERY out-of-sample window, this one compounded the most. Its
out-of-sample CAGR came in within 5 points of its in-sample CAGR, which is the
evidence that it is fitted to BTC's trend behaviour rather than to the sample.

Do not tune these numbers against the full history. If you want to change
them, change the grid in scripts/research.py and re-run the walk-forward, so
whatever comes out is judged on data it has never seen.
"""

from __future__ import annotations

CHAMPION = {
    "name": "trend_core_v1",
    "tf": "4h",

    # Entry: break the 30-bar high, above the 100-EMA regime filter.
    "entry_window": 30,
    "regime_ema": 100,
    "require_stack": False,

    # Macro gate: yesterday's daily close must be above its 100-day EMA.
    #
    # Disclosure on how this was chosen. The walk-forward selection above ran
    # WITHOUT this filter; it was added afterwards, once the validated config
    # turned out to be bleeding through the 2025-26 bear alongside buy & hold.
    # That ordering is a real bias risk, so the filter was held to a higher
    # bar than "it fixes the recent window": it had to keep the full-sample
    # return intact (153x vs 152x), improve the full-sample drawdown
    # (-37% vs -51%) and Calmar (2.03 vs 1.47), and still win all five
    # out-of-sample windows. It does all four. What it costs is out-of-sample
    # compounded return across windows (17x vs 27x), almost entirely from one
    # window where a rally began below the daily EMA and the filter sat it out.
    # See scripts/test_macro_filter.py to reproduce.
    "macro_daily_ema": 100,

    # Exit: tight initial stop, wide chandelier trail. The asymmetry is the
    # point — cut fast, then give a live trend room to actually run.
    "atr_stop_mult": 2.0,
    "trail_mult": 5.0,
    "breakeven_at_r": 1.0,
    "tp_r": 0.0,            # no fixed target: capping the winners kills the edge
    "exit_window": 0,

    # Size: scale risk toward a 60% annualised volatility target, so a trade in
    # a calm market and a trade in a violent one carry comparable real risk.
    "vol_target": 0.6,
    "vol_cap": 2.0,

    # Position size. See RISK_PROFILES for the alternatives.
    #
    # Set to the `aggressive` profile by choice: it maximises return, and its
    # Calmar (2.47) is the best of the three because up to this point the extra
    # size scales return faster than drawdown.
    #
    # Read RISK_WARNING below before leaving it here.
    "risk_per_trade": 0.05,
    "max_leverage": 8.0,
}

DEFAULT_PROFILE = "aggressive"

# The realised history shows -49% max drawdown, but the Monte-Carlo reshuffle
# says that path sat on the lucky side of the distribution: median max
# drawdown -47%, and a 38% chance of exceeding -50%. At `balanced` (3%/5x)
# that same figure is 4%. Stepping from balanced to aggressive multiplies the
# return by roughly 7.5x and the odds of a halving by roughly 10x. That is a
# risk preference, not a backtest finding.
RISK_WARNING = (
    "aggressive (5%/8x): 38% Monte-Carlo chance of a >50% drawdown, "
    "vs 4% at balanced (3%/5x)"
)

# Same signal, three appetites. The signal is what was validated; the sizing is
# a preference, and the drawdown column is the part to read before choosing.
RISK_PROFILES = {
    "conservative": {"risk_per_trade": 0.01, "max_leverage": 3.0},
    "balanced":     {"risk_per_trade": 0.03, "max_leverage": 5.0},
    "aggressive":   {"risk_per_trade": 0.05, "max_leverage": 8.0},
}

# Costs assumed everywhere. Chosen to be slightly worse than a real taker fill
# on OKX/Binance, so live results are more likely to beat the backtest than
# miss it.
COSTS = {
    "taker_fee": 0.0005,
    "slippage_bps": 3.0,
    "stop_slippage_bps": 8.0,
    "funding": True,
}
