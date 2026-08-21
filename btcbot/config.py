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
# Shown in the dashboard's disclosure box. It has to name the worst thing on the
# page, not the default — a warning that quietly stops covering the riskiest
# tier is how a dashboard ends up implying the deep-drawdown option was vetted
# to the same standard as the others.
RISK_WARNING = (
    "aggressive (5%/8x): 38% Monte-Carlo chance of a >50% drawdown, "
    "vs 4% at balanced (3%/5x). "
    "filtered_hot (15%/15x) is the deep end: -79% max drawdown over the full "
    "sample, i.e. what is left has to make 4.8x to break even — and the same "
    "signal wipes out entirely at 30% risk. Paper only, sized from 33 trades."
)

# Same signal, three appetites. The signal is what was validated; the sizing is
# a preference, and the drawdown column is the part to read before choosing.
# The first three profiles are the same signal at different position sizes.
# `filtered` is the exception and the only one that changes the entry rule, so
# it is labelled separately everywhere it is displayed.
RISK_PROFILES = {
    "conservative": {"risk_per_trade": 0.01, "max_leverage": 3.0},
    "balanced":     {"risk_per_trade": 0.03, "max_leverage": 5.0},
    "aggressive":   {"risk_per_trade": 0.05, "max_leverage": 8.0},

    # ON PROBATION — paper only. Not the default, and not a recommendation yet.
    #
    # What it is: the same trend signal plus two entry-quality gates, sized up
    # to 6% because the gates cut trade count roughly 5x. Both gates came from
    # comparing the four 2026 paper trades, then survived a real test:
    # scripts/experiment_filters_{entry,risk_matched,robustness}.py.
    #
    # What the test actually said (research/entry_filter_*.json):
    #   * At equal risk-per-trade it LOSES to aggressive (2,174% vs 5,053% OOS).
    #     The filters do not find a better signal. Anyone quoting them as an
    #     edge has read the wrong column.
    #   * The original claim here — "at equal drawdown it roughly holds serve,
    #     4,584% vs 5,053%" — was WRONG, and wrong in an instructive way: that
    #     figure came from rolling-window compounding, which restarts equity
    #     every year. On one continuous account at the same realised drawdown
    #     (-48.5% vs -48.7%) it is 3,655% vs 121,846%. A 33x gap, not a tie.
    #   * What survives: the Monte-Carlo probability of a >50% drawdown is 0.0%
    #     here vs 34.2% for aggressive. That is the only remaining argument for
    #     this profile, and it is bought with 33x less return.
    #   * It is worse than aggressive in 2 of 6 out-of-sample windows, both of
    #     them bull-market legs, where a filter is pure cost.
    #
    # Why it stays on probation: only 18 out-of-sample trades in nine years.
    # The 11.6 profit factor those trades imply is far too few samples to trust,
    # and the honest expectation is that the real number is materially lower.
    # Let it run on paper next to the other three before touching DEFAULT_PROFILE.
    "filtered": {
        "risk_per_trade": 0.06,
        "max_leverage": 8.0,
        "strategy": {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25},
    },

    # ON PROBATION, AND THE MOST DANGEROUS SETTING IN THIS FILE. Paper only.
    #
    # Same two gates as `filtered`, run hot: 15% risk per trade at 15x. Added on
    # request after an explicit "I can tolerate deep drawdowns for more return".
    # It is a legitimate point on the frontier, not a mistake — but it is a
    # different animal from the tiers above it, so read all four of these:
    #
    # 1. -79.4% max drawdown over the full sample. The survivors of that have to
    #    make 4.8x just to get back to even. That is the actual meaning of the
    #    number, and it is worth re-reading before this is ever run with money.
    # 2. There is a liquidation cliff not far above. At 8x the same signal takes
    #    28% risk with zero liquidations and 30% with a total wipeout
    #    (2020-05-10, equity -$3.77). The exact location of that cliff is an
    #    artifact of one candle in one price path. Do not creep toward it.
    # 3. The size was picked from a return peak estimated on 33 trades. Optimal
    #    leverage estimated from a sample that small is biased high — this is a
    #    known statistical property, not a hunch. Half of it (~7%) is the
    #    defensible reading of the same evidence.
    # 4. Do NOT rank this against the others by out-of-sample compounded return.
    #    That metric restarts equity every window, so it cannot represent ruin:
    #    the 30%-risk config that wipes out scores HIGHEST on it. Judge position
    #    size on full-sample return plus liquidation count, never on that column.
    #
    # Why the filters earn their keep here rather than being dropped: at 20%/15x
    # the unfiltered signal is liquidated twice (-100%) while this one survives
    # at -87% and returns 8.6x more. The gates are not the brake, they are what
    # buys the headroom to run this hot at all.
    "filtered_hot": {
        "risk_per_trade": 0.15,
        "max_leverage": 15.0,
        "strategy": {"min_thrust_atr": 0.50, "min_dvol_ratio": 1.25},
    },
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
