"""
backtest/strategies.py — Strategy registry for the backtesting framework.

ALL_STRATEGIES:    Ordered list of all strategy names. This is the
                   single source of truth for valid --strategy CLI values
                   and walk-forward reporting labels.

STRATEGY_REGISTRY: Dict mapping strategy name → human-readable label.
                   Used by analytics, reporting, and the CLI help text.

Strategy groups
───────────────
Trend family  (S1 + sub-strategies):
  S1_LONDON_BRK   — London Asian-range breakout (base)
  S1B_FAILED_BRK  — Failed breakout reversal
  S1D_PYRAMID     — M5 EMA20 pullback re-entry (pyramid)
  S1E_PYRAMID     — Confirmed-winner pyramid add (post partial + BE)
  S1F_POST_TK     — Post London time-kill NY re-entry

Mean reversion:
  S2_MEAN_REV     — RSI + EMA20 deviation mean reversion (RANGING_CLEAR only)

Pattern:
  S3_STOP_HUNT_REV — Stop-hunt sweep + reclaim reversal

Pullback / breakout:
  S4_EMA_PULLBACK  — London EMA20(H1) pullback in trending regime
  S5_NY_COMPRESS   — NY session compression breakout

OCO pairs:
  S6_ASIAN_BRK    — Asian range OCO breakout pair (placed at Asian open)
  S7_DAILY_STRUCT — Previous-day high/low structure OCO breakout pair

Independent lanes (fire concurrently with trend family):
  S8_NEWS_SPIKE   — ATR spike continuation (2×ATR bar + confirmation)
  R3_CAL_MOMENTUM — Post high-impact event calendar momentum (30-min window)

Parity with live algo: all 13 strategies above match the live
signal_engine.py evaluate_* functions as of fix-V3.0.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY NAMES
# Order matters: it determines display order in reports and CLI help text.
# ─────────────────────────────────────────────────────────────────────────────

ALL_STRATEGIES: list[str] = [
    # Trend family
    "S1_LONDON_BRK",
    "S1B_FAILED_BRK",
    "S1D_PYRAMID",
    "S1E_PYRAMID",
    "S1F_POST_TK",
    # Mean reversion
    "S2_MEAN_REV",
    # Pattern
    "S3_STOP_HUNT_REV",
    # Pullback / NY breakout
    "S4_EMA_PULLBACK",
    "S5_NY_COMPRESS",
    # OCO pairs
    "S6_ASIAN_BRK",
    "S7_DAILY_STRUCT",
    # Independent lanes
    "S8_NEWS_SPIKE",
    "R3_CAL_MOMENTUM",
]

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRY  {name: label}
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, str] = {
    "S1_LONDON_BRK":   "S1  — London Range Breakout",
    "S1B_FAILED_BRK":  "S1b — Failed Breakout Reversal",
    "S1D_PYRAMID":     "S1d — M5 EMA20 Pullback Re-entry",
    "S1E_PYRAMID":     "S1e — Confirmed Winner Pyramid Add",
    "S1F_POST_TK":     "S1f — Post Time-Kill NY Re-entry",
    "S2_MEAN_REV":     "S2  — Mean Reversion (Ranging)",
    "S3_STOP_HUNT_REV":"S3  — Stop Hunt Reversal",
    "S4_EMA_PULLBACK": "S4  — London EMA20 Pullback",
    "S5_NY_COMPRESS":  "S5  — NY Session Compression Breakout",
    "S6_ASIAN_BRK":    "S6  — Asian Range Breakout OCO",
    "S7_DAILY_STRUCT": "S7  — Daily Structure Breakout OCO",
    "S8_NEWS_SPIKE":   "S8  — ATR Spike Continuation",
    "R3_CAL_MOMENTUM": "R3  — Calendar Momentum (Post-Event)",
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY GROUPS  — for grouped reporting / selective runs
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_GROUPS: dict[str, list[str]] = {
    "trend_family":   ["S1_LONDON_BRK", "S1B_FAILED_BRK", "S1D_PYRAMID",
                       "S1E_PYRAMID", "S1F_POST_TK"],
    "mean_reversion": ["S2_MEAN_REV"],
    "pattern":        ["S3_STOP_HUNT_REV"],
    "pullback":       ["S4_EMA_PULLBACK", "S5_NY_COMPRESS"],
    "oco_pairs":      ["S6_ASIAN_BRK", "S7_DAILY_STRUCT"],
    "independent":    ["S8_NEWS_SPIKE", "R3_CAL_MOMENTUM"],
}
