"""
state.py — System runtime state.

Phase 2 update:
  - Added R3, S4, S5 state keys
  - Added ks7_pre_event_price for R3 direction determination
  - Bug 2 fix: all S3 sweep resets consolidated in reset_daily_counters()
  - F1-F7 state keys merged (S8, spread_multiplier, dxy_ewma_variance, etc.)
  - V3.0 fix: S8 independent position lane keys added
  - V3.0 fix: DI+/DI- H4 cache keys added (for S6/S7 ADX filter)
"""
from datetime import datetime
import pytz
import config
from utils.logger import log_event


def build_initial_state() -> dict:
    return {
        # ── Regime ────────────────────────────────────────────────────────────
        "current_regime":               "NO_TRADE",
        "pending_regime_state":         None,
        "consecutive_regime_readings":  0,
        "regime_calculated_at":         datetime.now(pytz.utc),
        "size_multiplier":              0.0,
        "_last_persisted_regime":       None,

        # ── Macro ─────────────────────────────────────────────────────────────
        "macro_bias":                   "BOTH_PERMITTED",
        "dxy_corr_50":                  -0.50,
        "macro_boost":                  False,
        "macro_proxy_instrument":       "TLT",
        "dxy_ewma_variance":            0.0,      # F4: EWMA variance of DXY returns

        # ── Conviction ────────────────────────────────────────────────────────
        "conviction_level":             "STANDARD",

        # ── P&L ───────────────────────────────────────────────────────────────
        "daily_net_pnl_pct":            0.0,
        "daily_commission_paid":        0.0,
        "weekly_net_pnl_pct":           0.0,
        "peak_equity":                  0.0,
        "current_drawdown_pct":         0.0,

        # ── Main trend-family position ─────────────────────────────────────────
        "open_position":                None,
        "entry_price":                  0.0,
        "stop_price_original":          0.0,
        "stop_price_current":           0.0,
        "original_lot_size":            0.0,
        "open_trade_id":                None,
        "open_campaign_id":             None,
        "trend_family_occupied":        False,
        "trend_family_strategy":        None,
        "position_partial_done":        False,
        "position_be_activated":        False,
        "position_pyramid_done":        False,
        "position_m5_count":            0,

        # ── Loss tracking ─────────────────────────────────────────────────────
        "consecutive_losses":           0,
        "consecutive_m5_losses":        0,

        # ── Daily counters ────────────────────────────────────────────────────
        "s1_family_attempts_today":     0,
        "s1f_attempts_today":           0,

        # ── Failed breakout (S1b) ─────────────────────────────────────────────
        "failed_breakout_flag":         False,
        "failed_breakout_flag_candles": 0,
        "failed_breakout_direction":    None,

        # ── S1 tracking ───────────────────────────────────────────────────────
        "last_s1_direction":            None,
        "last_s1_max_r":                0.0,

        # ── Stop hunt (S1c) ───────────────────────────────────────────────────
        "stop_hunt_detected":           False,
        "stop_hunt_direction":          None,
        "stop_hunt_candles":            0,

        # ── KS7 event blackout ────────────────────────────────────────────────
        "ks7_active":                   False,
        "ks7_pre_event_atr":            0.0,
        "ks7_pre_event_price":          0.0,   # Phase 2: for R3 direction determination
        "ks7_severity_score":           0.0,   # F5: current severity score

        # ── System health ─────────────────────────────────────────────────────
        "trading_enabled":              True,
        "shutdown_reason":              None,
        "network_fail_count":           0,

        # ── Spread tracking ───────────────────────────────────────────────────
        "session_spread_initialized":   False,
        "spread_fallback_active":       True,
        "spread_readings_count":        0,
        "current_equity":               0.0,
        "spread_elevated_reading_count": 0,   # F6: consecutive wide-spread readings
        "spread_elevated_last_ratio":    0.0, # F6: last measured spread ratio
        "spread_multiplier":             1.0, # F6: current spread lot multiplier

        # ── Range (S1 pre-London) ─────────────────────────────────────────────
        "range_data":                   None,

        # ── Session time-kills ────────────────────────────────────────────────
        "london_tk_fired_today":        False,
        "ny_tk_fired_today":            False,

        # ── Cached indicator values ───────────────────────────────────────────
        "last_adx_h4":                  0.0,
        "last_atr_pct_h1":              0.0,
        "tlt_slope":                    0.0,
        "last_atr_h1_raw":              0.0,   # F1-F7 Bug 1 fix: cached H1 ATR value

        # ── Cached DI+/DI- values (V3.0: for S6/S7 ADX trend filter) ──────────
        "last_di_plus_h4":              None,
        "last_di_minus_h4":             None,

        # ── S1 pending STOP orders ────────────────────────────────────────────
        "s1_pending_buy_ticket":        None,
        "s1_pending_sell_ticket":       None,

        # ── Reversal family (S1b + S3) ────────────────────────────────────────
        "reversal_family_occupied":     False,
        "s1b_pending_ticket":           None,

        # ── S1d first-touch guards ────────────────────────────────────────────
        "s1d_ema_touched_today":        False,
        "s1d_fired_today":              False,

        # ── KS4 countdown ─────────────────────────────────────────────────────
        "ks4_reduced_trades_remaining": 0,

        # ── S3 Stop Hunt Reversal ─────────────────────────────────────────────
        "s3_sweep_candle_time":         None,
        "s3_sweep_low":                 0.0,
        "s3_sweep_high":                0.0,
        "s3_fired_today":               False,

        # ── S6 Asian Range Breakout ───────────────────────────────────────────
        "s6_pending_buy_ticket":        None,
        "s6_pending_sell_ticket":       None,
        "s6_range_high":                0.0,
        "s6_range_low":                 0.0,
        "s6_fired_today":               False,

        # ── S7 Daily Structure ────────────────────────────────────────────────
        "s7_pending_buy_ticket":        None,
        "s7_pending_sell_ticket":       None,
        "s7_prev_day_high":             0.0,
        "s7_prev_day_low":              0.0,
        "s7_fired_today":               False,
        "s7_daily_atr":                 0.0,
        "high_corr_pairs":              [],

        # ── S8 ATR Spike Trade (F7) — signal/arm state ─────────────────────────
        "s8_armed":                     False,
        "s8_arm_time":                  None,
        "s8_arm_candle_time":           None,
        "s8_spike_high":                0.0,
        "s8_spike_low":                 0.0,
        "s8_direction":                 None,
        "s8_spike_atr":                 0.0,
        "s8_spike_candle_idx":          None,
        "s8_fired_today":               False,

        # ── S8 ATR Spike Trade — independent position lane (V3.0) ──────────────
        # S8 does NOT occupy trend_family. It has its own open-position keys
        # so it can coexist with S1/S4/S5 trend trades.
        "s8_open_ticket":               None,
        "s8_entry_price":               0.0,
        "s8_stop_price_original":       0.0,
        "s8_stop_price_current":        0.0,
        "s8_trade_direction":           None,
        "s8_be_activated":              False,
        "s8_open_time_utc":             None,

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — NEW STATE KEYS
        # ══════════════════════════════════════════════════════════════════════

        # ── R3 Calendar Momentum (independent family) ─────────────────────────
        # R3 tracks its own open position separately from open_position
        # because it can coexist with an active S1/S2/S4/S5 trend trade.
        "r3_armed":                     False,
        "r3_arm_time":                  None,
        "r3_direction":                 None,
        "r3_fired_today":               False,
        "r3_open_ticket":               None,   # independent of open_position
        "r3_open_time":                 None,
        "r3_entry_price":               0.0,
        "r3_stop_price":                0.0,
        "r3_tp_price":                  0.0,
        "r3_event_scheduled_utc":       None,   # the event that triggered R3

        # ── S4 London Pullback ────────────────────────────────────────────────
        "s4_ema_touched":               False,  # first London EMA20 touch this session
        "s4_fired_today":               False,
        "s4_touch_bar_low":             0.0,    # M15 low of touch candle (LONG stop ref)
        "s4_touch_bar_high":            0.0,    # M15 high of touch candle (SHORT stop ref)

        # ── S5 NY Compression Breakout ────────────────────────────────────────
        "london_session_high":          0.0,    # running London session high (07:00-12:00 UTC)
        "london_session_low":           0.0,    # running London session low
        "london_session_tracking_active": False,
        "s5_compression_confirmed":     False,
        "s5_fired_today":               False,
        "d1_atr_14":                    0.0,    # D1 ATR(14,RMA) — computed at midnight
    }


# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED STATE KEYS — schema contract
# Every key must be typed here. validate_state_keys() enforces this.
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_STATE_KEYS: dict[str, type | tuple] = {
    # Regime
    "current_regime":               str,
    "pending_regime_state":         (str, type(None)),
    "consecutive_regime_readings":  int,
    "regime_calculated_at":         datetime,
    "size_multiplier":              float,
    "_last_persisted_regime":       (str, type(None)),

    # Macro
    "macro_bias":                   str,
    "dxy_corr_50":                  float,
    "macro_boost":                  bool,
    "macro_proxy_instrument":       str,
    "dxy_ewma_variance":            float,

    # Conviction
    "conviction_level":             str,

    # P&L
    "daily_net_pnl_pct":            float,
    "daily_commission_paid":        float,
    "weekly_net_pnl_pct":           float,
    "peak_equity":                  float,
    "current_drawdown_pct":         float,

    # Main position
    "open_position":                (int, type(None)),
    "trend_family_occupied":        bool,
    "trend_family_strategy":        (str, type(None)),
    "position_partial_done":        bool,
    "position_be_activated":        bool,
    "position_pyramid_done":        bool,
    "position_m5_count":            int,

    # Loss tracking
    "consecutive_losses":           int,
    "consecutive_m5_losses":        int,

    # Daily counters
    "s1_family_attempts_today":     int,
    "s1f_attempts_today":           int,

    # Failed breakout
    "failed_breakout_flag":         bool,
    "failed_breakout_flag_candles": int,
    "failed_breakout_direction":    (str, type(None)),

    # S1 tracking
    "last_s1_direction":            (str, type(None)),
    "last_s1_max_r":                float,

    # Stop hunt
    "stop_hunt_detected":           bool,
    "stop_hunt_direction":          (str, type(None)),
    "stop_hunt_candles":            int,

    # KS7
    "ks7_active":                   bool,
    "ks7_pre_event_atr":            float,
    "ks7_pre_event_price":          float,
    "ks7_severity_score":           float,

    # System health
    "trading_enabled":              bool,
    "shutdown_reason":              (str, type(None)),
    "network_fail_count":           int,

    # Spread
    "session_spread_initialized":   bool,
    "spread_fallback_active":       bool,
    "spread_readings_count":        int,
    "current_equity":               float,
    "spread_elevated_reading_count": int,
    "spread_elevated_last_ratio":    float,
    "spread_multiplier":             float,

    # Range
    "range_data":                   (dict, type(None)),

    # Time-kills
    "london_tk_fired_today":        bool,
    "ny_tk_fired_today":            bool,

    # Cached indicators
    "last_adx_h4":                  float,
    "last_atr_pct_h1":              float,
    "tlt_slope":                    float,
    "last_atr_h1_raw":              float,

    # Cached DI+/DI- (V3.0)
    "last_di_plus_h4":              (float, type(None)),
    "last_di_minus_h4":             (float, type(None)),

    # S1 pending
    "s1_pending_buy_ticket":        (int, type(None)),
    "s1_pending_sell_ticket":       (int, type(None)),

    # Reversal family
    "reversal_family_occupied":     bool,
    "s1b_pending_ticket":           (int, type(None)),

    # S1d
    "s1d_ema_touched_today":        bool,
    "s1d_fired_today":              bool,

    # KS4
    "ks4_reduced_trades_remaining": int,

    # S3
    "s3_sweep_candle_time":         (datetime, type(None)),
    "s3_sweep_low":                 float,
    "s3_sweep_high":                float,
    "s3_fired_today":               bool,

    # S6
    "s6_pending_buy_ticket":        (int, type(None)),
    "s6_pending_sell_ticket":       (int, type(None)),
    "s6_range_high":                float,
    "s6_range_low":                 float,
    "s6_fired_today":               bool,

    # S7
    "s7_pending_buy_ticket":        (int, type(None)),
    "s7_pending_sell_ticket":       (int, type(None)),
    "s7_prev_day_high":             float,
    "s7_prev_day_low":              float,
    "s7_fired_today":               bool,
    "s7_daily_atr":                 float,
    "high_corr_pairs":              list,

    # S8 signal/arm state (F7)
    "s8_armed":                     bool,
    "s8_arm_time":                  (datetime, type(None)),
    "s8_arm_candle_time":           (datetime, type(None)),
    "s8_spike_high":                float,
    "s8_spike_low":                 float,
    "s8_direction":                 (str, type(None)),
    "s8_fired_today":               bool,
    "s8_spike_atr":                 float,
    "s8_spike_candle_idx":          (int, type(None)),

    # S8 independent position lane (V3.0)
    "s8_open_ticket":               (int, type(None)),
    "s8_entry_price":               float,
    "s8_stop_price_original":       float,
    "s8_stop_price_current":        float,
    "s8_trade_direction":           (str, type(None)),
    "s8_be_activated":              bool,
    "s8_open_time_utc":             (str, type(None)),

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    # R3
    "r3_armed":                     bool,
    "r3_arm_time":                  (datetime, type(None)),
    "r3_direction":                 (str, type(None)),
    "r3_fired_today":               bool,
    "r3_open_ticket":               (int, type(None)),
    "r3_open_time":                 (datetime, type(None)),
    "r3_entry_price":               float,
    "r3_stop_price":                float,
    "r3_tp_price":                  float,
    "r3_event_scheduled_utc":       (datetime, type(None)),

    # S4
    "s4_ema_touched":               bool,
    "s4_fired_today":               bool,
    "s4_touch_bar_low":             float,
    "s4_touch_bar_high":            float,

    # S5
    "london_session_high":          float,
    "london_session_low":           float,
    "london_session_tracking_active": bool,
    "s5_compression_confirmed":     bool,
    "s5_fired_today":               bool,
    "d1_atr_14":                    float,
}


def validate_state_keys(state: dict) -> None:
    for key, expected_type in REQUIRED_STATE_KEYS.items():
        assert key in state, (
            f"STATE_KEY_MISSING: '{key}' not found in state. "
            f"Add it to both build_initial_state() and REQUIRED_STATE_KEYS."
        )
        assert isinstance(state[key], expected_type), (
            f"STATE_TYPE_ERROR: '{key}' expected {expected_type}, "
            f"got {type(state[key]).__name__} = {state[key]!r}"
        )
    extra_keys = set(state.keys()) - set(REQUIRED_STATE_KEYS.keys())
    if extra_keys:
        log_event("STATE_EXTRA_KEYS_WARNING",
                  keys=",".join(sorted(extra_keys)),
                  note="Add these to REQUIRED_STATE_KEYS")


def reset_daily_counters(state: dict) -> None:
    """
    Bug 2 fix: ALL daily state resets are in ONE place.
    This is the single source of truth — midnight_reset_job calls this
    and does NOT scatter additional resets across other files.

    RULE: consecutive_losses and peak_equity are NEVER reset here.
    A losing streak that spans midnight is still a streak.
    """
    # ── P&L ───────────────────────────────────────────────────────────────────
    state["daily_net_pnl_pct"]          = 0.0
    state["daily_commission_paid"]      = 0.0

    # ── Signal attempt counters ────────────────────────────────────────────────
    state["s1_family_attempts_today"]   = 0
    state["s1f_attempts_today"]         = 0

    # ── Spread warmup ─────────────────────────────────────────────────────────
    state["session_spread_initialized"] = False
    state["spread_fallback_active"]     = True
    state["spread_readings_count"]      = 0
    state["spread_elevated_reading_count"] = 0
    state["spread_elevated_last_ratio"] = 0.0
    state["spread_multiplier"]          = 1.0

    # ── Session time-kills ────────────────────────────────────────────────────
    state["london_tk_fired_today"]      = False
    state["ny_tk_fired_today"]          = False

    # ── M5 cycling ────────────────────────────────────────────────────────────
    state["consecutive_m5_losses"]      = 0

    # ── S1 family daily guards ────────────────────────────────────────────────
    state["s1d_ema_touched_today"]      = False
    state["s1d_fired_today"]            = False

    # ── S3 Stop Hunt — Bug 2 fix: ALL S3 resets here, not split across files ──
    state["s3_fired_today"]             = False
    state["s3_sweep_candle_time"]       = None
    state["s3_sweep_low"]               = 0.0
    state["s3_sweep_high"]              = 0.0

    # ── Reversal family reset ─────────────────────────────────────────────────
    # Only reset occupied flag — don't clear s1b_pending_ticket
    # (it may still be a valid resting order from yesterday's session)
    state["reversal_family_occupied"]   = False

    # ── S6 daily guard ────────────────────────────────────────────────────────
    state["s6_fired_today"]             = False

    # ── S7 daily guard ────────────────────────────────────────────────────────
    state["s7_fired_today"]             = False

    # ── S8 daily guard + arm state (V3.0: extended reset) ─────────────────────
    # NOTE: s8_open_ticket is NOT reset here.
    # An S8 position can be held overnight. Only reset daily signal flags.
    state["s8_fired_today"]             = False
    state["s8_armed"]                   = False
    state["s8_arm_time"]                = None
    state["s8_arm_candle_time"]         = None
    state["s8_spike_high"]              = 0.0
    state["s8_spike_low"]               = 0.0
    state["s8_direction"]               = None
    state["s8_spike_candle_idx"]        = None

    # ── Phase 2: R3 daily reset ───────────────────────────────────────────────
    state["r3_armed"]                   = False
    state["r3_arm_time"]                = None
    state["r3_direction"]               = None
    state["r3_fired_today"]             = False
    # NOTE: r3_open_ticket is NOT reset here.
    # If a trade is open at midnight (unlikely but possible), it stays tracked.

    # ── Phase 2: S4 daily reset ───────────────────────────────────────────────
    state["s4_ema_touched"]             = False
    state["s4_fired_today"]             = False
    state["s4_touch_bar_low"]           = 0.0
    state["s4_touch_bar_high"]          = 0.0

    # ── Phase 2: S5 + London session tracking reset ───────────────────────────
    state["s5_fired_today"]             = False
    state["s5_compression_confirmed"]   = False
    state["london_session_high"]        = 0.0
    state["london_session_low"]         = 0.0
    state["london_session_tracking_active"] = False
    # NOTE: d1_atr_14 is NOT reset here.
    # midnight_reset_job recomputes it fresh from MT5 D1 bars and stores it.

    log_event("DAILY_COUNTERS_RESET")
