"""
backtest/engine.py — Main replay loop for the backtesting framework.

BacktestEngine orchestrates:
  1. M5 bar iteration from HistoricalDataFeed
  2. Higher-TF candle building via BarBuffer
  3. Indicator computation (ADX, ATR, EMA, RSI) via pandas_ta
  4. Regime classification (reuses logic from engines/regime_engine.py)
  5. Strategy signal generation (S1, S1b, S1d, S1e, S1f, S2, S3, S4, S5,
     S6, S7, S8, R3) — all strategies now implemented
  6. Kill switches: KS3 (daily loss), KS4 (cooldown), KS5 (weekly loss),
     KS6 (drawdown), KS1/KS2 (spread — via regime)
  7. TLT macro bias gate — blocks direction not permitted by macro state
  8. Order fill simulation via ExecutionSimulator
  9. Position management (SL/TP, partial, BE, ATR trail)
  10. Equity curve tracking

CRITICAL DESIGN PRINCIPLE:
  Reuses the EXACT SAME thresholds and logic from config.py and
  engines/regime_engine.py wherever possible. Strategy implementations
  are simplified versions of the live signal_engine.py strategies.

PARITY GAPS FIXED IN THIS REVISION:
  GAP-1  Add S1b (failed breakout reversal) strategy evaluation
  GAP-2  Add S1d (M5 EMA20 pullback re-entry) strategy evaluation
  GAP-3  Add S1e (pyramid into confirmed winners) strategy evaluation
  GAP-4  Add S1f (post-time-kill NY re-entry) strategy evaluation
  GAP-5  Add S4 (London EMA20 pullback) strategy evaluation
  GAP-6  Add S5 (NY session compression breakout) strategy evaluation
  GAP-7  Add S8 (ATR spike continuation, independent lane) evaluation
  GAP-8  Add R3 (calendar momentum, independent lane) evaluation
  GAP-9  Enforce KS5 weekly loss kill switch
  GAP-10 Enforce KS6 account drawdown kill switch
  GAP-11 Apply TLT macro bias gate (blocks counter-bias direction entries)
  GAP-12 Evaluate S1/S1b on every M5 bar (not just M15 completion)
  GAP-13 Deduct $7/lot round-trip commission per trade (live broker cost)
  GAP-14 KS4 cooldown: cap s1 family attempts to 2 after 3 consecutive losses
  GAP-15 Weekly PnL tracking accumulated per trade for KS5 gate
  CRIT-4 Asian range now uses hard UTC 00:00-07:00 session filter instead of
         last-84-bars approximation, matching live calculate_asian_range()
         exactly.  HIGH-1 and HIGH-2 were already correctly implemented in
         execution_simulator.py (confirmed via full file read).

PARITY FIXES (v3.0 backtest sync):
  FIX-2  S3: UNSTABLE regime now ALLOWED (stop-hunt prime territory).
         Only NO_TRADE blocks S3. Mirrors live Change 3.3.
  FIX-3  S6: UNSTABLE regime now BLOCKED (noisy Asian ranges).
         Mirrors live Change 3.4.
  FIX-4  S6/S7: ADX/DI leg filter — when ADX>25 and DI ratio>1.3,
         counter-trend leg is suppressed. Mirrors live Changes 3.6/3.7.
  FIX-5  S1F: H1 EMA20 direction check added before M5 entry logic.
         Reject if price is on wrong side of H1 EMA20. Mirrors live
         Change 3.5.
  FIX-6  S4/S5: now use _can_trend_family_fire() — independent of
         s1_family_attempts_today. Mirrors live Change 3.9.
"""
import sys
import os
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Optional

from backtest.models import (
    SimOrder, SimPosition, TradeRecord, EquityPoint, SimulatedState,
)
# FIX Bug 1: compute_atr_percentile is defined in THIS file — do NOT import
# from data_feed (it doesn't exist there). Only import what data_feed provides.
from backtest.data_feed import (
    HistoricalDataFeed, BarBuffer, HistoricalSpreadFeed, HistoricalEventFeed,
)
from backtest.execution_simulator import ExecutionSimulator

logger = logging.getLogger("backtest.engine")

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    import config
except ImportError:
    config = None  # type: ignore


# -----------------------------------------------------------------------------
# REGIME CLASSIFICATION
# -----------------------------------------------------------------------------

def classify_regime_backtest(
    adx_h4: float,
    atr_pct_h1: float,
    session: str,
    has_upcoming_event: bool = False,
    spread_ratio: float = 1.0,
) -> tuple[str, float]:
    """
    Simplified regime classification for backtesting.
    Mirrors calculate_regime() from engines/regime_engine.py.
    Returns (regime_name, size_multiplier).
    """
    no_trade_thresh = getattr(config, "ATR_PCT_NO_TRADE_THRESHOLD", 95)
    unstable_thresh = getattr(config, "ATR_PCT_UNSTABLE_THRESHOLD", 85)
    super_thresh     = getattr(config, "ATR_PCT_SUPER_THRESHOLD", 55)
    ks2_spread_mult  = getattr(config, "KS2_SPREAD_MULTIPLIER", 2.5)

    if atr_pct_h1 > no_trade_thresh:  return "NO_TRADE", 0.0
    if has_upcoming_event:            return "NO_TRADE", 0.0
    if session == "OFF_HOURS":        return "NO_TRADE", 0.0
    if spread_ratio > ks2_spread_mult: return "NO_TRADE", 0.0

    if atr_pct_h1 > unstable_thresh or adx_h4 > 55 or 18 <= adx_h4 <= 20:
        return "UNSTABLE", 0.4

    if adx_h4 < 18:
        return "RANGING_CLEAR", 0.7

    session_mult = 1.0 if session in ("LONDON_NY_OVERLAP", "LONDON", "NY") else 0.7

    if adx_h4 > 35 and atr_pct_h1 > super_thresh:
        return "SUPER_TRENDING", round(1.2 * session_mult, 3)
    elif adx_h4 > 26:
        return "NORMAL_TRENDING", round(1.0 * session_mult, 3)
    else:
        return "WEAK_TRENDING", round(0.8 * session_mult, 3)


def compute_atr_percentile(atr_series: pd.Series, current_atr: float) -> float:
    """
    Compute ATR percentile using EWMA weighting (lambda=0.94).
    Defined here (NOT in data_feed) — this is the authoritative location.
    """
    values = atr_series.dropna().values
    if len(values) == 0:
        return 50.0
    n = len(values)
    lambda_decay = 0.94
    indices = np.arange(n)
    weights = np.power(lambda_decay, indices[::-1])
    weights = weights / weights.sum()
    below_current = (values < current_atr).astype(float)
    return float(np.dot(weights, below_current) * 100)


# -----------------------------------------------------------------------------
# STRATEGY HELPERS
# -----------------------------------------------------------------------------

def _compute_asian_range(bar_buffer: BarBuffer, current_time: datetime) -> Optional[dict]:
    """
    Compute Asian session range using a hard UTC 00:00-07:00 window.

    CRIT-4 FIX: The previous implementation used the last 84 M5 bars as a
    proxy for the Asian session.  That approximation drifts when bars are
    missing, spans session boundaries, or runs during the first bars after
    midnight — producing a range that does not correspond to the actual Asian
    session window.

    This implementation mirrors the live calculate_asian_range() from
    engines/session.py exactly:
      - Determine the current UTC trading date (the date of current_time in UTC).
      - Select only M5 bars whose timestamp falls in [00:00 UTC, 07:00 UTC)
        on that same date.
      - Require at least 12 qualifying bars (same floor as before) to guard
        against the range being computed from only a handful of thin bars.

    Falls back gracefully (returns None) when data is insufficient.
    """
    m5_df = bar_buffer.get_series("M5", count=100)
    if m5_df.empty:
        return None

    if isinstance(m5_df.index, pd.DatetimeIndex):
        ts_series = m5_df.index.to_series()
    elif "time" in m5_df.columns:
        ts_series = m5_df["time"]
    else:
        m5_df_fallback = m5_df.iloc[-84:]
        if len(m5_df_fallback) < 12:
            return None
        rh = float(m5_df_fallback["high"].max())
        rl = float(m5_df_fallback["low"].min())
        rs = rh - rl
        min_range = getattr(config, "MIN_RANGE_SIZE_PTS", 10)
        if rs < min_range:
            return None
        breakout_dist_pct  = getattr(config, "BREAKOUT_DIST_PCT", 0.12)
        hunt_threshold_pct = getattr(config, "HUNT_THRESHOLD_PCT", 0.08)
        return {
            "range_high": rh, "range_low": rl, "range_size": rs,
            "breakout_dist": rs * breakout_dist_pct,
            "hunt_threshold": rs * hunt_threshold_pct,
        }

    if hasattr(ts_series.dtype, "tz") and ts_series.dtype.tz is not None:
        ts_utc = ts_series.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_utc = ts_series

    if current_time.tzinfo is not None:
        ct_utc = current_time.astimezone(pytz.utc).replace(tzinfo=None)
    else:
        ct_utc = current_time

    trading_date = ct_utc.date()

    session_start = datetime(trading_date.year, trading_date.month, trading_date.day, 0, 0, 0)
    session_end   = datetime(trading_date.year, trading_date.month, trading_date.day, 7, 0, 0)

    mask = (ts_utc >= session_start) & (ts_utc < session_end)
    asian_bars = m5_df[mask.values]

    if len(asian_bars) < 12:
        return None

    rh = float(asian_bars["high"].max())
    rl = float(asian_bars["low"].min())
    rs = rh - rl

    min_range = getattr(config, "MIN_RANGE_SIZE_PTS", 10)
    if rs < min_range:
        return None

    breakout_dist_pct  = getattr(config, "BREAKOUT_DIST_PCT", 0.12)
    hunt_threshold_pct = getattr(config, "HUNT_THRESHOLD_PCT", 0.08)
    return {
        "range_high": rh, "range_low": rl, "range_size": rs,
        "breakout_dist": rs * breakout_dist_pct,
        "hunt_threshold": rs * hunt_threshold_pct,
    }


def _get_prev_day_hl(bar_buffer: BarBuffer) -> Optional[dict]:
    d1_df = bar_buffer.get_series("D1")
    if len(d1_df) < 2:
        return None
    prev = d1_df.iloc[-1]
    return {
        "high": float(prev["high"]),
        "low":  float(prev["low"]),
        "range": float(prev["high"] - prev["low"]),
    }


# -----------------------------------------------------------------------------
# GAP-11: MACRO BIAS GATE HELPER
# In live system, calculate_macro_bias() runs at 09:00 IST daily and sets
# state["macro_bias"] to "LONG_ONLY", "SHORT_ONLY", or "BOTH_PERMITTED".
# The backtest cannot call the live TLT feed, so we approximate:
#   - If EMA20_H1 is rising (close > EMA, ADX > 20, DI+ > DI-) -> LONG_ONLY
#   - If EMA20_H1 is falling (close < EMA, ADX > 20, DI- > DI+) -> SHORT_ONLY
#   - Otherwise -> BOTH_PERMITTED
# This is a structural proxy that captures the same market regime the TLT
# filter targets: don't trade against the prevailing trend direction.
# -----------------------------------------------------------------------------

def _derive_macro_bias(
    price: float,
    ema20_h1: Optional[float],
    adx_h4: float,
    di_plus: Optional[float],
    di_minus: Optional[float],
) -> str:
    """
    GAP-11: Proxy for TLT macro bias.
    Returns 'LONG_ONLY', 'SHORT_ONLY', or 'BOTH_PERMITTED'.
    """
    if ema20_h1 is None or di_plus is None or di_minus is None:
        return "BOTH_PERMITTED"
    if adx_h4 < 20:
        return "BOTH_PERMITTED"
    if price > ema20_h1 and di_plus > di_minus:
        return "LONG_ONLY"
    if price < ema20_h1 and di_minus > di_plus:
        return "SHORT_ONLY"
    return "BOTH_PERMITTED"


def _macro_permits(direction: str, macro_bias: str) -> bool:
    """Return True if macro_bias allows this direction."""
    if macro_bias == "BOTH_PERMITTED":
        return True
    if macro_bias == "LONG_ONLY" and direction == "LONG":
        return True
    if macro_bias == "SHORT_ONLY" and direction == "SHORT":
        return True
    return False


# -----------------------------------------------------------------------------
# FIX-6: SHARED TREND FAMILY GATE FOR S4/S5
# S4 and S5 are trend-following strategies that belong to the trend family
# but should NOT consume s1_family_attempts_today. They use their own
# fired_today flags and check trend_family_occupied independently.
# Mirrors live Change 3.9: _can_trend_family_fire().
# -----------------------------------------------------------------------------

def _can_trend_family_fire(state: "SimulatedState") -> bool:
    """
    FIX-6: Shared gate for S4/S5 — checks trend_family_occupied + regime.
    Does NOT check s1_family_attempts_today (that counter is S1/S1B only).
    Returns True if the trend family slot is free and regime allows trading.
    """
    if state.trend_family_occupied:
        return False
    if state.current_regime in ("NO_TRADE", "RANGING_CLEAR"):
        return False
    return True


# -----------------------------------------------------------------------------
# S1 — LONDON RANGE BREAKOUT
# -----------------------------------------------------------------------------

def _evaluate_s1(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    S1: London Range Breakout.
    Entry: BUY_STOP/SELL_STOP at range boundary + breakout_dist.
    Stop:  ATR-based buffer. TP: 2.5R.
    GAP-11: macro bias gate applied.
    GAP-12: called on every M5 bar (not just M15 completion).
    """
    orders: list[SimOrder] = []
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING", "WEAK_TRENDING"):
        return orders
    max_attempts = getattr(config, "MAX_S1_FAMILY_ATTEMPTS", 4)
    # GAP-14: KS4 cooldown — after 3 consecutive losses cap attempts at 2
    if state.consecutive_losses >= 3:
        max_attempts = min(max_attempts, 2)
    if state.s1_family_attempts_today >= max_attempts:
        return orders
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if london_time.hour < 8:
        return orders
    if london_time.hour > 16 or (london_time.hour == 16 and london_time.minute >= 30):
        return orders
    if not state.range_computed:
        return orders
    rh = state.range_high
    rl = state.range_low
    rs = state.range_size
    if rs < getattr(config, "MIN_RANGE_SIZE_PTS", 10):
        return orders
    breakout_dist = rs * getattr(config, "BREAKOUT_DIST_PCT", 0.12)
    last_m15 = bar_buffer.get_last_bar("M15")
    if last_m15 is None:
        return orders
    close = last_m15["close"]
    stop_buffer = max(atr_h1 * 0.3, 5.0) if atr_h1 > 0 else rs * 0.15
    direction = None
    if close > rh + breakout_dist:
        direction = "LONG"
    elif close < rl - breakout_dist:
        direction = "SHORT"
    if direction is None:
        return orders
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        logger.debug(f"S1 {direction} blocked by macro_bias={macro_bias}")
        return orders
    if direction == "LONG":
        entry = round(rh + breakout_dist, 3)
        stop  = round(rl - stop_buffer, 3)
    else:
        entry = round(rl - breakout_dist, 3)
        stop  = round(rh + stop_buffer, 3)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return orders
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    tp = round(entry + stop_dist * 2.5, 3) if direction == "LONG" else round(entry - stop_dist * 2.5, 3)
    expiry_london = london_time.replace(hour=16, minute=30, second=0)
    expiry_utc    = expiry_london.astimezone(pytz.utc)
    orders.append(SimOrder(
        strategy="S1_LONDON_BRK", direction=direction,
        order_type="BUY_STOP" if direction == "LONG" else "SELL_STOP",
        price=entry, sl=stop, tp=tp, lots=lots,
        expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# GAP-1: S1b — FAILED BREAKOUT REVERSAL
# -----------------------------------------------------------------------------

def _evaluate_s1b(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-1: S1b Failed Breakout Reversal.
    Requires failed_breakout_flag to be set on state (set by _check_s1b_trigger).
    """
    orders: list[SimOrder] = []
    if not state.failed_breakout_flag:
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if london_time.hour > 16 or (london_time.hour == 16 and london_time.minute >= 30):
        return orders
    max_attempts = getattr(config, "MAX_S1_FAMILY_ATTEMPTS", 4)
    if state.consecutive_losses >= 3:
        max_attempts = min(max_attempts, 2)
    if state.s1_family_attempts_today >= max_attempts:
        return orders
    if not state.range_computed:
        return orders
    rh = state.range_high
    rl = state.range_low
    rs = state.range_size
    bd = rs * getattr(config, "BREAKOUT_DIST_PCT", 0.12)
    last_m15 = bar_buffer.get_last_bar("M15")
    if last_m15 is None:
        return orders
    orig_dir  = state.failed_breakout_direction
    direction = None
    if orig_dir == "LONG"  and last_m15["close"] < rl - bd:
        direction = "SHORT"
    elif orig_dir == "SHORT" and last_m15["close"] > rh + bd:
        direction = "LONG"
    if direction is None:
        return orders
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    if direction == "LONG":
        entry = round(rh + bd, 3)
        stop  = round(rl - rs * 0.10, 3)
    else:
        entry = round(rl - bd, 3)
        stop  = round(rh + rs * 0.10, 3)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return orders
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    expiry_utc = current_time.astimezone(london_tz).replace(hour=16, minute=30, second=0).astimezone(pytz.utc)
    orders.append(SimOrder(
        strategy="S1B_FAILED_BRK", direction=direction,
        order_type="BUY_STOP" if direction == "LONG" else "SELL_STOP",
        price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    # Reset flag immediately after signal generated
    state.failed_breakout_flag      = False
    state.failed_breakout_direction = None
    return orders


def _check_s1b_trigger(bar_buffer: BarBuffer, state: "SimulatedState") -> None:
    """
    GAP-1: Check whether to SET failed_breakout_flag.
    Called on every M15 bar close. Condition: last S1 reached -0.5R
    and M15 then closes BACK INSIDE the Asian range.
    """
    if state.failed_breakout_flag:
        return
    if state.last_s1_max_r > -0.5:
        return
    if not state.range_computed:
        return
    last_m15 = bar_buffer.get_last_bar("M15")
    if last_m15 is None:
        return
    rh = state.range_high
    rl = state.range_low
    if rl < last_m15["close"] < rh:
        state.failed_breakout_flag      = True
        state.failed_breakout_direction = state.last_s1_direction
        logger.debug(f"S1b flag SET: orig_dir={state.last_s1_direction} max_r={state.last_s1_max_r:.2f}")


# -----------------------------------------------------------------------------
# GAP-2: S1d — M5 EMA20 PULLBACK RE-ENTRY
# -----------------------------------------------------------------------------

def _evaluate_s1d(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_m15: float,
    ema20_m15: Optional[float],
) -> list[SimOrder]:
    """
    GAP-2: S1d M5 EMA20 pullback re-entry.
    Requires: open S1 family position, EMA20 on M15, M5 body close in direction.
    Max re-entries: SUPER=8, NORMAL=5. Size = 0.5x normal lots.
    """
    orders: list[SimOrder] = []
    if not state.trend_family_occupied:
        return orders
    if state.trend_family_strategy not in (
        "S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK", "S1E_PYRAMID",
    ):
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if london_time.hour > 16 or (london_time.hour == 16 and london_time.minute >= 30):
        return orders
    if ema20_m15 is None or atr_m15 <= 0:
        return orders
    max_m5 = 8 if state.current_regime == "SUPER_TRENDING" else 5
    if state.s1d_pyramid_count >= max_m5:
        return orders
    direction = state.last_s1_direction
    if not direction:
        return orders
    last_m5 = bar_buffer.get_last_bar("M5")
    if last_m5 is None:
        return orders
    # G3 Fix: full body close (both open AND close) in the trend direction
    if direction == "LONG":
        if not (last_m5["open"] > ema20_m15 and last_m5["close"] > ema20_m15):
            return orders
        entry = round(ema20_m15, 3)
        stop_dist = max(atr_m15 * 0.75, getattr(config, "S1D_STOP_POINTS_MIN", 8.0))
        stop = round(entry - stop_dist, 3)
    else:
        if not (last_m5["open"] < ema20_m15 and last_m5["close"] < ema20_m15):
            return orders
        entry = round(ema20_m15, 3)
        stop_dist = max(atr_m15 * 0.75, getattr(config, "S1D_STOP_POINTS_MIN", 8.0))
        stop = round(entry + stop_dist, 3)
    base_lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    lots = max(0.01, round(base_lots * 0.5, 2))
    expiry_utc = current_time + timedelta(minutes=getattr(config, "M5_LIMIT_EXPIRY_MIN", 5))
    orders.append(SimOrder(
        strategy="S1D_PYRAMID", direction=direction,
        order_type="BUY_LIMIT" if direction == "LONG" else "SELL_LIMIT",
        price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# GAP-3: S1e — PYRAMID INTO CONFIRMED WINNERS
# -----------------------------------------------------------------------------

def _evaluate_s1e(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, ema20_m15: Optional[float],
) -> list[SimOrder]:
    """
    GAP-3: S1e pyramid add after partial exit + BE activation.
    Hard limit: one pyramid per S1 trade. Size = 0.5x original lots.
    """
    orders: list[SimOrder] = []
    if state.s1e_pyramid_done:
        return orders
    if not state.position_partial_done:
        return orders
    if not state.position_be_activated:
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING"):
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if london_time.hour > 16 or (london_time.hour == 16 and london_time.minute >= 30):
        return orders
    if ema20_m15 is None:
        return orders
    direction = state.last_s1_direction
    if not direction:
        return orders
    last_m5 = bar_buffer.get_last_bar("M5")
    if last_m5 is None:
        return orders
    # G3 Fix: full body close in direction
    if direction == "LONG":
        if not (last_m5["open"] > ema20_m15 and last_m5["close"] > ema20_m15):
            return orders
        entry = round(ema20_m15, 3)
        stop  = state.stop_price_current if state.stop_price_current > 0 else round(ema20_m15 - 10.0, 3)
    else:
        if not (last_m5["open"] < ema20_m15 and last_m5["close"] < ema20_m15):
            return orders
        entry = round(ema20_m15, 3)
        stop  = state.stop_price_current if state.stop_price_current > 0 else round(ema20_m15 + 10.0, 3)
    stop_dist = abs(entry - stop)
    if stop_dist < 1.0:
        return orders
    original_lots = state.original_lot_size if state.original_lot_size > 0 else 0.01
    lots = max(0.01, round(original_lots * 0.5, 2))
    expiry_utc = current_time + timedelta(minutes=getattr(config, "M5_LIMIT_EXPIRY_MIN", 5))
    orders.append(SimOrder(
        strategy="S1E_PYRAMID", direction=direction,
        order_type="BUY_LIMIT" if direction == "LONG" else "SELL_LIMIT",
        price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# GAP-4: S1f — POST-TIME-KILL NY RE-ENTRY
# FIX-5: Added H1 EMA20 direction check (mirrors live Change 3.5).
# -----------------------------------------------------------------------------

def _evaluate_s1f(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, ema20_m15: Optional[float],
    macro_bias: str = "BOTH_PERMITTED",
    ema20_h1: Optional[float] = None,
) -> list[SimOrder]:
    """
    GAP-4: S1f post-time-kill NY re-entry.
    Requires London TK to have fired today. SUPER/NORMAL only. Max 1/day.

    FIX-5: H1 EMA20 direction check added as macro safety gate.
    Rejects S1F if current price is on the wrong side of H1 EMA20,
    indicating the trend may have reversed since S1 fired.
    H1 = macro safety gate. M5 = entry timing. Both required.
    """
    orders: list[SimOrder] = []
    # London TK must have fired — we check by whether it's past 16:30 London time
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if not (london_time.hour > 16 or (london_time.hour == 16 and london_time.minute >= 30)):
        return orders
    if state.current_session not in ("NY", "LONDON_NY_OVERLAP"):
        return orders
    ny_tz   = pytz.timezone("America/New_York")
    ny_time = current_time.astimezone(ny_tz)
    if ny_time.hour >= 13:
        return orders
    if state.s1f_attempts_today >= 1:
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING"):
        return orders
    if state.trend_family_occupied:
        return orders
    if ema20_m15 is None:
        return orders
    direction = state.last_s1_direction
    if not direction:
        return orders
    # FIX-5: H1 EMA20 direction validation — reject if trend reversed
    # This is the macro safety gate. If price has crossed the H1 EMA20
    # in the wrong direction, the original S1 trend is likely over.
    if ema20_h1 is not None:
        last_m5 = bar_buffer.get_last_bar("M5")
        if last_m5 is not None:
            current_price = last_m5["close"]
            if direction == "LONG" and current_price < ema20_h1:
                logger.debug(
                    f"S1F rejected: LONG but price {current_price:.3f} < H1 EMA20 {ema20_h1:.3f}"
                )
                return orders
            elif direction == "SHORT" and current_price > ema20_h1:
                logger.debug(
                    f"S1F rejected: SHORT but price {current_price:.3f} > H1 EMA20 {ema20_h1:.3f}"
                )
                return orders
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    last_m5 = bar_buffer.get_last_bar("M5")
    if last_m5 is None:
        return orders
    # G3 Fix: full body close in direction
    s1f_stop_pts = getattr(config, "S1F_STOP_POINTS", 15.0)
    if direction == "LONG":
        if not (last_m5["open"] > ema20_m15 and last_m5["close"] > ema20_m15):
            return orders
        entry = round(ema20_m15, 3)
        stop  = round(entry - s1f_stop_pts, 3)
    else:
        if not (last_m5["open"] < ema20_m15 and last_m5["close"] < ema20_m15):
            return orders
        entry = round(ema20_m15, 3)
        stop  = round(entry + s1f_stop_pts, 3)
    stop_dist = abs(entry - stop)
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    expiry_utc = current_time + timedelta(minutes=getattr(config, "M5_LIMIT_EXPIRY_MIN", 5))
    orders.append(SimOrder(
        strategy="S1F_POST_TK", direction=direction,
        order_type="BUY_LIMIT" if direction == "LONG" else "SELL_LIMIT",
        price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    state.s1f_attempts_today += 1
    return orders


# -----------------------------------------------------------------------------
# S2 — MEAN REVERSION
# -----------------------------------------------------------------------------

def _evaluate_s2(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    ema20_h1: Optional[float], rsi_h1: Optional[float],
) -> list[SimOrder]:
    """S2: Mean Reversion — RANGING_CLEAR only."""
    orders: list[SimOrder] = []
    if state.current_regime != "RANGING_CLEAR":
        return orders
    if state.s2_fired_today:
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    # G7: not within 30 min of London open
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    total_min   = london_time.hour * 60 + london_time.minute
    if 7 * 60 + 30 <= total_min < 8 * 60:
        return orders
    if ema20_h1 is None or atr_h1 <= 0 or rsi_h1 is None:
        return orders
    last_m15 = bar_buffer.get_last_bar("M15")
    if last_m15 is None:
        return orders
    price     = last_m15["close"]
    deviation = abs(price - ema20_h1)
    if deviation < atr_h1 * 1.5:
        return orders
    if price < ema20_h1 and rsi_h1 < 35:
        direction = "LONG"
    elif price > ema20_h1 and rsi_h1 > 65:
        direction = "SHORT"
    else:
        return orders
    entry     = round(price, 3)
    stop_dist = atr_h1 * 2.0
    stop = round(entry - stop_dist, 3) if direction == "LONG" else round(entry + stop_dist, 3)
    tp   = round(ema20_h1, 3)
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    expiry_utc  = london_time.replace(hour=16, minute=30, second=0).astimezone(pytz.utc)
    orders.append(SimOrder(
        strategy="S2_MEAN_REV", direction=direction,
        order_type="MARKET", price=entry, sl=stop, tp=tp, lots=lots,
        expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# S3 — STOP HUNT REVERSAL
# FIX-2: UNSTABLE regime now ALLOWED. Only NO_TRADE blocks S3.
# Rationale: UNSTABLE (85-95th ATR pct) is prime stop-hunt territory.
# The 0.4x size multiplier already reduces risk in UNSTABLE.
# -----------------------------------------------------------------------------

def _evaluate_s3(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
) -> list[SimOrder]:
    """
    S3: Stop Hunt Reversal — sweep + reclaim pattern.

    FIX-2: UNSTABLE regime is now ALLOWED (was previously blocked).
    Only NO_TRADE blocks S3. UNSTABLE is prime stop-hunt territory
    and the 0.4x size multiplier in UNSTABLE already limits risk.
    Mirrors live Change 3.3.
    """
    orders: list[SimOrder] = []
    # FIX-2: block only NO_TRADE (removed UNSTABLE from block list)
    if state.current_regime == "NO_TRADE":
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    if state.s3_fired_today:
        return orders
    if not state.range_computed or atr_h1 <= 0:
        return orders
    sweep_thresh   = getattr(config, "S3_SWEEP_THRESHOLD_ATR", 0.3)
    stop_mult      = getattr(config, "S3_STOP_ATR_MULT", 0.5)
    reclaim_offset = getattr(config, "S3_RECLAIM_OFFSET_PTS", 2.0)
    window         = getattr(config, "S3_WINDOW_CANDLES", 3)
    m15_df = bar_buffer.get_series("M15", count=200)
    if len(m15_df) < window + 1:
        return orders
    recent = m15_df.iloc[-(window + 1):]
    rl = state.range_low
    rh = state.range_high
    # LONG: sweep below range_low then reclaim
    sweep_low   = float(recent["low"].min())
    sweep_depth = rl - sweep_low
    if sweep_depth >= atr_h1 * sweep_thresh:
        last_bar = recent.iloc[-1]
        if last_bar["close"] > rl:
            entry     = round(float(last_bar["high"]) + reclaim_offset, 3)
            stop      = round(sweep_low - atr_h1 * stop_mult, 3)
            stop_dist = abs(entry - stop)
            if stop_dist > 0:
                lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
                orders.append(SimOrder(
                    strategy="S3_STOP_HUNT_REV", direction="LONG",
                    order_type="BUY_STOP", price=entry, sl=stop,
                    tp=round(entry + stop_dist * 2.0, 3), lots=lots,
                    expiry=current_time + timedelta(hours=2), placed_time=current_time,
                ))
    # SHORT: sweep above range_high then reclaim
    sweep_high       = float(recent["high"].max())
    sweep_depth_high = sweep_high - rh
    if sweep_depth_high >= atr_h1 * sweep_thresh:
        last_bar = recent.iloc[-1]
        if last_bar["close"] < rh:
            entry     = round(float(last_bar["low"]) - reclaim_offset, 3)
            stop      = round(sweep_high + atr_h1 * stop_mult, 3)
            stop_dist = abs(entry - stop)
            if stop_dist > 0:
                lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
                orders.append(SimOrder(
                    strategy="S3_STOP_HUNT_REV", direction="SHORT",
                    order_type="SELL_STOP", price=entry, sl=stop,
                    tp=round(entry - stop_dist * 2.0, 3), lots=lots,
                    expiry=current_time + timedelta(hours=2), placed_time=current_time,
                ))
    return orders


# -----------------------------------------------------------------------------
# GAP-5: S4 — LONDON EMA20 PULLBACK
# FIX-6: Now uses _can_trend_family_fire() — does NOT consume
# s1_family_attempts_today. S4 has its own s4_fired_today gate.
# -----------------------------------------------------------------------------

def _evaluate_s4(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    ema20_h1: Optional[float], di_plus: Optional[float],
    di_minus: Optional[float], macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-5: S4 London EMA20 Pullback.
    Entry: M15 close within 0.5xATR of EMA20_H1 in trend direction.
    Stop: 1.5xATR beyond EMA20. TP: 2R.

    FIX-6: Uses _can_trend_family_fire() instead of S1 attempt counter.
    S4 does not consume s1_family_attempts_today. Mirrors live Change 3.9.
    """
    orders: list[SimOrder] = []
    if state.s4_fired_today:
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING"):
        return orders
    # FIX-6: use shared trend family gate (does NOT touch s1_family_attempts_today)
    if not _can_trend_family_fire(state):
        return orders
    if ema20_h1 is None or atr_h1 <= 0:
        return orders
    if di_plus is None or di_minus is None:
        return orders
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if london_time.hour < 8 or london_time.hour > 15:
        return orders
    # Determine trend direction via DI+/DI-
    direction = "LONG" if di_plus > di_minus else "SHORT"
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    last_m15 = bar_buffer.get_last_bar("M15")
    if last_m15 is None:
        return orders
    price     = last_m15["close"]
    proximity = abs(price - ema20_h1)
    if proximity > atr_h1 * 0.5:
        return orders
    # Confirm price is on correct side of EMA for a pullback
    if direction == "LONG" and price < ema20_h1 - atr_h1 * 0.1:
        return orders
    if direction == "SHORT" and price > ema20_h1 + atr_h1 * 0.1:
        return orders
    stop_dist = atr_h1 * 1.5
    if direction == "LONG":
        entry = round(price, 3)
        stop  = round(ema20_h1 - stop_dist, 3)
    else:
        entry = round(price, 3)
        stop  = round(ema20_h1 + stop_dist, 3)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return orders
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    expiry_utc = london_time.replace(hour=16, minute=30, second=0).astimezone(pytz.utc)
    orders.append(SimOrder(
        strategy="S4_EMA_PULLBACK", direction=direction,
        order_type="MARKET", price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# GAP-6: S5 — NY SESSION COMPRESSION BREAKOUT
# FIX-6: Now uses _can_trend_family_fire() — does NOT consume
# s1_family_attempts_today. S5 has its own s5_fired_today gate.
# -----------------------------------------------------------------------------

def _evaluate_s5(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-6: S5 NY Session Compression Breakout.
    Requires NY session (13:00-17:00 UTC), NORMAL/SUPER regime.
    Looks for tight 2-hour London consolidation and breakout.

    FIX-6: Uses _can_trend_family_fire() instead of S1 attempt counter.
    S5 does not consume s1_family_attempts_today. Mirrors live Change 3.9.
    """
    orders: list[SimOrder] = []
    if state.s5_fired_today:
        return orders
    if state.current_session not in ("NY", "LONDON_NY_OVERLAP"):
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING"):
        return orders
    # FIX-6: use shared trend family gate (does NOT touch s1_family_attempts_today)
    if not _can_trend_family_fire(state):
        return orders
    utc_hour = current_time.hour
    if not (13 <= utc_hour <= 17):
        return orders
    if atr_h1 <= 0:
        return orders
    m15_df = bar_buffer.get_series("M15", count=48)
    if len(m15_df) < 24:
        return orders
    ny_range_df = m15_df.iloc[-8:]
    ny_high = float(ny_range_df["high"].max())
    ny_low  = float(ny_range_df["low"].min())
    ny_range = ny_high - ny_low
    if ny_range > atr_h1 * 0.5:
        return orders
    if ny_range <= 0:
        return orders
    last_bar = m15_df.iloc[-1]
    price    = last_bar["close"]
    entry_offset = ny_range * 0.1
    direction = None
    if price > ny_high + entry_offset:
        direction = "LONG"
    elif price < ny_low - entry_offset:
        direction = "SHORT"
    if direction is None:
        return orders
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    stop_dist = atr_h1 * 1.0
    if direction == "LONG":
        entry = round(price, 3)
        stop  = round(ny_low - atr_h1 * 0.3, 3)
    else:
        entry = round(price, 3)
        stop  = round(ny_high + atr_h1 * 0.3, 3)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return orders
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    expiry_utc = current_time.astimezone(pytz.utc).replace(hour=17, minute=0, second=0, microsecond=0)
    orders.append(SimOrder(
        strategy="S5_NY_COMPRESS", direction=direction,
        order_type="MARKET", price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# S6 — ASIAN RANGE BREAKOUT OCO
# FIX-3: UNSTABLE regime now BLOCKED (noisy Asian ranges).
# FIX-4: ADX/DI leg filter — suppress counter-trend leg when ADX>25
#         and DI ratio>1.3. Mirrors live Changes 3.4 and 3.6.
# -----------------------------------------------------------------------------

def _evaluate_s6(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_m15: float,
    adx_h4: Optional[float] = None,
    di_plus: Optional[float] = None,
    di_minus: Optional[float] = None,
) -> list[SimOrder]:
    """
    S6: Asian Range Breakout OCO pair.

    FIX-3: UNSTABLE regime is now BLOCKED (was only blocking NO_TRADE).
    In UNSTABLE, Asian session ranges are noisy and breakout levels are
    unreliable. Mirrors live Change 3.4.

    FIX-4: ADX/DI leg filter applied before placing each OCO leg.
    When ADX>25 and DI ratio>1.3, the counter-trend leg is suppressed.
    Mirrors live Change 3.6.
    """
    orders: list[SimOrder] = []
    if state.s6_placed_today:
        return orders
    # FIX-3: block both NO_TRADE and UNSTABLE
    if state.current_regime in ("NO_TRADE", "UNSTABLE"):
        return orders
    utc_hour = current_time.hour
    utc_min  = current_time.minute
    if not (0 <= utc_hour < 5 or (utc_hour == 5 and utc_min <= 30)):
        return orders
    if not state.range_computed:
        return orders
    rh = state.range_high
    rl = state.range_low
    rs = state.range_size
    if rs < getattr(config, "S6_MIN_RANGE_PTS", 8.0):
        return orders
    dist_pts  = getattr(config, "S6_BREAKOUT_DIST_PTS", 2.0)
    stop_mult = getattr(config, "S6_STOP_ATR_MULT", 0.5)
    stop_dist = max(atr_m15 * stop_mult, 5.0) if atr_m15 > 0 else rs * 0.3
    buy_entry  = round(rh + dist_pts, 3)
    buy_stop   = round(buy_entry - stop_dist, 3)
    buy_lots   = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    sell_entry = round(rl - dist_pts, 3)
    sell_stop  = round(sell_entry + stop_dist, 3)
    sell_lots  = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    expiry = current_time.replace(hour=8, minute=0, second=0)
    if expiry <= current_time:
        expiry += timedelta(days=1)
    # FIX-4: ADX/DI leg filter — determine which legs are permitted
    place_buy  = True
    place_sell = True
    if adx_h4 is not None and adx_h4 > 25 and di_plus is not None and di_minus is not None:
        di_ratio = (di_plus / di_minus) if di_minus > 0 else 999.0
        if di_ratio > 1.3:
            # Strong uptrend: suppress sell (counter-trend) leg
            place_sell = False
            logger.debug(f"S6 sell leg filtered: strong uptrend ADX={adx_h4:.1f} DI_ratio={di_ratio:.2f}")
        elif di_ratio < (1.0 / 1.3):  # ~0.769
            # Strong downtrend: suppress buy (counter-trend) leg
            place_buy = False
            logger.debug(f"S6 buy leg filtered: strong downtrend ADX={adx_h4:.1f} DI_ratio={di_ratio:.2f}")
    if place_buy:
        orders.append(SimOrder(
            strategy="S6_ASIAN_BRK", direction="LONG", order_type="BUY_STOP",
            price=buy_entry, sl=buy_stop, tp=round(buy_entry + stop_dist * 2.0, 3),
            lots=buy_lots, expiry=expiry, placed_time=current_time,
            tag="s6_buy_leg", linked_tag="s6_sell_leg" if place_sell else None,
        ))
    if place_sell:
        orders.append(SimOrder(
            strategy="S6_ASIAN_BRK", direction="SHORT", order_type="SELL_STOP",
            price=sell_entry, sl=sell_stop, tp=round(sell_entry - stop_dist * 2.0, 3),
            lots=sell_lots, expiry=expiry, placed_time=current_time,
            tag="s6_sell_leg", linked_tag="s6_buy_leg" if place_buy else None,
        ))
    return orders


# -----------------------------------------------------------------------------
# S7 — DAILY STRUCTURE BREAKOUT OCO
# FIX-4: ADX/DI leg filter applied. Mirrors live Change 3.7.
# -----------------------------------------------------------------------------

def _evaluate_s7(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, daily_atr: Optional[float],
    adx_h4: Optional[float] = None,
    di_plus: Optional[float] = None,
    di_minus: Optional[float] = None,
) -> list[SimOrder]:
    """
    S7: Daily Structure Breakout OCO pair.

    FIX-4: ADX/DI leg filter applied before placing each OCO leg.
    When ADX>25 and DI ratio>1.3, the counter-trend leg is suppressed.
    Mirrors live Change 3.7.
    """
    orders: list[SimOrder] = []
    if state.s7_placed_today:
        return orders
    if state.current_regime == "NO_TRADE":
        return orders
    if current_time.hour != 0 or current_time.minute > 15:
        return orders
    prev_day = _get_prev_day_hl(bar_buffer)
    if prev_day is None:
        return orders
    ph         = prev_day["high"]
    pl         = prev_day["low"]
    prev_range = prev_day["range"]
    min_ratio  = getattr(config, "S7_MIN_RANGE_ATR_RATIO", 0.75)
    if daily_atr and prev_range < daily_atr * min_ratio:
        return orders
    entry_offset = getattr(config, "S7_ENTRY_OFFSET_PTS", 5.0)
    size_mult    = getattr(config, "S7_SIZE_MULTIPLIER", 0.5)
    stop_dist    = daily_atr * 0.5 if daily_atr else prev_range * 0.5
    buy_entry  = round(ph + entry_offset, 3)
    sell_entry = round(pl - entry_offset, 3)
    expiry = current_time.replace(hour=23, minute=59, second=0)
    # FIX-4: ADX/DI leg filter — determine which legs are permitted
    place_buy  = True
    place_sell = True
    if adx_h4 is not None and adx_h4 > 25 and di_plus is not None and di_minus is not None:
        di_ratio = (di_plus / di_minus) if di_minus > 0 else 999.0
        if di_ratio > 1.3:
            place_sell = False
            logger.debug(f"S7 sell leg filtered: strong uptrend ADX={adx_h4:.1f} DI_ratio={di_ratio:.2f}")
        elif di_ratio < (1.0 / 1.3):  # ~0.769
            place_buy = False
            logger.debug(f"S7 buy leg filtered: strong downtrend ADX={adx_h4:.1f} DI_ratio={di_ratio:.2f}")
    if place_buy:
        orders.append(SimOrder(
            strategy="S7_DAILY_STRUCT", direction="LONG", order_type="BUY_STOP",
            price=buy_entry, sl=round(buy_entry - stop_dist, 3),
            lots=_calculate_lot_size(state.balance, stop_dist, state.size_multiplier * size_mult),
            expiry=expiry, placed_time=current_time,
            tag="s7_buy_leg", linked_tag="s7_sell_leg" if place_sell else None,
        ))
    if place_sell:
        orders.append(SimOrder(
            strategy="S7_DAILY_STRUCT", direction="SHORT", order_type="SELL_STOP",
            price=sell_entry, sl=round(sell_entry + stop_dist, 3),
            lots=_calculate_lot_size(state.balance, stop_dist, state.size_multiplier * size_mult),
            expiry=expiry, placed_time=current_time,
            tag="s7_sell_leg", linked_tag="s7_buy_leg" if place_buy else None,
        ))
    return orders


# -----------------------------------------------------------------------------
# GAP-7: S8 — ATR SPIKE CONTINUATION (INDEPENDENT LANE)
# -----------------------------------------------------------------------------

def _evaluate_s8(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-7: S8 ATR Spike Continuation.
    Independent lane: fires alongside trend family if conditions met.
    Trigger: M15 bar range >= 2xATR(H1). Confirmation: next bar closes in same direction.
    Stop: 0.5xATR beyond spike low/high. TP: 1.5R. Max 1 per day.
    """
    orders: list[SimOrder] = []
    if state.s8_fired_today:
        return orders
    if state.current_regime in ("NO_TRADE", "RANGING_CLEAR"):
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP", "NY"):
        return orders
    if atr_h1 <= 0:
        return orders
    m15_df = bar_buffer.get_series("M15", count=5)
    if len(m15_df) < 3:
        return orders
    spike_bar   = m15_df.iloc[-3]
    confirm_bar = m15_df.iloc[-2]
    current_bar = m15_df.iloc[-1]
    spike_range = abs(spike_bar["high"] - spike_bar["low"])
    spike_thresh = getattr(config, "S8_SPIKE_ATR_MULTIPLIER", 2.0)
    if spike_range < atr_h1 * spike_thresh:
        return orders
    spike_bullish = spike_bar["close"] > spike_bar["open"]
    spike_bearish = spike_bar["close"] < spike_bar["open"]
    if not spike_bullish and not spike_bearish:
        return orders
    direction = "LONG" if spike_bullish else "SHORT"
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    if direction == "LONG" and confirm_bar["close"] <= confirm_bar["open"]:
        return orders
    if direction == "SHORT" and confirm_bar["close"] >= confirm_bar["open"]:
        return orders
    if direction == "LONG":
        entry = round(current_bar["open"], 3)
        stop  = round(spike_bar["low"] - atr_h1 * 0.5, 3)
    else:
        entry = round(current_bar["open"], 3)
        stop  = round(spike_bar["high"] + atr_h1 * 0.5, 3)
    actual_stop_dist = abs(entry - stop)
    if actual_stop_dist <= 0:
        return orders
    lots = _calculate_lot_size(state.balance, actual_stop_dist, state.size_multiplier)
    expiry_utc = current_time + timedelta(hours=2)
    orders.append(SimOrder(
        strategy="S8_NEWS_SPIKE", direction=direction,
        order_type="MARKET", price=entry, sl=stop,
        tp=round(entry + actual_stop_dist * 1.5, 3) if direction == "LONG" else round(entry - actual_stop_dist * 1.5, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# GAP-8: R3 — CALENDAR MOMENTUM (INDEPENDENT LANE)
# -----------------------------------------------------------------------------

def _evaluate_r3(
    bar_buffer: BarBuffer, state: "SimulatedState",
    current_time: datetime, atr_h1: float,
    event_feed: Optional["HistoricalEventFeed"],
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-8: R3 Calendar Momentum.
    Independent lane — fires after high-impact event with strong M15 candle.
    Uses r3_fired_today flag. Stop: 0.75xATR. TP: 1.5R. Time-exit: 30 min.
    """
    orders: list[SimOrder] = []
    if state.r3_fired_today:
        return orders
    if state.current_regime in ("NO_TRADE", "RANGING_CLEAR"):
        return orders
    if atr_h1 <= 0:
        return orders
    if event_feed is None:
        return orders
    just_had_event = event_feed.had_recent_high_impact_event(current_time, lookback_minutes=30)
    if not just_had_event:
        return orders
    m15_df = bar_buffer.get_series("M15", count=4)
    if len(m15_df) < 2:
        return orders
    last_bar   = m15_df.iloc[-2]
    bar_range  = abs(last_bar["high"] - last_bar["low"])
    bar_body   = abs(last_bar["close"] - last_bar["open"])
    if bar_range < atr_h1 * 1.5:
        return orders
    if bar_body < bar_range * 0.6:
        return orders
    direction = "LONG" if last_bar["close"] > last_bar["open"] else "SHORT"
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    stop_dist = atr_h1 * 0.75
    if direction == "LONG":
        entry = round(last_bar["close"], 3)
        stop  = round(entry - stop_dist, 3)
    else:
        entry = round(last_bar["close"], 3)
        stop  = round(entry + stop_dist, 3)
    actual_stop_dist = abs(entry - stop)
    if actual_stop_dist <= 0:
        return orders
    lots = _calculate_lot_size(state.balance, actual_stop_dist, state.size_multiplier)
    expiry_utc = current_time + timedelta(minutes=30)
    orders.append(SimOrder(
        strategy="R3_CAL_MOMENTUM", direction=direction,
        order_type="MARKET", price=entry, sl=stop,
        tp=round(entry + actual_stop_dist * 1.5, 3) if direction == "LONG" else round(entry - actual_stop_dist * 1.5, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# -----------------------------------------------------------------------------
# LOT SIZING
# -----------------------------------------------------------------------------

def _calculate_lot_size(
    balance: float,
    stop_dist: float,
    size_multiplier: float,
) -> float:
    """
    Risk-based lot sizing mirroring live execution_engine._calculate_lot_size().
    risk_pct x balance / (stop_dist x contract_size), capped at V1_LOT_HARD_CAP.
    """
    if stop_dist <= 0 or balance <= 0:
        return 0.01
    base_risk    = getattr(config, "BASE_RISK_PCT", 0.01)
    contract_sz  = getattr(config, "CONTRACT_SIZE", 100.0)
    hard_cap     = getattr(config, "V1_LOT_HARD_CAP", 1.0)
    lots = (balance * base_risk * size_multiplier) / (stop_dist * contract_sz)
    lots = max(0.01, min(round(lots, 2), hard_cap))
    return lots
