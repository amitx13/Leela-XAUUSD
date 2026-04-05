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


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

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
    Compute ATR percentile using EWMA weighting (λ=0.94).
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


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_asian_range(bar_buffer: BarBuffer, current_time: datetime) -> Optional[dict]:
    """Compute Asian session range (00:00–07:00 UTC) from the latest 84 M5 bars."""
    m5_df = bar_buffer.get_series("M5", count=84)
    if m5_df.empty or len(m5_df) < 12:
        return None
    rh = float(m5_df["high"].max())
    rl = float(m5_df["low"].min())
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


# ─────────────────────────────────────────────────────────────────────────────
# GAP-11: MACRO BIAS GATE HELPER
# In live system, calculate_macro_bias() runs at 09:00 IST daily and sets
# state["macro_bias"] to "LONG_ONLY", "SHORT_ONLY", or "BOTH_PERMITTED".
# The backtest cannot call the live TLT feed, so we approximate:
#   - If EMA20_H1 is rising (close > EMA, ADX > 20, DI+ > DI-) → LONG_ONLY
#   - If EMA20_H1 is falling (close < EMA, ADX > 20, DI- > DI+) → SHORT_ONLY
#   - Otherwise → BOTH_PERMITTED
# This is a structural proxy that captures the same market regime the TLT
# filter targets: don't trade against the prevailing trend direction.
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# S1 — LONDON RANGE BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s1(
    bar_buffer: BarBuffer, state: SimulatedState,
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


# ─────────────────────────────────────────────────────────────────────────────
# GAP-1: S1b — FAILED BREAKOUT REVERSAL
# Mirrors evaluate_s1b_signal() from engines/signal_engine.py.
# Fires when: price returned inside range after a failed S1 breakout,
# then breaks the OPPOSITE boundary.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s1b(
    bar_buffer: BarBuffer, state: SimulatedState,
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


def _check_s1b_trigger(bar_buffer: BarBuffer, state: SimulatedState) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# GAP-2: S1d — M5 EMA20 PULLBACK RE-ENTRY
# Fires when an open S1 family position is running and price pulls back to
# EMA20 on M5 with a full body close in the trend direction.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s1d(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_m15: float,
    ema20_m15: Optional[float],
) -> list[SimOrder]:
    """
    GAP-2: S1d M5 EMA20 pullback re-entry.
    Requires: open S1 family position, EMA20 on M15, M5 body close in direction.
    Max re-entries: SUPER=8, NORMAL=5. Size = 0.5× normal lots.
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
    # Respect M5 re-entry daily count
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
    # B4 Fix: 5-min expiry on limit order
    expiry_utc = current_time + timedelta(minutes=getattr(config, "M5_LIMIT_EXPIRY_MIN", 5))
    orders.append(SimOrder(
        strategy="S1D_PYRAMID", direction=direction,
        order_type="BUY_LIMIT" if direction == "LONG" else "SELL_LIMIT",
        price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# GAP-3: S1e — PYRAMID INTO CONFIRMED WINNERS
# Fires after partial exit done AND stop at BE — one add per S1 only.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s1e(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, ema20_m15: Optional[float],
) -> list[SimOrder]:
    """
    GAP-3: S1e pyramid add after partial exit + BE activation.
    Hard limit: one pyramid per S1 trade. Size = 0.5× original lots.
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
    state.s1e_pyramid_done = True
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# GAP-4: S1f — POST-TIME-KILL NY RE-ENTRY
# After London 16:30 TK, re-enters in NY session on M5 EMA20 body close.
# Uses own counter (s1f_attempts_today), max 1 per day.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s1f(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, ema20_m15: Optional[float],
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-4: S1f post-time-kill NY re-entry.
    Requires London TK to have fired today. SUPER/NORMAL only. Max 1/day.
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
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# S2 — MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s2(
    bar_buffer: BarBuffer, state: SimulatedState,
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


# ─────────────────────────────────────────────────────────────────────────────
# S3 — STOP HUNT REVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s3(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_h1: float,
) -> list[SimOrder]:
    """S3: Stop Hunt Reversal — sweep + reclaim pattern."""
    orders: list[SimOrder] = []
    if state.current_regime in ("NO_TRADE", "UNSTABLE"):
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


# ─────────────────────────────────────────────────────────────────────────────
# GAP-5: S4 — LONDON EMA20 PULLBACK
# Mirrors live evaluate_s4(): during London, in NORMAL/SUPER trend,
# price pulls back to EMA20(H1) from the trend direction then bounces.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s4(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_h1: float,
    ema20_h1: Optional[float], di_plus: Optional[float],
    di_minus: Optional[float], macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-5: S4 London EMA20 Pullback.
    Entry: M15 close within 0.5×ATR of EMA20_H1 in trend direction.
    Stop: 1.5×ATR beyond EMA20. TP: 2R.
    """
    orders: list[SimOrder] = []
    if state.s4_fired_today:
        return orders
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING"):
        return orders
    if state.trend_family_occupied:
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


# ─────────────────────────────────────────────────────────────────────────────
# GAP-6: S5 — NY SESSION COMPRESSION BREAKOUT
# Mirrors live evaluate_s5(): during NY session, when London range compressed
# into tight consolidation, breakout of that tight NY range.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s5(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_h1: float,
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-6: S5 NY Session Compression Breakout.
    Requires NY session (13:00–17:00 UTC), NORMAL/SUPER regime.
    Looks for tight 2-hour London consolidation and breakout.
    """
    orders: list[SimOrder] = []
    if state.s5_fired_today:
        return orders
    if state.current_session not in ("NY", "LONDON_NY_OVERLAP"):
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING"):
        return orders
    if state.trend_family_occupied:
        return orders
    utc_hour = current_time.hour
    if not (13 <= utc_hour <= 17):
        return orders
    if atr_h1 <= 0:
        return orders
    # Get last 8 H1 bars (London session range)
    m15_df = bar_buffer.get_series("M15", count=48)  # 12h = 48 M15 bars
    if len(m15_df) < 24:
        return orders
    # Define NY compression: last 8 M15 bars (2 hours)
    ny_range_df = m15_df.iloc[-8:]
    ny_high = float(ny_range_df["high"].max())
    ny_low  = float(ny_range_df["low"].min())
    ny_range = ny_high - ny_low
    # Must be compressed: range < 0.5× ATR (tight consolidation)
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
    expiry_utc = current_time.replace(hour=17, minute=0, second=0)
    orders.append(SimOrder(
        strategy="S5_NY_COMPRESS", direction=direction,
        order_type="MARKET", price=entry, sl=stop,
        tp=round(entry + stop_dist * 2.0, 3) if direction == "LONG" else round(entry - stop_dist * 2.0, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# S6 — ASIAN RANGE BREAKOUT OCO
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s6(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_m15: float,
) -> list[SimOrder]:
    """S6: Asian Range Breakout OCO pair."""
    orders: list[SimOrder] = []
    if state.s6_placed_today:
        return orders
    if state.current_regime in ("NO_TRADE",):
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
    orders.append(SimOrder(
        strategy="S6_ASIAN_BRK", direction="LONG", order_type="BUY_STOP",
        price=buy_entry, sl=buy_stop, tp=round(buy_entry + stop_dist * 2.0, 3),
        lots=buy_lots, expiry=expiry, placed_time=current_time,
        tag="s6_buy_leg", linked_tag="s6_sell_leg",
    ))
    orders.append(SimOrder(
        strategy="S6_ASIAN_BRK", direction="SHORT", order_type="SELL_STOP",
        price=sell_entry, sl=sell_stop, tp=round(sell_entry - stop_dist * 2.0, 3),
        lots=sell_lots, expiry=expiry, placed_time=current_time,
        tag="s6_sell_leg", linked_tag="s6_buy_leg",
    ))
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# S7 — DAILY STRUCTURE BREAKOUT OCO
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s7(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, daily_atr: Optional[float],
) -> list[SimOrder]:
    """S7: Daily Structure Breakout OCO pair."""
    orders: list[SimOrder] = []
    if state.s7_placed_today:
        return orders
    if state.current_regime == "NO_TRADE":
        return orders
    if current_time.hour != 0 or current_time.minute > 15:
        return orders
    prev_day = _get_prev_day_hl(bar_buffer)
    if prev_day is None:
        return None
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
    orders.append(SimOrder(
        strategy="S7_DAILY_STRUCT", direction="LONG", order_type="BUY_STOP",
        price=buy_entry, sl=round(buy_entry - stop_dist, 3),
        lots=_calculate_lot_size(state.balance, stop_dist, state.size_multiplier * size_mult),
        expiry=expiry, placed_time=current_time, tag="s7_buy_leg", linked_tag="s7_sell_leg",
    ))
    orders.append(SimOrder(
        strategy="S7_DAILY_STRUCT", direction="SHORT", order_type="SELL_STOP",
        price=sell_entry, sl=round(sell_entry + stop_dist, 3),
        lots=_calculate_lot_size(state.balance, stop_dist, state.size_multiplier * size_mult),
        expiry=expiry, placed_time=current_time, tag="s7_sell_leg", linked_tag="s7_buy_leg",
    ))
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# GAP-7: S8 — ATR SPIKE CONTINUATION (INDEPENDENT LANE)
# Mirrors live evaluate_s8(): fires when a large ATR spike candle forms and
# the next M15 candle closes in the same direction. Independent lane —
# does NOT block S1 family trades.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s8(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_h1: float,
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-7: S8 ATR Spike Continuation.
    Independent lane: fires alongside trend family if conditions met.
    Trigger: M15 bar range >= 2×ATR(H1). Confirmation: next bar closes in same direction.
    Stop: 0.5×ATR beyond spike low/high. TP: 1.5R. Max 1 per day.
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
    # Look at second-to-last closed bar as the "spike" bar
    spike_bar  = m15_df.iloc[-3]
    confirm_bar = m15_df.iloc[-2]
    spike_range = abs(spike_bar["high"] - spike_bar["low"])
    spike_thresh = getattr(config, "S8_SPIKE_ATR_MULTIPLIER", 2.0)
    if spike_range < atr_h1 * spike_thresh:
        return orders
    # Determine spike direction by candle body
    spike_bullish  = spike_bar["close"] > spike_bar["open"]
    spike_bearish  = spike_bar["close"] < spike_bar["open"]
    if not spike_bullish and not spike_bearish:
        return orders
    direction = "LONG" if spike_bullish else "SHORT"
    # GAP-11: macro bias gate
    if not _macro_permits(direction, macro_bias):
        return orders
    # Confirmation: next bar closes in same direction
    if direction == "LONG" and confirm_bar["close"] <= confirm_bar["open"]:
        return orders
    if direction == "SHORT" and confirm_bar["close"] >= confirm_bar["open"]:
        return orders
    stop_dist = atr_h1 * 0.5
    if direction == "LONG":
        entry = round(confirm_bar["close"], 3)
        stop  = round(spike_bar["low"] - stop_dist, 3)
    else:
        entry = round(confirm_bar["close"], 3)
        stop  = round(spike_bar["high"] + stop_dist, 3)
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


# ─────────────────────────────────────────────────────────────────────────────
# GAP-8: R3 — CALENDAR MOMENTUM (INDEPENDENT LANE)
# Mirrors live evaluate_r3(): after a high-impact economic event,
# rides the directional momentum for 30 minutes. Independent lane.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_r3(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_h1: float,
    event_feed: Optional["HistoricalEventFeed"],
    macro_bias: str = "BOTH_PERMITTED",
) -> list[SimOrder]:
    """
    GAP-8: R3 Calendar Momentum.
    Independent lane — fires after high-impact event with strong M15 candle.
    Uses r3_fired_today flag. Stop: 0.75×ATR. TP: 1.5R. Time-exit: 30 min.
    """
    orders: list[SimOrder] = []
    if state.r3_fired_today:
        return orders
    if state.current_regime in ("NO_TRADE", "RANGING_CLEAR"):
        return orders
    if atr_h1 <= 0:
        return orders
    # Check if we just passed a high-impact event (within last 30 min)
    if event_feed is None:
        return orders
    just_had_event = event_feed.had_recent_high_impact_event(current_time, lookback_minutes=30)
    if not just_had_event:
        return orders
    # Look for strong momentum candle on M15
    m15_df = bar_buffer.get_series("M15", count=4)
    if len(m15_df) < 2:
        return orders
    last_bar   = m15_df.iloc[-2]  # last closed bar
    bar_range  = abs(last_bar["high"] - last_bar["low"])
    bar_body   = abs(last_bar["close"] - last_bar["open"])
    # Must be a strong candle: body >= 60% of range, range >= 1.5×ATR
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
    # R3 time-exit: 30 minutes from entry
    expiry_utc = current_time + timedelta(minutes=30)
    orders.append(SimOrder(
        strategy="R3_CAL_MOMENTUM", direction=direction,
        order_type="MARKET", price=entry, sl=stop,
        tp=round(entry + actual_stop_dist * 1.5, 3) if direction == "LONG" else round(entry - actual_stop_dist * 1.5, 3),
        lots=lots, expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# LOT SIZING
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_lot_size(
    balance: float, stop_distance: float,
    size_multiplier: float, base_risk: float = None,
) -> float:
    """Risk-based lot sizing. Matches live execution path."""
    if base_risk is None:
        base_risk = getattr(config, "BASE_RISK_PHASE_1", 0.01)
    if stop_distance <= 0 or size_multiplier <= 0:
        return 0.01
    lots    = (balance * base_risk * size_multiplier) / (stop_distance * 100.0)
    lot_cap = getattr(config, "V1_LOT_HARD_CAP", 0.50)
    return max(0.01, min(round(lots, 2), lot_cap))


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Main backtesting engine. Replays M5 bars and simulates the full trading system.

    Now includes all 13 strategies (S1/S1b/S1d/S1e/S1f/S2/S3/S4/S5/S6/S7/S8/R3),
    full kill switch enforcement (KS3/KS4/KS5/KS6), TLT macro bias proxy,
    commission deduction, and weekly PnL tracking.

    Usage:
        engine = BacktestEngine(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2026, 3, 31),
        )
        results = engine.run()
        results.summary()
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        slippage_points: float = 0.7,
        strategies: Optional[list[str]] = None,
        cache_dir: str = "backtest_data",
    ):
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.cache_dir       = cache_dir

        self.strategies = strategies or [
            "S1_LONDON_BRK", "S1B_FAILED_BRK", "S1D_PYRAMID",
            "S1E_PYRAMID", "S1F_POST_TK",
            "S2_MEAN_REV", "S3_STOP_HUNT_REV",
            "S4_EMA_PULLBACK", "S5_NY_COMPRESS",
            "S6_ASIAN_BRK", "S7_DAILY_STRUCT",
            "S8_NEWS_SPIKE", "R3_CAL_MOMENTUM",
        ]

        self.data_feed   = HistoricalDataFeed(start_date, end_date, cache_dir=cache_dir)
        self.bar_buffer  = BarBuffer()
        self.exec_sim    = ExecutionSimulator(slippage_points=slippage_points)
        self.spread_feed: Optional[HistoricalSpreadFeed] = None
        self.event_feed:  Optional[HistoricalEventFeed]  = None

        self.state = SimulatedState(
            balance=initial_balance, equity=initial_balance, peak_equity=initial_balance,
        )

        self.pending_orders:  list[SimOrder]    = []
        self.open_positions:  list[SimPosition] = []
        self.closed_trades:   list[TradeRecord] = []
        self.equity_curve:    list[EquityPoint] = []

        # Indicator cache
        self._last_adx_h4:     float = 25.0
        self._last_atr_pct_h1: float = 50.0
        self._last_atr_h1_raw: float = 20.0
        self._last_atr_m15:    float = 5.0
        self._last_ema20_h1:   Optional[float] = None
        self._last_ema20_m15:  Optional[float] = None
        self._last_rsi_h1:     Optional[float] = None
        self._last_daily_atr:  Optional[float] = None
        self._last_di_plus:    Optional[float] = None
        self._last_di_minus:   Optional[float] = None

        # GAP-11: macro bias derived each H1 update
        self._current_macro_bias: str = "BOTH_PERMITTED"

        # GAP-15: weekly PnL tracking for KS5
        self._weekly_pnl:       float = 0.0
        self._current_week:     Optional[int] = None

        self._current_day:          Optional[int]      = None
        self._bars_processed:       int                = 0
        self._last_regime_update:   Optional[datetime] = None
        self._last_session_hour:    Optional[tuple[int, int]] = None

    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================

    def run(self) -> "BacktestResults":
        """Execute the backtest. Main replay loop."""
        from backtest.results import BacktestResults

        logger.info(
            f"Starting backtest: {self.start_date} to {self.end_date}, "
            f"balance=${self.initial_balance:,.2f}, strategies={self.strategies}"
        )

        m5_df = self.data_feed.load()
        self.spread_feed = HistoricalSpreadFeed(m5_df)
        self.event_feed  = HistoricalEventFeed(self.start_date, self.end_date)

        self.equity_curve.append(EquityPoint(
            timestamp=self.start_date, equity=self.initial_balance,
        ))

        total_bars    = len(m5_df)
        log_interval  = max(total_bars // 20, 1)

        for bar in self.data_feed.iter_m5_bars():
            self._bars_processed += 1
            current_time = bar["time"]

            if self._bars_processed % log_interval == 0:
                pct = self._bars_processed / total_bars * 100
                logger.info(
                    f"Progress: {pct:.0f}% ({self._bars_processed}/{total_bars}) "
                    f"| Equity: ${self.state.equity:,.2f} | Trades: {len(self.closed_trades)}"
                )

            completed = self.bar_buffer.add_m5(bar)
            self._check_daily_reset(current_time)
            self._check_weekly_reset(current_time)
            spread = self.spread_feed.get_spread_at(current_time)

            current_hour_key = (current_time.year, current_time.timetuple().tm_yday, current_time.hour)
            if current_hour_key != self._last_session_hour:
                self._update_session(current_time)
                self._last_session_hour = current_hour_key

            if completed.get("H1"): self._update_h1_indicators(current_time)
            if completed.get("H4"): self._update_h4_indicators()
            if completed.get("D1"): self._update_d1_indicators()
            if completed.get("M15"): self._update_m15_indicators()

            if completed.get("M15") or completed.get("H4"):
                self._update_regime(current_time, spread)

            self._process_pending_orders(bar, spread, current_time)
            self._manage_positions(bar, current_time)

            # GAP-12: evaluate S1/S1b on every M5 bar for intra-candle catches
            if "S1_LONDON_BRK" in self.strategies:
                s1_orders = _evaluate_s1(
                    self.bar_buffer, self.state, current_time,
                    self._last_atr_h1_raw, self._current_macro_bias,
                )
                for o in s1_orders:
                    if not any(
                        p.strategy == "S1_LONDON_BRK" and p.direction == o.direction
                        for p in self.pending_orders
                    ):
                        self.pending_orders.append(o)
                        self.state.s7_placed_today = self.state.s7_placed_today  # no-op, just guard

            if "S1B_FAILED_BRK" in self.strategies:
                _check_s1b_trigger(self.bar_buffer, self.state)
                s1b_orders = _evaluate_s1b(
                    self.bar_buffer, self.state, current_time,
                    self._last_atr_h1_raw, self._current_macro_bias,
                )
                self.pending_orders.extend(s1b_orders)

            if completed.get("M15"):
                self._update_range_data(current_time)
                self._evaluate_strategies(current_time)
                self._record_equity(current_time, bar["close"])

        # Force-close open positions at backtest end
        if self.open_positions:
            last_bar = self.bar_buffer.get_last_bar("M5")
            if last_bar:
                self._close_all_positions(last_bar["close"], last_bar["time"], "BACKTEST_END")

        logger.info(
            f"Backtest complete: {len(self.closed_trades)} trades, "
            f"final equity=${self.state.equity:,.2f}"
        )

        return BacktestResults(
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            initial_balance=self.initial_balance,
            start_date=self.start_date,
            end_date=self.end_date,
            strategies=self.strategies,
        )

    # =========================================================================
    # DAILY RESET  — FIX BUG 3
    # =========================================================================

    def _check_daily_reset(self, current_time: datetime) -> None:
        """
        Reset ALL daily-scoped flags at midnight UTC.
        Mirrors live system's on_new_day() completely.
        """
        day = current_time.timetuple().tm_yday
        if self._current_day is not None and day != self._current_day:
            self.state.s1_family_attempts_today = 0
            self.state.s1f_attempts_today       = 0
            self.state.s2_fired_today           = False
            self.state.s3_fired_today           = False
            self.state.s4_fired_today           = False
            self.state.s5_fired_today           = False
            self.state.s8_fired_today           = False
            self.state.s6_placed_today          = False
            self.state.s7_placed_today          = False
            self.state.s1d_ema_touched_today    = False
            self.state.s1d_fired_today          = False
            self.state.s1e_pyramid_done         = False
            self.state.s8_armed                 = False
            self.state.s8_arm_time              = None
            self.state.s8_confirmation_passed   = False
            self.state.r3_fired_today           = False
            self.state.range_computed           = False
            self.state.daily_pnl                = 0.0
            self.state.daily_trades             = 0
            # Do NOT reset consecutive_losses — preserved across midnight (live parity)
            logger.debug(f"Daily reset at {current_time.date()}")
        self._current_day = day

    # =========================================================================
    # GAP-15: WEEKLY RESET for KS5
    # =========================================================================

    def _check_weekly_reset(self, current_time: datetime) -> None:
        """
        GAP-15: Reset weekly PnL at start of each new ISO week (Monday midnight UTC).
        Required for KS5 (weekly loss kill switch) to work correctly.
        """
        iso_week = current_time.isocalendar()[1]
        if self._current_week is not None and iso_week != self._current_week:
            self._weekly_pnl = 0.0
            logger.debug(f"Weekly PnL reset at {current_time.date()} (week {iso_week})")
        self._current_week = iso_week

    # =========================================================================
    # SESSION UPDATE
    # =========================================================================

    def _update_session(self, current_time: datetime) -> None:
        from utils.session import get_session_for_datetime
        self.state.current_session = get_session_for_datetime(current_time)

    # =========================================================================
    # INDICATOR UPDATES
    # =========================================================================

    def _update_h1_indicators(self, current_time: datetime) -> None:
        """Recompute H1: ATR(14), EMA(20), RSI(14). Capped at 200 bars."""
        h1_df = self.bar_buffer.get_series("H1", count=200)
        if len(h1_df) < 20:
            return
        atr_mode   = getattr(config, "ATR_MAMODE", "RMA")
        atr_period = getattr(config, "ATR_PERIOD", 14)
        atr_series = ta.atr(h1_df["high"], h1_df["low"], h1_df["close"],
                            length=atr_period, mamode=atr_mode)
        if atr_series is not None and not atr_series.empty:
            last_atr = atr_series.dropna()
            if len(last_atr) > 0:
                self._last_atr_h1_raw = float(last_atr.iloc[-1])
                self.state.last_atr_h1_raw = self._last_atr_h1_raw
                self._last_atr_pct_h1 = compute_atr_percentile(atr_series, self._last_atr_h1_raw)
                self.state.last_atr_pct_h1 = self._last_atr_pct_h1
        ema_series = ta.ema(h1_df["close"], length=20)
        if ema_series is not None and not ema_series.empty:
            v = ema_series.dropna()
            if len(v) > 0: self._last_ema20_h1 = float(v.iloc[-1])
        rsi_series = ta.rsi(h1_df["close"], length=14)
        if rsi_series is not None and not rsi_series.empty:
            v = rsi_series.dropna()
            if len(v) > 0: self._last_rsi_h1 = float(v.iloc[-1])
        # GAP-11: derive macro bias proxy from updated H1 indicators
        if len(h1_df) > 0:
            current_price = float(h1_df["close"].iloc[-1])
            self._current_macro_bias = _derive_macro_bias(
                price=current_price,
                ema20_h1=self._last_ema20_h1,
                adx_h4=self._last_adx_h4,
                di_plus=self._last_di_plus,
                di_minus=self._last_di_minus,
            )

    def _update_h4_indicators(self) -> None:
        """
        Recompute H4: ADX(14) + DI+(14) + DI-(14). Capped at 100 bars.
        FIX Bug 5: reads DMP_14 and DMN_14.
        """
        h4_df = self.bar_buffer.get_series("H4", count=100)
        if len(h4_df) < 30:
            return
        adx_df = ta.adx(h4_df["high"], h4_df["low"], h4_df["close"], length=14)
        if adx_df is None or adx_df.empty:
            return
        if "ADX_14" in adx_df.columns:
            v = adx_df["ADX_14"].dropna()
            if len(v) > 0:
                self._last_adx_h4      = float(v.iloc[-1])
                self.state.last_adx_h4 = self._last_adx_h4
        if "DMP_14" in adx_df.columns:
            v = adx_df["DMP_14"].dropna()
            if len(v) > 0:
                self._last_di_plus             = float(v.iloc[-1])
                self.state.last_di_plus_h4     = self._last_di_plus
        if "DMN_14" in adx_df.columns:
            v = adx_df["DMN_14"].dropna()
            if len(v) > 0:
                self._last_di_minus            = float(v.iloc[-1])
                self.state.last_di_minus_h4    = self._last_di_minus

    def _update_d1_indicators(self) -> None:
        d1_df = self.bar_buffer.get_series("D1")
        if len(d1_df) < 16:
            return
        atr_mode   = getattr(config, "ATR_MAMODE", "RMA")
        atr_series = ta.atr(d1_df["high"], d1_df["low"], d1_df["close"],
                            length=14, mamode=atr_mode)
        if atr_series is not None and not atr_series.empty:
            v = atr_series.dropna()
            if len(v) > 0: self._last_daily_atr = float(v.iloc[-1])

    def _update_m15_indicators(self) -> None:
        """Recompute M15: ATR(14), EMA(20). Capped at 200 bars."""
        m15_df = self.bar_buffer.get_series("M15", count=200)
        if len(m15_df) < 20:
            return
        atr_mode   = getattr(config, "ATR_MAMODE", "RMA")
        atr_series = ta.atr(m15_df["high"], m15_df["low"], m15_df["close"],
                            length=14, mamode=atr_mode)
        if atr_series is not None and not atr_series.empty:
            v = atr_series.dropna()
            if len(v) > 0:
                self._last_atr_m15      = float(v.iloc[-1])
                self.state.last_atr_m15 = self._last_atr_m15
        ema_series = ta.ema(m15_df["close"], length=20)
        if ema_series is not None and not ema_series.empty:
            v = ema_series.dropna()
            if len(v) > 0: self._last_ema20_m15 = float(v.iloc[-1])

    # =========================================================================
    # REGIME
    # =========================================================================

    def _update_regime(self, current_time: datetime, spread: float) -> None:
        avg_spread   = self.spread_feed.get_avg_spread_24h() if self.spread_feed else 25.0
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0
        has_event    = self.event_feed.has_upcoming_event(current_time) if self.event_feed else False
        regime, mult = classify_regime_backtest(
            adx_h4=self._last_adx_h4, atr_pct_h1=self._last_atr_pct_h1,
            session=self.state.current_session, has_upcoming_event=has_event,
            spread_ratio=spread_ratio,
        )
        self.state.current_regime   = regime
        self.state.size_multiplier  = mult
        self._last_regime_update    = current_time

    # =========================================================================
    # RANGE DATA
    # =========================================================================

    def _update_range_data(self, current_time: datetime) -> None:
        if current_time.hour < 7 or self.state.range_computed:
            return
        range_data = _compute_asian_range(self.bar_buffer, current_time)
        if range_data:
            self.state.range_high    = range_data["range_high"]
            self.state.range_low     = range_data["range_low"]
            self.state.range_size    = range_data["range_size"]
            self.state.range_computed = True
            logger.debug(
                f"Asian range: {range_data['range_low']:.2f}-{range_data['range_high']:.2f} "
                f"(size={range_data['range_size']:.2f})"
            )

    # =========================================================================
    # ORDER PROCESSING
    # =========================================================================

    def _process_pending_orders(
        self, bar: dict, spread: float, current_time: datetime
    ) -> None:
        if not self.pending_orders:
            return
        filled, remaining = self.exec_sim.process_pending_orders(
            self.pending_orders, bar, spread, current_time
        )
        for pos in filled:
            pos.regime_at_entry = self.state.current_regime
            self.open_positions.append(pos)
            # OCO: cancel linked leg
            filled_tags: set[str] = set()
            for order in self.pending_orders:
                if order not in remaining and order.tag:
                    filled_tags.add(order.tag)
            new_remaining = [
                o for o in remaining
                if not (o.linked_tag and o.linked_tag in filled_tags)
            ]
            remaining = new_remaining
            # Update counters
            strat = pos.strategy
            if strat == "S1_LONDON_BRK":
                self.state.s1_family_attempts_today += 1
                self.state.trend_family_occupied     = True
                self.state.trend_family_strategy     = strat
                self.state.last_s1_direction         = pos.direction
                self.state.original_lot_size         = pos.lots
            elif strat == "S1B_FAILED_BRK":
                self.state.s1_family_attempts_today += 1
                self.state.trend_family_occupied     = True
                self.state.trend_family_strategy     = strat
            elif strat in ("S1D_PYRAMID", "S1E_PYRAMID"):
                self.state.s1d_pyramid_count        += 1
                self.state.trend_family_occupied     = True
                self.state.trend_family_strategy     = strat
            elif strat == "S1F_POST_TK":
                self.state.s1f_attempts_today       += 1
                self.state.trend_family_occupied     = True
                self.state.trend_family_strategy     = strat
            elif strat == "S7_DAILY_STRUCT":
                self.state.s7_placed_today = True
            elif strat == "S6_ASIAN_BRK":
                self.state.s6_placed_today = True
            elif strat == "S2_MEAN_REV":
                self.state.s2_fired_today  = True
                self.state.trend_family_occupied = True
                self.state.trend_family_strategy = strat
            elif strat == "S3_STOP_HUNT_REV":
                self.state.s3_fired_today  = True
                self.state.trend_family_occupied = True
                self.state.trend_family_strategy = strat
            elif strat == "S4_EMA_PULLBACK":
                self.state.s4_fired_today  = True
                self.state.trend_family_occupied = True
                self.state.trend_family_strategy = strat
            elif strat == "S5_NY_COMPRESS":
                self.state.s5_fired_today  = True
                self.state.trend_family_occupied = True
                self.state.trend_family_strategy = strat
            elif strat == "S8_NEWS_SPIKE":
                self.state.s8_fired_today  = True
                self.state.s8_open_ticket  = id(pos)
            elif strat == "R3_CAL_MOMENTUM":
                self.state.r3_fired_today  = True
                self.state.r3_open_ticket  = id(pos)
                self.state.r3_open_time    = current_time
        self.pending_orders = remaining

    # =========================================================================
    # POSITION MANAGEMENT  — FIX BUG 2 + BUG 6
    # =========================================================================

    def _close_position(
        self, pos: SimPosition, exit_price: float,
        current_time: datetime, reason: str,
    ) -> None:
        """
        FIX Bug 2 + Bug 4: Unified position-close helper.
        Clears lane flags after every close.
        """
        trade = self.exec_sim.close_position(
            pos, exit_price, current_time, reason,
            regime_at_exit=self.state.current_regime,
        )
        self._record_trade(trade)
        self._update_lane_flags(pos)

    def _update_lane_flags(self, pos: SimPosition) -> None:
        """Clear position-lane occupied flags when a position is closed."""
        strat = pos.strategy
        if strat in (
            "S1_LONDON_BRK", "S1B_FAILED_BRK", "S1D_PYRAMID", "S1E_PYRAMID",
            "S1F_POST_TK", "S2_MEAN_REV", "S3_STOP_HUNT_REV",
            "S4_EMA_PULLBACK", "S5_NY_COMPRESS",
            "S6_ASIAN_BRK", "S7_DAILY_STRUCT",
        ):
            self.state.trend_family_occupied  = False
            self.state.trend_family_strategy  = None
            self.state.open_trade_id          = None
        elif strat == "S8_NEWS_SPIKE":
            self.state.s8_open_ticket         = None
            self.state.s8_trade_direction     = None
        elif strat == "R3_CAL_MOMENTUM":
            self.state.r3_open_ticket         = None
            self.state.r3_direction           = None
            self.state.r3_armed               = False

    def _manage_r3_position(
        self, pos: SimPosition, price: float,
        current_time: datetime, bar: dict,
    ) -> bool:
        """R3 time-exit: close after 30 minutes."""
        if self.state.r3_open_time is None:
            return False
        elapsed = (current_time - self.state.r3_open_time).total_seconds() / 60
        if elapsed >= 30:
            self._close_position(pos, price, current_time, "TIME_KILL")
            self.state.r3_open_time = None
            return True
        return False

    def _manage_positions(self, bar: dict, current_time: datetime) -> None:
        """Manage all open positions each M5 bar."""
        still_open: list[SimPosition] = []

        for pos in self.open_positions:
            if pos.direction == "LONG":
                pos.max_favorable = max(pos.max_favorable, bar["high"] - pos.entry_price)
            else:
                pos.max_favorable = max(pos.max_favorable, pos.entry_price - bar["low"])
            pos.max_r = max(pos.max_r, pos.current_r(bar["close"]))

            if pos.strategy == "R3_CAL_MOMENTUM":
                if self._manage_r3_position(pos, bar["close"], current_time, bar):
                    continue

            closed, exit_price, reason = self.exec_sim.check_sl_tp(pos, bar)
            if closed:
                self._close_position(pos, exit_price, current_time, reason)
                continue

            partial_r = getattr(config, "PARTIAL_EXIT_R", 2.0)
            should_partial, partial_price = self.exec_sim.check_partial_exit(pos, bar, partial_r)
            if should_partial and not pos.partial_done:
                half_lots = round(pos.lots / 2, 2)
                if half_lots >= 0.01:
                    partial_pos = SimPosition(
                        strategy=pos.strategy, direction=pos.direction,
                        entry_price=pos.entry_price, entry_time=pos.entry_time,
                        lots=half_lots, stop_price_original=pos.stop_price_original,
                        current_sl=pos.current_sl, tp=pos.tp,
                        regime_at_entry=pos.regime_at_entry,
                    )
                    trade = self.exec_sim.close_position(
                        partial_pos, partial_price, current_time, "PARTIAL",
                        regime_at_exit=self.state.current_regime,
                    )
                    self._record_trade(trade)
                    pos.lots         = round(pos.lots - half_lots, 2)
                    pos.partial_done = True
                    # Update backtest state for S1e eligibility
                    if pos.strategy in ("S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK"):
                        self.state.position_partial_done = True

            be_r = getattr(config, "BE_ACTIVATION_R", 1.5)
            if not pos.be_activated and self.exec_sim.check_be_activation(pos, bar, be_r):
                pos.current_sl   = pos.entry_price
                pos.be_activated = True
                # Update backtest state for S1e eligibility
                if pos.strategy in ("S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK"):
                    self.state.position_be_activated = True
                    self.state.stop_price_current    = pos.entry_price
                logger.debug(f"BE activated: {pos.strategy} {pos.direction} @ {pos.entry_price:.2f}")

            if pos.be_activated:
                trail_mult = getattr(config, "ATR_TRAIL_MULTIPLIER", 2.5)
                new_sl = self.exec_sim.compute_atr_trail(pos, bar, self._last_atr_m15, trail_mult)
                if new_sl is not None:
                    pos.current_sl = new_sl
                    if pos.strategy in ("S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK"):
                        self.state.stop_price_current = new_sl

            still_open.append(pos)

        self.open_positions = still_open

    # =========================================================================
    # STRATEGY EVALUATION
    # =========================================================================

    def _evaluate_strategies(self, current_time: datetime) -> None:
        """
        Evaluate all enabled strategies for new signals.

        GAP-9:  KS5 weekly loss kill switch enforced.
        GAP-10: KS6 drawdown kill switch enforced.
        GAP-14: KS4 cooldown reduces S1 family attempt cap.
        """
        # ── KS3: Daily loss limit ──────────────────────────────────────────
        daily_loss_limit = getattr(config, "KS3_DAILY_LOSS_LIMIT_PCT", -0.04)
        if self.state.balance > 0:
            if self.state.daily_pnl / self.state.balance <= daily_loss_limit:
                logger.debug(f"KS3 HALT: daily_pnl={self.state.daily_pnl:.2f}")
                return

        # ── GAP-9: KS5 Weekly loss limit ─────────────────────────────────
        weekly_loss_limit = getattr(config, "KS5_WEEKLY_LOSS_LIMIT_PCT", -0.08)
        if self.state.balance > 0:
            weekly_loss_pct = self._weekly_pnl / self.state.balance
            if weekly_loss_pct <= weekly_loss_limit:
                logger.debug(f"KS5 HALT: weekly_pnl={self._weekly_pnl:.2f} ({weekly_loss_pct:.1%})")
                return

        # ── GAP-10: KS6 Account drawdown limit ───────────────────────────
        max_dd_limit = getattr(config, "KS6_MAX_DRAWDOWN_PCT", -0.15)
        if self.state.peak_equity > 0:
            current_dd = (self.state.equity - self.state.peak_equity) / self.state.peak_equity
            if current_dd <= max_dd_limit:
                logger.debug(f"KS6 HALT: drawdown={current_dd:.1%}")
                return

        new_orders: list[SimOrder] = []

        # Note: S1/S1b are evaluated on every M5 bar in run() loop (GAP-12)
        # Here we evaluate the rest (M15-cadence strategies)

        if "S1D_PYRAMID" in self.strategies:
            new_orders.extend(_evaluate_s1d(
                self.bar_buffer, self.state, current_time,
                self._last_atr_m15, self._last_ema20_m15,
            ))
        if "S1E_PYRAMID" in self.strategies:
            new_orders.extend(_evaluate_s1e(
                self.bar_buffer, self.state, current_time,
                self._last_ema20_m15,
            ))
        if "S1F_POST_TK" in self.strategies:
            new_orders.extend(_evaluate_s1f(
                self.bar_buffer, self.state, current_time,
                self._last_ema20_m15, self._current_macro_bias,
            ))
        if "S2_MEAN_REV" in self.strategies:
            new_orders.extend(_evaluate_s2(
                self.bar_buffer, self.state, current_time,
                self._last_atr_h1_raw, self._last_ema20_h1, self._last_rsi_h1,
            ))
        if "S3_STOP_HUNT_REV" in self.strategies:
            new_orders.extend(_evaluate_s3(
                self.bar_buffer, self.state, current_time, self._last_atr_h1_raw,
            ))
        if "S4_EMA_PULLBACK" in self.strategies:
            new_orders.extend(_evaluate_s4(
                self.bar_buffer, self.state, current_time,
                self._last_atr_h1_raw, self._last_ema20_h1,
                self._last_di_plus, self._last_di_minus, self._current_macro_bias,
            ))
        if "S5_NY_COMPRESS" in self.strategies:
            new_orders.extend(_evaluate_s5(
                self.bar_buffer, self.state, current_time,
                self._last_atr_h1_raw, self._current_macro_bias,
            ))
        if "S6_ASIAN_BRK" in self.strategies:
            new_orders.extend(_evaluate_s6(
                self.bar_buffer, self.state, current_time, self._last_atr_m15,
            ))
        if "S7_DAILY_STRUCT" in self.strategies:
            new_orders.extend(_evaluate_s7(
                self.bar_buffer, self.state, current_time, self._last_daily_atr,
            ))
        if "S8_NEWS_SPIKE" in self.strategies:
            new_orders.extend(_evaluate_s8(
                self.bar_buffer, self.state, current_time,
                self._last_atr_h1_raw, self._current_macro_bias,
            ))
        if "R3_CAL_MOMENTUM" in self.strategies:
            new_orders.extend(_evaluate_r3(
                self.bar_buffer, self.state, current_time,
                self._last_atr_h1_raw, self.event_feed, self._current_macro_bias,
            ))

        for order in new_orders:
            self.pending_orders.append(order)
            logger.debug(f"New order: {order.strategy} {order.direction} {order.order_type} @ {order.price:.2f}")
            if order.strategy == "S7_DAILY_STRUCT": self.state.s7_placed_today = True
            elif order.strategy == "S6_ASIAN_BRK":  self.state.s6_placed_today = True

    # =========================================================================
    # TRADE RECORDING
    # =========================================================================

    def _record_trade(self, trade: TradeRecord) -> None:
        """
        Record a closed trade and update account state.
        GAP-13: Deduct $7/lot round-trip commission per trade.
        GAP-15: Accumulate weekly PnL for KS5 gate.
        """
        # GAP-13: commission deduction — $7/lot round trip
        commission_per_lot = getattr(config, "COMMISSION_PER_LOT_RT", 7.0)
        commission = round(trade.lots * commission_per_lot, 2)
        net_pnl = trade.pnl - commission

        self.closed_trades.append(trade)
        self.state.balance      += net_pnl
        self.state.daily_pnl    += net_pnl
        self._weekly_pnl        += net_pnl   # GAP-15
        self.state.daily_trades += 1

        if trade.pnl < 0:
            self.state.consecutive_losses += 1
            # Update S1 last_s1_max_r for S1b trigger check
            if trade.strategy in ("S1_LONDON_BRK", "S1B_FAILED_BRK"):
                self.state.last_s1_max_r = min(self.state.last_s1_max_r, -abs(trade.r_multiple))
        else:
            self.state.consecutive_losses  = 0
            if trade.strategy in ("S1_LONDON_BRK", "S1B_FAILED_BRK"):
                self.state.last_s1_max_r = max(trade.r_multiple, 0.0)

        # Reset S1e / S1d / partial tracking when S1 family trade closes
        if trade.strategy in ("S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK"):
            if trade.exit_reason not in ("PARTIAL",):
                self.state.position_partial_done  = False
                self.state.position_be_activated  = False
                self.state.stop_price_current     = 0.0

        logger.debug(
            f"Trade closed: {trade.strategy} {trade.direction} "
            f"P&L=${net_pnl:+.2f} (gross=${trade.pnl:+.2f} comm=${commission:.2f}) "
            f"R={trade.r_multiple:+.2f} ({trade.exit_reason})"
        )

    def _record_equity(self, current_time: datetime, current_price: float) -> None:
        unrealized  = sum(pos.unrealized_pnl(current_price) for pos in self.open_positions)
        equity      = self.state.balance + unrealized
        self.state.equity      = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        dd_pct = (self.state.peak_equity - equity) / self.state.peak_equity if self.state.peak_equity > 0 else 0.0
        self.equity_curve.append(EquityPoint(timestamp=current_time, equity=equity, drawdown_pct=dd_pct))

    def _close_all_positions(self, price: float, time: datetime, reason: str) -> None:
        """Force-close all open positions."""
        for pos in self.open_positions:
            self._close_position(pos, price, time, reason)
        self.open_positions = []
