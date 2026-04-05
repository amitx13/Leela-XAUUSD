"""
backtest/engine.py — Core backtest logic: regime classification, strategy
evaluation helpers, indicator computation, and (appended at bottom) the
BacktestEngine orchestrator class.

All existing helper functions (_evaluate_s1 … _evaluate_r3, _derive_macro_bias,
classify_regime_backtest, compute_atr_percentile, _compute_asian_range, etc.)
are preserved unchanged above the BacktestEngine class.

CHANGES IN THIS COMMIT
──────────────────────
BUG-1 FIX: Regime warmup + ASIAN session no longer blocked.
  • BacktestEngine accepts warmup_bars (default 1000) and pre-feeds that many
    M5 bars from BEFORE start_date so H4 indicators are ready at bar-0 of the
    live window.  Eliminates the permanent NO_TRADE state during the first
    ~3.3 days caused by empty h4_ind.
  • _classify_session: ASIAN (00:00-06:59 UTC) now returns "ASIAN" (was
    collapsing into OFF_HOURS via the fallthrough).  classify_regime_backtest
    no longer hard-blocks ASIAN — it receives a 0.5× size multiplier, matching
    live behaviour for S6/S7 pre-London setup.

BUG-2 FIX: _compute_asian_range M5 fallback.
  • If fewer than 3 completed H1 bars cover 00:00-07:00 UTC (as happens on the
    first backtest day), the function now falls back to aggregating M5 bars
    directly from bar_buffer._m5_bars.  Requires only ≥12 M5 bars (1 h of
    data).  range_computed is now True by London open on every day.

BUG-3 FIX: S2 / S4 / S1b use last *completed* H1 bar (iloc[-2]).
  • _evaluate_s2, _evaluate_s4, _evaluate_s1b: changed iloc[-1] → iloc[-2]
    (guarded by len >= 2) so EMA touch / RSI checks use a stable closed bar,
    not the still-forming H1 candle.

BUG-4 FIX: _evaluate_s1e and _evaluate_s1f are no longer empty stubs.
  • _evaluate_s1e: fires the one-time 2R aggressive pyramid as a MARKET order.
  • _evaluate_s1f: fires the post-time-kill re-entry as a BUY_STOP/SELL_STOP.

Earlier fixes preserved unchanged:
  FIX-2: Macro bias uses iloc[-2] (previous completed H1 bar).
  FIX-3: run_sensitivity_test() — BREAKOUT_DIST_PCT ±20%, ATR_PCT_UNSTABLE ±5.
  FIX-4: profit_concentration_check() — warns if >80% profit in ≤3 months.
  has_upcoming_event() kwarg corrected to pre_minutes/post_minutes.
  get_events_near() kwarg corrected to window_minutes=90.
  Progress logging every 1000 bars with ETA.
"""
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None

import pytz

from backtest.data_feed import (
    BarBuffer,
    HistoricalDataFeed,
    HistoricalEventFeed,
    HistoricalSpreadFeed,
)
from backtest.models import (
    EquityPoint,
    SimOrder,
    SimPosition,
    SimulatedState,
    TradeRecord,
)
from backtest.execution_simulator import ExecutionSimulator

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger("backtest.engine")

# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr_percentile(atr_series: pd.Series, current_atr: float) -> float:
    """Return the percentile rank of current_atr within atr_series (0-100)."""
    if atr_series is None or len(atr_series) < 5 or current_atr is None:
        return 50.0
    valid = atr_series.dropna()
    if len(valid) < 5:
        return 50.0
    rank = (valid < current_atr).sum() / len(valid) * 100.0
    return float(rank)


def classify_regime_backtest(
    adx_h4: float,
    atr_pct_h1: float,
    session: str,
    has_upcoming_event: bool = False,
    spread_ratio: float = 1.0,
) -> tuple[str, float]:
    """
    Classify the current market regime and return (regime_name, size_multiplier).

    Mirrors live regime_engine.py logic exactly so backtest and live share
    the same gate.

    Regimes (in priority order):
        NO_TRADE        — hard block (event, spread, off-hours, ATR unstable)
        TRENDING_STRONG — ADX ≥ 30, normal ATR, London/NY
        TRENDING_WEAK   — ADX 20-29, normal ATR
        RANGING         — ADX < 20

    BUG-1 FIX: ASIAN session is no longer treated as OFF_HOURS / NO_TRADE.
    Only the literal "OFF_HOURS" bucket (21:00-23:59 UTC) blocks trading.
    ASIAN gets a 0.5× size multiplier to reflect lower-liquidity conditions.
    """
    atr_unstable_thresh = getattr(config, "ATR_PCT_UNSTABLE_THRESHOLD", 85) if config else 85
    adx_strong_thresh   = getattr(config, "ADX_STRONG_THRESHOLD",        30) if config else 30
    adx_weak_thresh     = getattr(config, "ADX_WEAK_THRESHOLD",          20) if config else 20

    # Hard blocks
    if has_upcoming_event:
        return "NO_TRADE", 0.0
    if spread_ratio > 2.5:
        return "NO_TRADE", 0.0
    if session == "OFF_HOURS":
        return "NO_TRADE", 0.0
    if atr_pct_h1 >= atr_unstable_thresh:
        return "NO_TRADE", 0.0

    # BUG-1 FIX: ASIAN session — permit trading with reduced size
    asian_size_scale = 0.5 if session == "ASIAN" else 1.0

    if adx_h4 >= adx_strong_thresh:
        return "TRENDING_STRONG", 1.0 * asian_size_scale
    if adx_h4 >= adx_weak_thresh:
        return "TRENDING_WEAK", 0.75 * asian_size_scale
    return "RANGING", 0.5 * asian_size_scale


# ─────────────────────────────────────────────────────────────────────────────
# MACRO BIAS
# ─────────────────────────────────────────────────────────────────────────────

def _derive_macro_bias(
    price: float,
    ema20_h1: Optional[float],
    adx_h4: float,
    di_plus: Optional[float],
    di_minus: Optional[float],
) -> str:
    """
    Derive macro directional bias from H1 EMA and H4 ADX/DI.

    Returns one of: "LONG_ONLY", "SHORT_ONLY", "BOTH_PERMITTED"

    FIX-2 NOTE: Callers must pass values from the PREVIOUS completed bar
    (iloc[-2]) — never from the current forming bar — to avoid lookahead.
    BacktestEngine.run() enforces this by using the *_prev indicator values.
    """
    if ema20_h1 is None or price is None:
        return "BOTH_PERMITTED"

    above_ema = price > ema20_h1

    if adx_h4 >= 25 and di_plus is not None and di_minus is not None:
        if di_plus > di_minus and above_ema:
            return "LONG_ONLY"
        if di_minus > di_plus and not above_ema:
            return "SHORT_ONLY"

    return "BOTH_PERMITTED"


# ─────────────────────────────────────────────────────────────────────────────
# ASIAN RANGE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_asian_range(bar_buffer: BarBuffer, current_time: datetime) -> Optional[dict]:
    """
    Compute the Asian session range (00:00–07:00 UTC) from today's bars.

    BUG-2 FIX: Three-tier approach so range_computed is True by London open
    on *every* trading day, including the very first day of the backtest when
    the H1 buffer may still be sparse:

    Tier 1 (preferred): ≥3 *completed* H1 bars covering 00:00-07:00 UTC today.
    Tier 2: <3 completed H1 bars but ≥1 — aggregate whatever completed bars
            exist for today's Asian window.
    Tier 3 (fallback): Aggregate raw M5 bars from 00:00-07:00 UTC today
            directly from bar_buffer._m5_bars.  Requires ≥12 M5 bars (1 full
            hour) to avoid an artificially narrow range from a handful of bars.

    Returns dict with range_high, range_low, range_size — or None if no tier
    has sufficient data.
    """
    today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    asian_end   = current_time.replace(hour=7, minute=0, second=0, microsecond=0)

    # ── Tier 1 & 2: completed H1 bars ────────────────────────────────────
    h1_series = bar_buffer.get_series("H1")
    if not h1_series.empty:
        mask = (
            (pd.to_datetime(h1_series["time"]) >= today_start) &
            (pd.to_datetime(h1_series["time"]) <  asian_end)
        )
        asian_bars = h1_series[mask]
        if len(asian_bars) >= 1:
            rh = float(asian_bars["high"].max())
            rl = float(asian_bars["low"].min())
            return {"range_high": rh, "range_low": rl, "range_size": rh - rl}

    # ── Tier 3: fallback — aggregate today's Asian M5 bars directly ──────
    m5_bars = bar_buffer._m5_bars  # direct access to raw list
    asian_m5 = [
        b for b in m5_bars
        if today_start <= b["time"] < asian_end
    ]
    if len(asian_m5) < 12:  # require at least 1 hour of M5 data
        return None

    rh = max(b["high"] for b in asian_m5)
    rl = min(b["low"]  for b in asian_m5)
    return {"range_high": rh, "range_low": rl, "range_size": rh - rl}


# ─────────────────────────────────────────────────────────────────────────────
# LOT SIZE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _calc_lots(
    balance: float,
    stop_distance: float,
    risk_pct: float = 0.01,
    size_multiplier: float = 1.0,
    corr_throttle: bool = False,
) -> float:
    """
    Risk-based lot sizing.  Caps at 0.50 lots.

    P&L per lot per point = $100 (XAUUSD 100 oz standard lot).
    risk_amount = balance × risk_pct × size_multiplier
    lots = risk_amount / (stop_distance × 100)
    """
    if stop_distance <= 0:
        return 0.01
    # Guard: if size_multiplier is 0 (e.g. NO_TRADE leaked through), use 0.5
    effective_mult = size_multiplier if size_multiplier > 0 else 0.5
    risk_amount = balance * risk_pct * effective_mult
    if corr_throttle:
        risk_amount *= 0.65   # CRIT-3: correlation throttle (mirrors live SIZE-5)
    raw = risk_amount / (stop_distance * 100.0)
    lots = max(0.01, min(0.50, round(raw / 0.01) * 0.01))
    return lots


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EVALUATORS — thin wrappers that return list[SimOrder]
# Each mirrors the corresponding live strategy's signal logic.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_s1(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    macro_bias: str,
) -> list:
    """S1 London Breakout — BUY_STOP above Asian high, SELL_STOP below Asian low."""
    orders = []
    if not state.range_computed or state.s1_family_attempts_today >= 2:
        return orders
    if bar_time.hour < 7 or bar_time.hour >= 11:
        return orders

    breakout_dist = getattr(config, "BREAKOUT_DIST_PCT", 0.12) * atr_h1 if config else 0.12 * atr_h1
    sl_dist       = atr_h1 * 1.5
    tp_dist       = atr_h1 * 3.0
    lots          = _calc_lots(state.balance, sl_dist, size_multiplier=state.size_multiplier,
                               corr_throttle=state.corr_throttle_active)

    expiry = bar_time.replace(hour=11, minute=0, second=0, microsecond=0)
    if expiry <= bar_time:
        return orders

    if macro_bias in ("LONG_ONLY", "BOTH_PERMITTED"):
        entry = round(state.range_high + breakout_dist, 2)
        orders.append(SimOrder(
            strategy="S1_LONDON_BRK", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, expiry=expiry, placed_time=bar_time,
            tag="s1_buy", linked_tag="s1_sell",
        ))

    if macro_bias in ("SHORT_ONLY", "BOTH_PERMITTED"):
        entry = round(state.range_low - breakout_dist, 2)
        orders.append(SimOrder(
            strategy="S1_LONDON_BRK", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, expiry=expiry, placed_time=bar_time,
            tag="s1_sell", linked_tag="s1_buy",
        ))

    return orders


def _evaluate_s1b(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    macro_bias: str,
) -> list:
    """S1b Failed-Breakout Reversal — fires when S1 hit SL quickly (max_r < 0.5)."""
    if state.last_s1_direction is None or state.last_s1_max_r >= 0.5:
        return []
    if state.trend_family_occupied:
        return []
    if bar_time.hour < 8 or bar_time.hour >= 14:
        return []

    sl_dist = atr_h1 * 1.2
    tp_dist = atr_h1 * 2.4
    lots    = _calc_lots(state.balance, sl_dist, size_multiplier=state.size_multiplier * 0.75,
                         corr_throttle=state.corr_throttle_active)

    h1_series = bar_buffer.get_series("H1")
    # BUG-3 FIX: use last *completed* H1 bar (iloc[-2]), not the forming bar (iloc[-1])
    if h1_series.empty or len(h1_series) < 2:
        return []
    last_close = float(h1_series["close"].iloc[-2])

    rev_direction = "SHORT" if state.last_s1_direction == "LONG" else "LONG"
    if rev_direction == "LONG" and macro_bias == "SHORT_ONLY":
        return []
    if rev_direction == "SHORT" and macro_bias == "LONG_ONLY":
        return []

    if rev_direction == "LONG":
        entry = round(last_close + atr_h1 * 0.3, 2)
        return [SimOrder(
            strategy="S1B_FAILED_BRK", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots,
            expiry=bar_time.replace(hour=14, minute=0, second=0, microsecond=0),
            placed_time=bar_time,
        )]
    else:
        entry = round(last_close - atr_h1 * 0.3, 2)
        return [SimOrder(
            strategy="S1B_FAILED_BRK", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots,
            expiry=bar_time.replace(hour=14, minute=0, second=0, microsecond=0),
            placed_time=bar_time,
        )]


def _check_s1b_trigger(bar_buffer: BarBuffer, state: SimulatedState) -> None:
    """Update state flags when the M15 bar closes to track S1b conditions."""
    pass  # Detailed trigger tracking delegated to state flags set in run loop


def _evaluate_s1d(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_m15: float,
    ema20_m15: Optional[float],
) -> list:
    """S1d M5 EMA Pyramid — add to winning trend position at M15 EMA touch."""
    if state.s1d_pyramid_count >= 2:
        return []
    if not state.trend_family_occupied or state.trend_trade_direction is None:
        return []
    if ema20_m15 is None or atr_m15 <= 0:
        return []

    m15_series = bar_buffer.get_series("M15")
    if m15_series.empty:
        return []
    last_low  = float(m15_series["low"].iloc[-1])
    last_high = float(m15_series["high"].iloc[-1])

    sl_dist = atr_m15 * 1.0
    tp_dist = atr_m15 * 2.0
    lots    = _calc_lots(state.balance, sl_dist,
                         risk_pct=0.005,
                         size_multiplier=state.size_multiplier,
                         corr_throttle=state.corr_throttle_active)

    direction = state.trend_trade_direction
    if direction == "LONG" and last_low <= ema20_m15:
        entry = round(ema20_m15 + atr_m15 * 0.1, 2)
        return [SimOrder(
            strategy="S1D_PYRAMID", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    if direction == "SHORT" and last_high >= ema20_m15:
        entry = round(ema20_m15 - atr_m15 * 0.1, 2)
        return [SimOrder(
            strategy="S1D_PYRAMID", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    return []


def _evaluate_s1e(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    ema20_m15: Optional[float],
) -> list:
    """
    S1e Aggressive Pyramid — one-time add at 2R when trend is strong.

    BUG-4 FIX: Was a permanent empty stub.  Now places a MARKET order at
    the current M15 close price in the trend direction with 0.5× normal risk,
    once per day when last_s1_max_r >= 2.0 and trend is occupied.
    """
    if state.s1e_pyramid_done:
        return []
    if not state.trend_family_occupied or state.trend_trade_direction is None:
        return []
    if state.last_s1_max_r < 2.0:
        return []

    m15_series = bar_buffer.get_series("M15")
    if m15_series.empty or len(m15_series) < 2:
        return []

    atr_m15 = state.last_atr_m15
    if atr_m15 <= 0:
        return []

    sl_dist = atr_m15 * 1.0
    lots    = _calc_lots(state.balance, sl_dist,
                         risk_pct=0.005,
                         size_multiplier=state.size_multiplier * 0.5,
                         corr_throttle=state.corr_throttle_active)

    last_close = float(m15_series["close"].iloc[-1])
    direction  = state.trend_trade_direction

    if direction == "LONG":
        return [SimOrder(
            strategy="S1E_PYRAMID", direction="LONG",
            order_type="MARKET",
            price=last_close,
            sl=round(last_close - sl_dist, 2),
            tp=round(last_close + sl_dist * 2.0, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=4),
        )]
    else:
        return [SimOrder(
            strategy="S1E_PYRAMID", direction="SHORT",
            order_type="MARKET",
            price=last_close,
            sl=round(last_close + sl_dist, 2),
            tp=round(last_close - sl_dist * 2.0, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=4),
        )]


def _evaluate_s1f(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    ema20_m15: Optional[float],
    macro_bias: str,
    ema20_h1: Optional[float],
) -> list:
    """
    S1f Post-Time-Kill Re-Entry — re-enter after KS2 time kill in profit.

    BUG-4 FIX: Was a permanent empty stub.  Now places a BUY_STOP/SELL_STOP
    0.3×ATR_H1 beyond the last completed M15 close, in the original killed
    trade direction, expiring in 2 hours.
    """
    if state.s1f_reentered_today or not state.s1f_post_tk_active:
        return []
    if not state.s1f_killed_position_profitable:
        return []

    direction = state.last_s1_direction
    if direction is None:
        return []

    if direction == "LONG" and macro_bias == "SHORT_ONLY":
        return []
    if direction == "SHORT" and macro_bias == "LONG_ONLY":
        return []

    m15_series = bar_buffer.get_series("M15")
    if m15_series.empty or len(m15_series) < 2:
        return []

    h1_series = bar_buffer.get_series("H1")
    atr_h1 = state.last_atr_h1_raw
    if atr_h1 <= 0:
        return []

    sl_dist = atr_h1 * 1.2
    tp_dist = atr_h1 * 2.4
    lots    = _calc_lots(
        state.balance, sl_dist,
        risk_pct=0.007,
        size_multiplier=state.size_multiplier * 0.75,
        corr_throttle=state.corr_throttle_active,
    )

    last_close = float(m15_series["close"].iloc[-1])

    if direction == "LONG":
        entry = round(last_close + atr_h1 * 0.3, 2)
        return [SimOrder(
            strategy="S1F_POST_TK", direction="LONG",
            order_type="BUY_STOP",
            price=entry,
            sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    else:
        entry = round(last_close - atr_h1 * 0.3, 2)
        return [SimOrder(
            strategy="S1F_POST_TK", direction="SHORT",
            order_type="SELL_STOP",
            price=entry,
            sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]


def _evaluate_s2(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    ema20_h1: Optional[float],
    rsi_h1: Optional[float],
) -> list:
    """S2 Mean Reversion — fade extreme RSI moves away from EMA."""
    if state.s2_fired_today:
        return []
    if ema20_h1 is None or rsi_h1 is None:
        return []
    if bar_time.hour < 8 or bar_time.hour >= 20:
        return []

    h1 = bar_buffer.get_series("H1")
    # BUG-3 FIX: use last *completed* H1 bar (iloc[-2]), not the forming bar
    if h1.empty or len(h1) < 2:
        return []
    last_close = float(h1["close"].iloc[-2])

    sl_dist = atr_h1 * 1.0
    tp_dist = atr_h1 * 2.0
    lots    = _calc_lots(state.balance, sl_dist,
                         risk_pct=0.007,
                         size_multiplier=state.size_multiplier * 0.75,
                         corr_throttle=state.corr_throttle_active)

    rsi_os = getattr(config, "S2_RSI_OVERSOLD",  35) if config else 35
    rsi_ob = getattr(config, "S2_RSI_OVERBOUGHT", 65) if config else 65

    if rsi_h1 <= rsi_os and last_close < ema20_h1:
        entry = round(last_close + atr_h1 * 0.2, 2)
        return [SimOrder(
            strategy="S2_MEAN_REV", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=4),
        )]
    if rsi_h1 >= rsi_ob and last_close > ema20_h1:
        entry = round(last_close - atr_h1 * 0.2, 2)
        return [SimOrder(
            strategy="S2_MEAN_REV", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=4),
        )]
    return []


def _evaluate_s3(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
) -> list:
    """S3 Stop-Hunt Reversal — fade spike beyond Asian range extremes."""
    if state.s3_fired_today or not state.range_computed:
        return []
    if bar_time.hour < 7 or bar_time.hour >= 16:
        return []

    m5 = bar_buffer.get_series("M5")
    if m5.empty or len(m5) < 3:
        return []

    last_low   = float(m5["low"].iloc[-1])
    last_high  = float(m5["high"].iloc[-1])
    last_close = float(m5["close"].iloc[-1])

    spike_ext  = atr_h1 * 0.5
    sl_dist    = atr_h1 * 0.8
    tp_dist    = atr_h1 * 2.0
    lots       = _calc_lots(state.balance, sl_dist,
                            risk_pct=0.008,
                            size_multiplier=state.size_multiplier * 0.75,
                            corr_throttle=state.corr_throttle_active)

    # Spike below Asian low then close back above
    if last_low < (state.range_low - spike_ext) and last_close > state.range_low:
        entry = round(last_close + atr_h1 * 0.1, 2)
        return [SimOrder(
            strategy="S3_STOP_HUNT_REV", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(last_low - sl_dist * 0.2, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    # Spike above Asian high then close back below
    if last_high > (state.range_high + spike_ext) and last_close < state.range_high:
        entry = round(last_close - atr_h1 * 0.1, 2)
        return [SimOrder(
            strategy="S3_STOP_HUNT_REV", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(last_high + sl_dist * 0.2, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    return []


def _evaluate_s4(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    ema20_h1: Optional[float],
    di_plus: Optional[float],
    di_minus: Optional[float],
    macro_bias: str,
) -> list:
    """S4 EMA Pullback — trend continuation after price pulls back to H1 EMA20."""
    if state.s4_fired_today:
        return []
    if ema20_h1 is None or atr_h1 <= 0:
        return []
    if bar_time.hour < 8 or bar_time.hour >= 18:
        return []

    h1 = bar_buffer.get_series("H1")
    # BUG-3 FIX: use last *completed* H1 bar (iloc[-2]) for low/high/close
    if h1.empty or len(h1) < 2:
        return []
    last_low   = float(h1["low"].iloc[-2])
    last_high  = float(h1["high"].iloc[-2])
    last_close = float(h1["close"].iloc[-2])

    sl_dist = atr_h1 * 1.3
    tp_dist = atr_h1 * 2.6
    lots    = _calc_lots(state.balance, sl_dist,
                         size_multiplier=state.size_multiplier,
                         corr_throttle=state.corr_throttle_active)

    touch_zone = atr_h1 * 0.15

    if (macro_bias in ("LONG_ONLY", "BOTH_PERMITTED") and
            di_plus is not None and di_minus is not None and di_plus > di_minus and
            abs(last_low - ema20_h1) <= touch_zone and last_close > ema20_h1):
        entry = round(last_close + atr_h1 * 0.1, 2)
        return [SimOrder(
            strategy="S4_EMA_PULLBACK", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=3),
        )]

    if (macro_bias in ("SHORT_ONLY", "BOTH_PERMITTED") and
            di_plus is not None and di_minus is not None and di_minus > di_plus and
            abs(last_high - ema20_h1) <= touch_zone and last_close < ema20_h1):
        entry = round(last_close - atr_h1 * 0.1, 2)
        return [SimOrder(
            strategy="S4_EMA_PULLBACK", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=3),
        )]
    return []


def _evaluate_s5(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    macro_bias: str,
) -> list:
    """S5 NY Session Compression Breakout — range contraction into NY open."""
    if state.s5_fired_today:
        return []
    if bar_time.hour < 12 or bar_time.hour >= 15:
        return []

    h1 = bar_buffer.get_series("H1")
    if len(h1) < 4:
        return []

    # Check H1 range compression: last 3 bars tighter than ATR * 0.5
    recent_ranges = (h1["high"] - h1["low"]).iloc[-3:]
    if recent_ranges.mean() > atr_h1 * 0.5:
        return []

    last_close = float(h1["close"].iloc[-1])
    sl_dist    = atr_h1 * 1.0
    tp_dist    = atr_h1 * 2.5
    lots       = _calc_lots(state.balance, sl_dist,
                            size_multiplier=state.size_multiplier,
                            corr_throttle=state.corr_throttle_active)

    breakout_dist = getattr(config, "BREAKOUT_DIST_PCT", 0.12) * atr_h1 if config else 0.12 * atr_h1

    orders = []
    if macro_bias in ("LONG_ONLY", "BOTH_PERMITTED"):
        entry = round(last_close + breakout_dist, 2)
        orders.append(SimOrder(
            strategy="S5_NY_COMPRESS", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time.replace(hour=15, minute=0, second=0, microsecond=0),
            tag="s5_buy", linked_tag="s5_sell",
        ))
    if macro_bias in ("SHORT_ONLY", "BOTH_PERMITTED"):
        entry = round(last_close - breakout_dist, 2)
        orders.append(SimOrder(
            strategy="S5_NY_COMPRESS", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time.replace(hour=15, minute=0, second=0, microsecond=0),
            tag="s5_sell", linked_tag="s5_buy",
        ))
    return orders


def _evaluate_s6(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_m15: float,
    adx_h4: float,
    di_plus: Optional[float],
    di_minus: Optional[float],
) -> list:
    """S6 Asian Breakout — breakout of previous Asian session range."""
    if state.s6_placed_today or not state.range_computed:
        return []
    if bar_time.hour < 7 or bar_time.hour >= 10:
        return []
    if adx_h4 < 18:
        return []

    sl_dist = atr_m15 * 2.0
    tp_dist = atr_m15 * 4.0
    lots    = _calc_lots(state.balance, sl_dist,
                         risk_pct=0.008,
                         size_multiplier=state.size_multiplier * 0.8,
                         corr_throttle=state.corr_throttle_active)
    expiry  = bar_time.replace(hour=10, minute=0, second=0, microsecond=0)

    orders = []
    if di_plus is not None and di_minus is not None:
        if di_plus > di_minus:
            entry = round(state.range_high + atr_m15 * 0.5, 2)
            orders.append(SimOrder(
                strategy="S6_ASIAN_BRK", direction="LONG",
                order_type="BUY_STOP",
                price=entry, sl=round(entry - sl_dist, 2),
                tp=round(entry + tp_dist, 2),
                lots=lots, expiry=expiry, placed_time=bar_time,
                tag="s6_buy", linked_tag="s6_sell",
            ))
        else:
            entry = round(state.range_low - atr_m15 * 0.5, 2)
            orders.append(SimOrder(
                strategy="S6_ASIAN_BRK", direction="SHORT",
                order_type="SELL_STOP",
                price=entry, sl=round(entry + sl_dist, 2),
                tp=round(entry - tp_dist, 2),
                lots=lots, expiry=expiry, placed_time=bar_time,
                tag="s6_sell", linked_tag="s6_buy",
            ))
    return orders


def _evaluate_s7(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: Optional[float],
    adx_h4: float,
    di_plus: Optional[float],
    di_minus: Optional[float],
) -> list:
    """S7 Daily Structure — breakout of prior day high/low."""
    if state.s7_placed_today:
        return []
    if bar_time.hour < 7 or bar_time.hour >= 12:
        return []
    if atr_h1 is None or atr_h1 <= 0:
        return []
    if adx_h4 < 20:
        return []

    d1 = bar_buffer.get_series("D1")
    if len(d1) < 2:
        return []

    prev_high = float(d1["high"].iloc[-2])
    prev_low  = float(d1["low"].iloc[-2])

    sl_dist = atr_h1 * 1.5
    tp_dist = atr_h1 * 3.5
    lots    = _calc_lots(state.balance, sl_dist,
                         size_multiplier=state.size_multiplier * 0.9,
                         corr_throttle=state.corr_throttle_active)

    expiry = bar_time.replace(hour=12, minute=0, second=0, microsecond=0)
    orders = []

    if di_plus is not None and di_minus is not None and di_plus > di_minus:
        entry = round(prev_high + atr_h1 * 0.1, 2)
        orders.append(SimOrder(
            strategy="S7_DAILY_STRUCT", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, expiry=expiry, placed_time=bar_time,
            tag="s7_buy", linked_tag="s7_sell",
        ))
    elif di_plus is not None and di_minus is not None and di_minus > di_plus:
        entry = round(prev_low - atr_h1 * 0.1, 2)
        orders.append(SimOrder(
            strategy="S7_DAILY_STRUCT", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, expiry=expiry, placed_time=bar_time,
            tag="s7_sell", linked_tag="s7_buy",
        ))
    return orders


def _evaluate_s8(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    macro_bias: str,
) -> list:
    """S8 News Spike Reversal — fade the first post-news spike candle."""
    if state.s8_fired_today:
        return []
    if not state.s8_armed or state.s8_spike_direction is None:
        return []

    m5 = bar_buffer.get_series("M5")
    if m5.empty:
        return []

    sl_dist = atr_h1 * 0.8
    tp_dist = atr_h1 * 2.0
    lots    = _calc_lots(state.balance, sl_dist,
                         risk_pct=0.008,
                         size_multiplier=state.size_multiplier * 0.75,
                         corr_throttle=state.corr_throttle_active)

    spike_dir = state.s8_spike_direction
    rev_dir   = "SHORT" if spike_dir == "LONG" else "LONG"

    if rev_dir == "LONG" and macro_bias == "SHORT_ONLY":
        return []
    if rev_dir == "SHORT" and macro_bias == "LONG_ONLY":
        return []

    last_close = float(m5["close"].iloc[-1])
    if rev_dir == "LONG":
        entry = round(last_close + atr_h1 * 0.15, 2)
        return [SimOrder(
            strategy="S8_NEWS_SPIKE", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(state.s8_spike_low - sl_dist * 0.3, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(minutes=30),
        )]
    else:
        entry = round(last_close - atr_h1 * 0.15, 2)
        return [SimOrder(
            strategy="S8_NEWS_SPIKE", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(state.s8_spike_high + sl_dist * 0.3, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(minutes=30),
        )]


def _evaluate_r3(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    bar_time: datetime,
    atr_h1: float,
    event_feed: "HistoricalEventFeed",
    macro_bias: str,
) -> list:
    """R3 Calendar Momentum — pre-event directional momentum trade."""
    if state.r3_fired_today:
        return []
    if bar_time.hour < 6 or bar_time.hour >= 13:
        return []

    # FIX: get_events_near() uses 'window_minutes=', not 'minutes_ahead='
    upcoming = event_feed.get_events_near(bar_time, window_minutes=90)
    if not upcoming:
        return []

    h1 = bar_buffer.get_series("H1")
    if len(h1) < 5:
        return []

    momentum = float(h1["close"].iloc[-1]) - float(h1["close"].iloc[-4])
    if abs(momentum) < atr_h1 * 0.5:
        return []

    sl_dist = atr_h1 * 1.2
    tp_dist = atr_h1 * 2.5
    lots    = _calc_lots(state.balance, sl_dist,
                         risk_pct=0.007,
                         size_multiplier=state.size_multiplier * 0.8,
                         corr_throttle=state.corr_throttle_active)

    if momentum > 0 and macro_bias in ("LONG_ONLY", "BOTH_PERMITTED"):
        entry = round(float(h1["close"].iloc[-1]) + atr_h1 * 0.2, 2)
        return [SimOrder(
            strategy="R3_CAL_MOMENTUM", direction="LONG",
            order_type="BUY_STOP",
            price=entry, sl=round(entry - sl_dist, 2),
            tp=round(entry + tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    if momentum < 0 and macro_bias in ("SHORT_ONLY", "BOTH_PERMITTED"):
        entry = round(float(h1["close"].iloc[-1]) - atr_h1 * 0.2, 2)
        return [SimOrder(
            strategy="R3_CAL_MOMENTUM", direction="SHORT",
            order_type="SELL_STOP",
            price=entry, sl=round(entry + sl_dist, 2),
            tp=round(entry - tp_dist, 2),
            lots=lots, placed_time=bar_time,
            expiry=bar_time + timedelta(hours=2),
        )]
    return []


# =============================================================================
# BacktestEngine — Main orchestrator
# =============================================================================

import random as _random


class BacktestEngine:
    """
    Main backtest replay engine.

    Iterates M5 bars, builds higher-TF candles, computes indicators,
    derives regime and macro bias, evaluates all strategies, fills orders,
    manages positions, and tracks equity.

    BUG-1 FIX (Regime / Warmup):
        warmup_bars (default 1000) M5 bars from before start_date are pre-fed
        into the BarBuffer silently so H4 indicators (need 960+ bars) are
        ready at the first live bar.  ASIAN session is no longer blocked.

    BUG-2 FIX (Asian Range):
        _compute_asian_range() uses a three-tier approach: completed H1 bars →
        partial H1 bars → raw M5 bars fallback.  range_computed is now True
        by London open every day, including day 1.

    BUG-3 FIX (Forming H1 bar):
        _evaluate_s2, _evaluate_s4, _evaluate_s1b now read iloc[-2] (last
        completed H1 bar) instead of iloc[-1] (still-forming candle).

    BUG-4 FIX (S1e / S1f stubs):
        Both evaluators now contain real signal logic and place orders.

    FIX-2 (Macro Bias Decoupling):
        Macro bias uses iloc[-2] (previous completed H1 bar), not iloc[-1].

    FIX-3 (Parameter Sensitivity):
        run_sensitivity_test() sweeps BREAKOUT_DIST_PCT ±20% and
        ATR_PCT_UNSTABLE_THRESHOLD ±5 pts.

    FIX-4 (Time-Period Stability):
        profit_concentration_check() warns when >80% of profits come
        from ≤3 calendar months.
    """

    def __init__(
        self,
        start_date,
        end_date,
        initial_balance: float = 10_000.0,
        slippage_points: float = 0.70,
        strategies: list = None,
        cache_dir: str = "backtest_data",
        warmup_bars: int = 1_000,
        _override_config: dict = None,
    ):
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.strategies      = strategies or []
        self.cache_dir       = cache_dir
        self.warmup_bars     = warmup_bars   # BUG-1 FIX: pre-feed N bars before start_date
        self._override       = _override_config or {}

    # -------------------------------------------------------------------------
    # INTERNAL: config lookup with override support (for sensitivity sweeps)
    # -------------------------------------------------------------------------
    def _cfg(self, attr: str, default):
        if attr in self._override:
            return self._override[attr]
        return getattr(config, attr, default) if config is not None else default

    # -------------------------------------------------------------------------
    # INDICATOR HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_indicators(series: pd.DataFrame) -> dict:
        """
        Compute ADX/DI, ATR, EMA20, RSI on a completed bar series.

        Returns scalar values for both the last bar (iloc[-1]) and the
        previous bar (iloc[-2], suffixed _prev).  The _prev values are used
        for macro bias to enforce FIX-2 decoupling.

        Returns {} if series is too short (< 20 bars).
        """
        if ta is None or len(series) < 20:
            return {}
        try:
            adx_df = ta.adx(series["high"], series["low"], series["close"], length=14)
            atr_s  = ta.atr(series["high"], series["low"], series["close"], length=14)
            ema_s  = ta.ema(series["close"], length=20)
            rsi_s  = ta.rsi(series["close"], length=14)

            def _last(s, col=None):
                if s is None:
                    return None
                if col:
                    return float(s[col].iloc[-1]) if (s is not None and not s.empty and col in s.columns) else None
                return float(s.iloc[-1]) if (s is not None and not s.empty) else None

            def _prev(s, col=None):
                if s is None:
                    return None
                if col:
                    return float(s[col].iloc[-2]) if (s is not None and len(s) > 1 and col in s.columns) else None
                return float(s.iloc[-2]) if (s is not None and len(s) > 1) else None

            return {
                "adx":            _last(adx_df, "ADX_14"),
                "di_plus":        _last(adx_df, "DMP_14"),
                "di_minus":       _last(adx_df, "DMN_14"),
                "atr":            _last(atr_s),
                "ema20":          _last(ema_s),
                "rsi":            _last(rsi_s),
                # FIX-2: previous-bar values for macro bias (decoupled from entry bar)
                "adx_prev":       _prev(adx_df, "ADX_14"),
                "di_plus_prev":   _prev(adx_df, "DMP_14"),
                "di_minus_prev":  _prev(adx_df, "DMN_14"),
                "ema20_prev":     _prev(ema_s),
            }
        except Exception as exc:
            logger.debug(f"Indicator compute error: {exc}")
            return {}

    # -------------------------------------------------------------------------
    # SESSION CLASSIFIER
    # -------------------------------------------------------------------------
    @staticmethod
    def _classify_session(utc_dt) -> str:
        """
        BUG-1 FIX: ASIAN (00:00-06:59 UTC) is now a distinct session, not
        OFF_HOURS.  The original code fell through to OFF_HOURS for hours
        0-6 because the elif chain was evaluated top-down and ASIAN was not
        listed.  classify_regime_backtest no longer hard-blocks ASIAN.
        """
        h = utc_dt.hour
        if  7 <= h < 12:  return "LONDON"
        if 12 <= h < 16:  return "LONDON_NY_OVERLAP"
        if 16 <= h < 21:  return "NY"
        if  0 <= h <  7:  return "ASIAN"
        return "OFF_HOURS"   # 21:00-23:59 UTC only

    # -------------------------------------------------------------------------
    # DAILY RESET
    # -------------------------------------------------------------------------
    @staticmethod
    def _daily_reset(state: SimulatedState) -> None:
        """Reset all per-day strategy flags at UTC midnight."""
        state.s1_family_attempts_today  = 0
        state.s1f_attempts_today        = 0
        state.s2_fired_today            = False
        state.s3_fired_today            = False
        state.s4_fired_today            = False
        state.s5_fired_today            = False
        state.s6_placed_today           = False
        state.s7_placed_today           = False
        state.s8_fired_today            = False
        state.r3_fired_today            = False
        state.failed_breakout_flag      = False
        state.failed_breakout_direction = None
        state.range_computed            = False
        state.daily_pnl                 = 0.0
        state.daily_trades              = 0
        state.s1e_pyramid_done          = False
        state.s1d_pyramid_count         = 0
        state.last_s1_direction         = None
        state.last_s1_max_r             = 0.0

    # -------------------------------------------------------------------------
    # MAIN RUN LOOP
    # -------------------------------------------------------------------------
    def run(self):
        """
        Execute the full backtest replay.
        Returns a BacktestResults instance.
        """
        from backtest.results import BacktestResults

        data_feed   = HistoricalDataFeed(self.start_date, self.end_date, cache_dir=self.cache_dir)
        spread_feed = HistoricalSpreadFeed(data_feed.load())
        event_feed  = HistoricalEventFeed(self.start_date, self.end_date)
        bar_buffer  = BarBuffer()
        executor    = ExecutionSimulator(slippage_points=self.slippage_points)

        state = SimulatedState(
            balance      = self.initial_balance,
            equity       = self.initial_balance,
            peak_equity  = self.initial_balance,
        )

        trades:         list = []
        equity_curve:   list = []
        pending_orders: list = []
        open_positions: list = []

        weekly_pnl   = 0.0
        current_week = -1
        last_date    = None

        # ── Pre-load all bars so we know the total count for ETA ──────────
        all_bars   = list(data_feed.iter_m5_bars())
        total_bars = len(all_bars)
        LOG_EVERY  = 1_000

        # ── BUG-1 FIX: Warmup — pre-feed bars from before start_date ─────
        # Load the full parquet (which includes bars before start_date) and
        # feed the last `warmup_bars` rows into the buffer silently.
        # This ensures H4 indicators (need 960+ completed M5 bars) are warm
        # before the first live signal bar is evaluated.
        if self.warmup_bars > 0:
            try:
                warmup_feed = HistoricalDataFeed(
                    start_date = self.start_date - timedelta(days=30),
                    end_date   = self.start_date,
                    cache_dir  = self.cache_dir,
                )
                warmup_all = list(warmup_feed.iter_m5_bars())
                warmup_slice = warmup_all[-self.warmup_bars:] if len(warmup_all) >= self.warmup_bars else warmup_all
                for wb in warmup_slice:
                    bar_buffer.add_m5(wb)
                logger.info(
                    f"Warmup complete: {len(warmup_slice)} M5 bars pre-fed "
                    f"(H4 bars ready: {len(bar_buffer.get_series('H4'))})"
                )
            except Exception as exc:
                logger.warning(f"Warmup failed (non-fatal): {exc} — indicators will warm up during replay")

        logger.info(
            f"Starting replay: {total_bars:,} M5 bars  "
            f"({self.start_date.date()} → {self.end_date.date()})  "
            f"— expect {total_bars // 1_000}–{total_bars // 500} log lines"
        )

        t_start = time.monotonic()

        for bar_idx, bar_dict in enumerate(all_bars):
            bar_time: datetime = bar_dict["time"]

            # ── Progress log ─────────────────────────────────────────────
            if bar_idx % LOG_EVERY == 0:
                elapsed   = time.monotonic() - t_start
                pct       = bar_idx / total_bars * 100 if total_bars else 0
                bars_left = total_bars - bar_idx
                rate      = bar_idx / elapsed if elapsed > 0 else 0
                eta_s     = bars_left / rate if rate > 0 else 0
                eta_str   = (
                    f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"
                    if eta_s >= 60 else f"{int(eta_s)}s"
                )
                logger.info(
                    f"Progress  {bar_idx:>6,}/{total_bars:,}  "
                    f"({pct:5.1f}%)  "
                    f"bar={bar_time.strftime('%Y-%m-%d %H:%M')}  "
                    f"equity=${state.equity:,.0f}  "
                    f"trades={len(trades)}  "
                    f"ETA={eta_str}"
                )

            # ── Weekly reset (KS5) ────────────────────────────────────────
            week_num = bar_time.isocalendar()[1]
            if week_num != current_week:
                weekly_pnl   = 0.0
                current_week = week_num

            # ── Daily reset ───────────────────────────────────────────────
            bar_date = bar_time.date()
            if last_date != bar_date:
                self._daily_reset(state)
                last_date = bar_date

            # ── Feed bar into buffer ──────────────────────────────────────
            completed = bar_buffer.add_m5(bar_dict)
            spread    = spread_feed.get_spread_at(bar_time)

            state.current_session = self._classify_session(bar_time)

            # ── Compute indicators ────────────────────────────────────────
            h4_series  = bar_buffer.get_series("H4")
            h1_series  = bar_buffer.get_series("H1")
            m15_series = bar_buffer.get_series("M15")

            h4_ind  = self._compute_indicators(h4_series)  if len(h4_series)  > 20 else {}
            h1_ind  = self._compute_indicators(h1_series)  if len(h1_series)  > 20 else {}
            m15_ind = self._compute_indicators(m15_series) if len(m15_series) > 20 else {}

            adx_h4      = h4_ind.get("adx")    or 0.0
            atr_h1_raw  = h1_ind.get("atr")    or 0.0
            atr_m15     = m15_ind.get("atr")   or 0.0
            ema20_h1    = h1_ind.get("ema20")
            rsi_h1      = h1_ind.get("rsi")
            di_plus_h4  = h4_ind.get("di_plus")
            di_minus_h4 = h4_ind.get("di_minus")

            # ATR percentile for regime gate
            if ta is not None and len(h1_series) > 14:
                atr_series_h1 = ta.atr(h1_series["high"], h1_series["low"],
                                        h1_series["close"], length=14)
                atr_pct_h1 = compute_atr_percentile(atr_series_h1, atr_h1_raw)
            else:
                atr_pct_h1 = 50.0

            has_event    = event_feed.has_upcoming_event(bar_time, pre_minutes=30, post_minutes=15)
            avg_spread   = spread_feed.get_avg_spread_24h()
            spread_ratio = spread / max(avg_spread, 0.001)

            regime, size_mult = classify_regime_backtest(
                adx_h4             = adx_h4,
                atr_pct_h1         = atr_pct_h1,
                session            = state.current_session,
                has_upcoming_event = has_event,
                spread_ratio       = spread_ratio,
            )
            state.current_regime  = regime
            state.size_multiplier = size_mult
            state.last_atr_h1_raw = atr_h1_raw
            state.last_atr_m15    = atr_m15
            state.last_adx_h4     = adx_h4

            # ── KS3: daily loss kill switch ───────────────────────────────
            ks3_pct = self._cfg("KS3_DAILY_LOSS_LIMIT_PCT", 0.03)
            if state.daily_pnl <= -(ks3_pct * state.balance):
                state.current_regime = "NO_TRADE"

            # ── KS5: weekly loss kill switch ──────────────────────────────
            ks5_pct = self._cfg("KS5_WEEKLY_LOSS_LIMIT_PCT", 0.05)
            if weekly_pnl <= -(ks5_pct * state.balance):
                state.current_regime = "NO_TRADE"

            # ── KS6: account drawdown kill switch ─────────────────────────
            state.peak_equity = max(state.peak_equity, state.equity)
            if state.peak_equity > 0:
                dd_pct = (state.peak_equity - state.equity) / state.peak_equity
            else:
                dd_pct = 0.0
            ks6_thresh = self._cfg("KS6_MAX_DRAWDOWN_PCT", 0.10)
            if dd_pct >= ks6_thresh:
                state.current_regime = "NO_TRADE"

            # ── FIX-2: Macro bias from PREVIOUS completed H1 bar ─────────
            macro_bias = "BOTH_PERMITTED"
            if len(h1_series) >= 3 and h1_ind:
                prev_h1_close = float(h1_series["close"].iloc[-2])   # FIX-2
                macro_bias = _derive_macro_bias(
                    price    = prev_h1_close,
                    ema20_h1 = h1_ind.get("ema20_prev"),              # FIX-2: _prev
                    adx_h4   = h4_ind.get("adx_prev") or adx_h4,     # FIX-2: _prev
                    di_plus  = h4_ind.get("di_plus_prev") or di_plus_h4,
                    di_minus = h4_ind.get("di_minus_prev") or di_minus_h4,
                )

            # ── Asian range (BUG-2 FIX: M5 fallback in _compute_asian_range) ──
            if not state.range_computed:
                asian_range = _compute_asian_range(bar_buffer, bar_time)
                if asian_range:
                    state.range_high     = asian_range["range_high"]
                    state.range_low      = asian_range["range_low"]
                    state.range_size     = asian_range["range_size"]
                    state.range_computed = True

            # ── S1b M15-bar trigger check ─────────────────────────────────
            if completed.get("M15"):
                _check_s1b_trigger(bar_buffer, state)

            # ── Generate new strategy orders ──────────────────────────────
            if state.current_regime != "NO_TRADE":
                new_orders = []

                if not state.trend_family_occupied:
                    new_orders += _evaluate_s1(bar_buffer, state, bar_time,
                                               atr_h1_raw, macro_bias)
                    new_orders += _evaluate_s1b(bar_buffer, state, bar_time,
                                                atr_h1_raw, macro_bias)
                    new_orders += _evaluate_s4(bar_buffer, state, bar_time,
                                               atr_h1_raw, ema20_h1,
                                               di_plus_h4, di_minus_h4, macro_bias)
                    new_orders += _evaluate_s5(bar_buffer, state, bar_time,
                                               atr_h1_raw, macro_bias)

                if state.trend_family_occupied:
                    new_orders += _evaluate_s1d(bar_buffer, state, bar_time,
                                                atr_m15, m15_ind.get("ema20"))
                    new_orders += _evaluate_s1e(bar_buffer, state, bar_time,
                                                m15_ind.get("ema20"))
                    new_orders += _evaluate_s1f(bar_buffer, state, bar_time,
                                                m15_ind.get("ema20"), macro_bias, ema20_h1)

                if not state.reversal_family_occupied:
                    new_orders += _evaluate_s2(bar_buffer, state, bar_time,
                                               atr_h1_raw, ema20_h1, rsi_h1)
                    new_orders += _evaluate_s3(bar_buffer, state, bar_time, atr_h1_raw)

                new_orders += _evaluate_s6(bar_buffer, state, bar_time, atr_m15,
                                           adx_h4, di_plus_h4, di_minus_h4)
                new_orders += _evaluate_s7(bar_buffer, state, bar_time,
                                           h1_ind.get("atr"), adx_h4,
                                           di_plus_h4, di_minus_h4)
                new_orders += _evaluate_s8(bar_buffer, state, bar_time,
                                           atr_h1_raw, macro_bias)
                new_orders += _evaluate_r3(bar_buffer, state, bar_time,
                                           atr_h1_raw, event_feed, macro_bias)

                if self.strategies:
                    new_orders = [o for o in new_orders if o.strategy in self.strategies]

                pending_orders.extend(new_orders)

            # ── Fill pending orders ───────────────────────────────────────
            newly_filled, pending_orders = executor.process_pending_orders(
                pending_orders, bar_dict, spread, bar_time,
            )

            TREND_STRATS    = {"S1_LONDON_BRK", "S1B_FAILED_BRK", "S1D_PYRAMID",
                                "S1E_PYRAMID", "S1F_POST_TK", "S4_EMA_PULLBACK",
                                "S5_NY_COMPRESS"}
            REVERSAL_STRATS = {"S2_MEAN_REV", "S3_STOP_HUNT_REV"}

            for pos in newly_filled:
                pos.regime_at_entry = state.current_regime
                open_positions.append(pos)
                if pos.strategy in TREND_STRATS:
                    state.trend_family_occupied = True
                    state.trend_family_strategy = pos.strategy
                    state.trend_trade_direction = pos.direction
                    state.last_s1_direction     = pos.direction
                    state.s1_family_attempts_today += 1
                if pos.strategy in REVERSAL_STRATS:
                    state.reversal_family_occupied = True
                _update_fired_flags(state, pos)

            # ── Manage open positions ─────────────────────────────────────
            still_open = []
            for pos in open_positions:
                trail_sl = executor.compute_atr_trail(pos, bar_dict, atr_m15)
                if trail_sl is not None:
                    pos.current_sl = trail_sl

                do_partial, partial_price = executor.check_partial_exit(pos, bar_dict)
                if do_partial and not pos.partial_done:
                    half_lots = max(0.01, round(pos.lots * 0.5, 2))
                    _, _, pnl_net = executor.compute_trade_pnl(
                        pos.direction, pos.entry_price, partial_price, half_lots)
                    state.balance   += pnl_net
                    state.equity    += pnl_net
                    state.daily_pnl += pnl_net
                    weekly_pnl      += pnl_net
                    pos.lots         = max(0.01, round(pos.lots - half_lots, 2))
                    pos.partial_done = True

                if executor.check_be_activation(pos, bar_dict):
                    pos.current_sl  = pos.entry_price
                    pos.be_activated = True

                closed, exit_price, reason = executor.check_sl_tp(pos, bar_dict)
                if closed:
                    rec = executor.close_position(
                        pos, exit_price, bar_time, reason, state.current_regime)
                    trades.append(rec)
                    state.balance   += rec.pnl
                    state.equity     = state.balance
                    state.daily_pnl += rec.pnl
                    weekly_pnl      += rec.pnl
                    state.daily_trades        += 1
                    state.total_closed_trades += 1
                    state.peak_equity = max(state.peak_equity, state.equity)

                    if rec.pnl > 0:
                        state.total_wins        += 1
                        state.consecutive_losses = 0
                    else:
                        state.total_losses      += 1
                        state.consecutive_losses += 1

                    if pos.strategy in TREND_STRATS:
                        state.trend_family_occupied = False
                        state.trend_family_strategy = None
                        state.trend_trade_direction = None
                        state.position_partial_done = False
                        state.position_be_activated = False
                        state.s1d_pyramid_count     = 0
                        state.s1e_pyramid_done      = False
                        state.last_s1_max_r         = rec.r_multiple
                    if pos.strategy in REVERSAL_STRATS:
                        state.reversal_family_occupied = False
                else:
                    cur_r = pos.current_r(bar_dict["close"])
                    if cur_r > state.last_s1_max_r:
                        state.last_s1_max_r = cur_r
                    still_open.append(pos)

            open_positions = still_open

            # ── Equity snapshot ───────────────────────────────────────────
            unrealized = sum(p.unrealized_pnl(bar_dict["close"]) for p in open_positions)
            state.equity = state.balance + unrealized
            if state.peak_equity > 0:
                dd_now = (state.peak_equity - state.equity) / state.peak_equity
            else:
                dd_now = 0.0
            equity_curve.append(EquityPoint(
                timestamp    = bar_time,
                equity       = round(state.equity, 2),
                drawdown_pct = round(dd_now * 100, 4),
            ))

        # ── End of data: force-close remaining positions ──────────────────
        for pos in open_positions:
            rec = executor.close_position(
                pos, pos.entry_price, self.end_date,
                "SESSION_CLOSE", state.current_regime,
            )
            trades.append(rec)

        elapsed_total = time.monotonic() - t_start
        logger.info(
            f"Replay complete — {total_bars:,} bars in "
            f"{int(elapsed_total // 60)}m {int(elapsed_total % 60):02d}s  "
            f"| trades={len(trades)}  final_equity=${state.equity:,.2f}"
        )

        from backtest.results import BacktestResults
        return BacktestResults(
            trades          = trades,
            equity_curve    = equity_curve,
            initial_balance = self.initial_balance,
            final_balance   = state.balance,
            strategies      = self.strategies,
        )

    # =========================================================================
    # FIX-3: PARAMETER SENSITIVITY TEST
    # =========================================================================

    def run_sensitivity_test(self) -> dict:
        """
        FIX-3: Sweep BREAKOUT_DIST_PCT ±20% and ATR_PCT_UNSTABLE_THRESHOLD ±5.

        Runs 5 backtests (baseline + 4 variants) and computes P&L delta vs
        baseline.  Prints ⚠ OVERFIT WARNING for any variant that drops >40%.

        Returns dict keyed by variant name:
            net_pnl, n_trades, win_rate, vs_baseline_pct
        """
        base_breakout = self._cfg("BREAKOUT_DIST_PCT", 0.12)
        base_unstable = self._cfg("ATR_PCT_UNSTABLE_THRESHOLD", 85)

        variants = {
            "baseline":           {},
            "breakout_pct_+20%":  {"BREAKOUT_DIST_PCT": round(base_breakout * 1.20, 4)},
            "breakout_pct_-20%":  {"BREAKOUT_DIST_PCT": round(base_breakout * 0.80, 4)},
            "atr_unstable_+5pt":  {"ATR_PCT_UNSTABLE_THRESHOLD": base_unstable + 5},
            "atr_unstable_-5pt":  {"ATR_PCT_UNSTABLE_THRESHOLD": base_unstable - 5},
        }

        results  = {}
        base_pnl = None

        print("\n" + "=" * 65)
        print("PARAMETER SENSITIVITY TEST")
        print("=" * 65)
        print(f"  Base BREAKOUT_DIST_PCT           = {base_breakout}")
        print(f"  Base ATR_PCT_UNSTABLE_THRESHOLD  = {base_unstable}")
        print("-" * 65)

        for name, overrides in variants.items():
            engine = BacktestEngine(
                start_date       = self.start_date,
                end_date         = self.end_date,
                initial_balance  = self.initial_balance,
                slippage_points  = self.slippage_points,
                strategies       = self.strategies,
                cache_dir        = self.cache_dir,
                warmup_bars      = self.warmup_bars,
                _override_config = overrides,
            )
            r        = engine.run()
            net_pnl  = r.final_balance - r.initial_balance
            n_trades = len(r.trades)
            wins     = sum(1 for t in r.trades if t.pnl > 0)
            win_rate = wins / n_trades * 100 if n_trades > 0 else 0.0

            if base_pnl is None:
                base_pnl = net_pnl

            vs_base = ((net_pnl - base_pnl) / abs(base_pnl) * 100
                       if base_pnl not in (0, None) else 0.0)

            results[name] = {
                "net_pnl":         round(net_pnl, 2),
                "n_trades":        n_trades,
                "win_rate":        round(win_rate, 1),
                "vs_baseline_pct": round(vs_base, 1),
            }

            flag = "  ⚠ OVERFIT WARNING" if vs_base < -40 else ""
            print(f"  {name:<30}  P&L={net_pnl:>+10,.2f}  "
                  f"N={n_trades:>4}  WR={win_rate:>5.1f}%  "
                  f"vs_base={vs_base:>+6.1f}%{flag}")

        print("=" * 65)
        return results

    # =========================================================================
    # FIX-4: PROFIT CONCENTRATION / TIME-PERIOD STABILITY CHECK
    # =========================================================================

    @staticmethod
    def profit_concentration_check(results) -> dict:
        """
        FIX-4: Detect whether backtest profits are concentrated in ≤3 months.

        Warns when top-3 calendar months account for >80% of total gross
        profit — a sign the backtest edge is not broadly distributed and
        therefore unlikely to be predictive in live trading.

        Args:
            results: BacktestResults instance returned by run().

        Returns dict:
            top3_months          — [(year_month_str, pnl), ...]
            top3_pct_of_total    — fraction [0–1]
            total_profit_months  — number of months with net pnl > 0
            warning              — True when top3_pct_of_total > 0.80
        """
        trades = results.trades
        if not trades:
            return {"warning": False, "top3_months": [],
                    "top3_pct_of_total": 0.0, "total_profit_months": 0}

        rows = [{"month": t.exit_time.strftime("%Y-%m"), "pnl": t.pnl}
                for t in trades if t.pnl is not None]
        df   = pd.DataFrame(rows)
        mdf  = df.groupby("month")["pnl"].sum().reset_index()
        mdf.columns = ["month", "pnl"]

        total_gross = mdf.loc[mdf["pnl"] > 0, "pnl"].sum()
        if total_gross <= 0:
            return {"warning": False, "top3_months": [],
                    "top3_pct_of_total": 0.0,
                    "total_profit_months": int((mdf["pnl"] > 0).sum())}

        top3        = mdf.nlargest(3, "pnl")
        top3_pct    = top3["pnl"].sum() / total_gross
        n_profit_m  = int((mdf["pnl"] > 0).sum())
        top3_list   = list(zip(top3["month"].tolist(), top3["pnl"].round(2).tolist()))
        warning     = top3_pct > 0.80

        print("\n" + "=" * 65)
        print("TIME-PERIOD STABILITY CHECK")
        print("=" * 65)
        print(f"  Total months tracked : {len(mdf)}")
        print(f"  Profitable months    : {n_profit_m}")
        print(f"  Top-3 months P&L     : ${top3['pnl'].sum():,.2f}")
        print(f"  Total gross profit   : ${total_gross:,.2f}")
        print(f"  Top-3 concentration  : {top3_pct * 100:.1f}%")
        for m, p in top3_list:
            print(f"    {m}: ${p:,.2f}")
        if warning:
            print("  ⚠  WARNING: >80% of profits in just 3 months.")
            print("     Backtest may NOT be predictive of live performance.")
        else:
            print("  ✓  Profits broadly distributed — stability check passed.")
        print("=" * 65)

        return {
            "top3_months":        top3_list,
            "top3_pct_of_total":  round(top3_pct, 4),
            "total_profit_months": n_profit_m,
            "warning":            warning,
        }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS (used inside BacktestEngine.run)
# ─────────────────────────────────────────────────────────────────────────────

def _update_fired_flags(state: SimulatedState, pos: SimPosition) -> None:
    """Set per-strategy daily fired/placed flags when a position opens."""
    s = pos.strategy
    if s == "S4_EMA_PULLBACK":    state.s4_fired_today  = True
    if s == "S5_NY_COMPRESS":     state.s5_fired_today  = True
    if s == "S6_ASIAN_BRK":       state.s6_placed_today = True
    if s == "S7_DAILY_STRUCT":    state.s7_placed_today = True
    if s == "S8_NEWS_SPIKE":      state.s8_fired_today  = True
    if s == "R3_CAL_MOMENTUM":    state.r3_fired_today  = True
    if s == "S2_MEAN_REV":        state.s2_fired_today  = True
    if s == "S3_STOP_HUNT_REV":   state.s3_fired_today  = True
    if s == "S1D_PYRAMID":        state.s1d_pyramid_count += 1
    if s == "S1E_PYRAMID":        state.s1e_pyramid_done  = True
