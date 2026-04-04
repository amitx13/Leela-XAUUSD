"""
backtest/engine.py — Main replay loop for the backtesting framework.

BacktestEngine orchestrates:
  1. M5 bar iteration from HistoricalDataFeed
  2. Higher-TF candle building via BarBuffer
  3. Indicator computation (ADX, ATR, EMA, RSI) via pandas_ta
  4. Regime classification (reuses logic from engines/regime_engine.py)
  5. Strategy signal generation (S1, S2, S3, S6, S7)
  6. Order fill simulation via ExecutionSimulator
  7. Position management (SL/TP, partial, BE, ATR trail)
  8. Equity curve tracking

CRITICAL DESIGN PRINCIPLE:
  Reuses the EXACT SAME thresholds and logic from config.py and
  engines/regime_engine.py wherever possible. Strategy implementations
  are simplified versions of the live signal_engine.py strategies.
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
        return "SUPER_TRENDING", round(1.2 * session_mult, 3)  # FIX Bug 9: was NORMAL_TRENDING
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


def _evaluate_s1(
    bar_buffer: BarBuffer, state: SimulatedState,
    current_time: datetime, atr_h1: float,
) -> list[SimOrder]:
    """
    S1: London Range Breakout.
    Entry: BUY_STOP/SELL_STOP at range boundary + breakout_dist.
    Stop:  ATR-based buffer. TP: 2.5R.
    """
    orders: list[SimOrder] = []
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders
    if state.current_regime not in ("SUPER_TRENDING", "NORMAL_TRENDING", "WEAK_TRENDING"):
        return orders
    max_attempts = getattr(config, "MAX_S1_FAMILY_ATTEMPTS", 4)
    if state.s1_family_attempts_today >= max_attempts:
        return orders
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
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
    london_tz   = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    expiry_utc  = london_time.replace(hour=16, minute=30, second=0).astimezone(pytz.utc)
    orders.append(SimOrder(
        strategy="S2_MEAN_REV", direction=direction,
        order_type="MARKET", price=entry, sl=stop, tp=tp, lots=lots,
        expiry=expiry_utc, placed_time=current_time,
    ))
    return orders


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
    if state.s3_fired_today:  # FIX Bug 3: gate now works after daily reset fix
        return orders
    if not state.range_computed or atr_h1 <= 0:
        return orders
    sweep_thresh   = getattr(config, "S3_SWEEP_THRESHOLD_ATR", 0.3)
    stop_mult      = getattr(config, "S3_STOP_ATR_MULT", 0.5)
    reclaim_offset = getattr(config, "S3_RECLAIM_OFFSET_PTS", 2.0)
    window         = getattr(config, "S3_WINDOW_CANDLES", 3)
    # FIX Performance: cap at 200 bars
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


def _calculate_lot_size(
    balance: float, stop_distance: float,
    size_multiplier: float, base_risk: float = None,
) -> float:
    """Risk-based lot sizing: lots = (balance * risk_pct * size_mult) / (stop_dist * 100). Matches live main.py / execution path risk sizing assumptions."""
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
            "S1_LONDON_BRK", "S2_MEAN_REV", "S3_STOP_HUNT_REV",
            "S6_ASIAN_BRK", "S7_DAILY_STRUCT",
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
        self._last_di_plus:    Optional[float] = None   # Bug 5 fix
        self._last_di_minus:   Optional[float] = None   # Bug 5 fix

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
            spread = self.spread_feed.get_spread_at(current_time)

            current_hour_key = (current_time.year, current_time.timetuple().tm_yday, current_time.hour)
            if current_hour_key != self._last_session_hour:
                self._update_session(current_time)
                self._last_session_hour = current_hour_key

            if completed.get("H1"): self._update_h1_indicators()
            if completed.get("H4"): self._update_h4_indicators()
            if completed.get("D1"): self._update_d1_indicators()
            if completed.get("M15"): self._update_m15_indicators()

            if completed.get("M15") or completed.get("H4"):
                self._update_regime(current_time, spread)

            self._process_pending_orders(bar, spread, current_time)
            self._manage_positions(bar, current_time)

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

        FIX: original code only reset 6 of ~15 flags.
        Now mirrors the live system's on_new_day() function completely.
        Missing flags: s3/s4/s5/s8_fired_today, s8_armed, s1d/s1e state,
        r3_fired_today, stop_hunt_detected, failed_breakout_flag.
        """
        day = current_time.timetuple().tm_yday
        if self._current_day is not None and day != self._current_day:
            # ── Strategy attempt counters ──────────────────────────────────
            self.state.s1_family_attempts_today = 0
            self.state.s1f_attempts_today       = 0
            # ── Strategy fired-today flags (ALL of them) ───────────────────
            self.state.s2_fired_today           = False
            self.state.s3_fired_today           = False
            self.state.s4_fired_today           = False
            self.state.s5_fired_today           = False
            self.state.s8_fired_today           = False
            # ── Placed-today flags ─────────────────────────────────────────
            self.state.s6_placed_today          = False
            self.state.s7_placed_today          = False
            # ── S1D / S1E intra-day state ──────────────────────────────────
            self.state.s1d_ema_touched_today    = False
            self.state.s1d_fired_today          = False
            self.state.s1e_pyramid_done         = False
            # ── S8 arm state ───────────────────────────────────────────────
            self.state.s8_armed                 = False
            self.state.s8_arm_time              = None
            self.state.s8_confirmation_passed   = False
            # ── R3 daily flag ──────────────────────────────────────────────
            self.state.r3_fired_today           = False
            # ── Range state ────────────────────────────────────────────────
            self.state.range_computed           = False
            # ── Account metrics ────────────────────────────────────────────
            self.state.daily_pnl                = 0.0
            self.state.daily_trades             = 0
            # consecutive_losses intentionally not reset; live state preserves loss streaks across midnight
            logger.debug(f"Daily reset at {current_time.date()}")
        self._current_day = day

    # =========================================================================
    # SESSION UPDATE
    # =========================================================================

    def _update_session(self, current_time: datetime) -> None:
        from utils.session import get_session_for_datetime
        self.state.current_session = get_session_for_datetime(current_time)

    # =========================================================================
    # INDICATOR UPDATES
    # =========================================================================

    def _update_h1_indicators(self) -> None:
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

    def _update_h4_indicators(self) -> None:
        """
        Recompute H4: ADX(14) + DI+(14) + DI-(14). Capped at 100 bars.

        FIX Bug 5: original code only read ADX_14 and ignored DMP_14/DMN_14.
        state.last_di_plus_h4 / last_di_minus_h4 were always None, so any
        strategy that checks DI direction (S1/S3/S5) was silently broken.
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
                self._last_adx_h4     = float(v.iloc[-1])
                self.state.last_adx_h4 = self._last_adx_h4
        # FIX: read DI columns
        if "DMP_14" in adx_df.columns:
            v = adx_df["DMP_14"].dropna()
            if len(v) > 0:
                self._last_di_plus            = float(v.iloc[-1])
                self.state.last_di_plus_h4    = self._last_di_plus
        if "DMN_14" in adx_df.columns:
            v = adx_df["DMN_14"].dropna()
            if len(v) > 0:
                self._last_di_minus           = float(v.iloc[-1])
                self.state.last_di_minus_h4   = self._last_di_minus

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
                self._last_atr_m15     = float(v.iloc[-1])
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
            if pos.strategy == "S1_LONDON_BRK":
                self.state.s1_family_attempts_today += 1
            elif pos.strategy == "S7_DAILY_STRUCT":
                self.state.s7_placed_today = True
            elif pos.strategy == "S6_ASIAN_BRK":
                self.state.s6_placed_today = True
            elif pos.strategy == "S2_MEAN_REV":
                self.state.s2_fired_today  = True
            elif pos.strategy == "S3_STOP_HUNT_REV":
                self.state.s3_fired_today  = True  # FIX: was never set
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

        - Calls exec_sim.close_position() WITHOUT passing max_r (Bug 4 fix:
          TradeRecord has no max_r field — stray kwarg caused TypeError).
        - Always calls _update_lane_flags(pos) after recording the trade so
          trend_family_occupied / reversal_family_occupied are cleared.
          (Bug 2 fix: these flags were never reset, blocking all future trades.)
        """
        trade = self.exec_sim.close_position(
            pos, exit_price, current_time, reason,
            regime_at_exit=self.state.current_regime,
            # NOTE: do NOT pass max_r — TradeRecord dataclass has no such field
        )
        self._record_trade(trade)
        self._update_lane_flags(pos)

    def _update_lane_flags(self, pos: SimPosition) -> None:
        """
        Clear position-lane occupied flags when a position is closed.
        Covers both the trend-family lane and the independent S8/R3 lanes.
        """
        strat = pos.strategy
        # Trend family lane
        if strat in (
            "S1_LONDON_BRK", "S2_MEAN_REV", "S3_STOP_HUNT_REV",
            "S4_EMA_PULLBACK", "S5_BREAKOUT_RETEST",
            "S6_ASIAN_BRK", "S7_DAILY_STRUCT",
        ):
            self.state.trend_family_occupied  = False
            self.state.trend_family_strategy  = None
            self.state.open_trade_id          = None
        # S8 independent lane
        elif strat == "S8_NEWS_SPIKE":
            self.state.s8_open_ticket         = None
            self.state.s8_trade_direction     = None
        # R3 independent lane
        elif strat == "R3_CAL_MOMENTUM":
            self.state.r3_open_ticket         = None
            self.state.r3_direction           = None
            self.state.r3_armed               = False

    def _manage_r3_position(
        self, pos: SimPosition, price: float,
        current_time: datetime, bar: dict,
    ) -> bool:
        """
        R3 time-exit: close after 30 minutes if not already SL/TP'd.
        Returns True if position was closed.
        """
        if self.state.r3_open_time is None:
            return False
        elapsed = (current_time - self.state.r3_open_time).total_seconds() / 60
        if elapsed >= 30:
            self._close_position(pos, price, current_time, "TIME_KILL")
            self.state.r3_open_time = None
            return True
        return False

    def _manage_positions(self, bar: dict, current_time: datetime) -> None:
        """
        Manage all open positions each M5 bar.

        FIX Bug 6: _manage_r3_position() is now called for R3 positions.
        FIX Bug 2: _close_position() helper clears lane flags on every close.
        """
        still_open: list[SimPosition] = []

        for pos in self.open_positions:
            # Update MFE / max-R tracking
            if pos.direction == "LONG":
                pos.max_favorable = max(pos.max_favorable, bar["high"] - pos.entry_price)
            else:
                pos.max_favorable = max(pos.max_favorable, pos.entry_price - bar["low"])
            pos.max_r = max(pos.max_r, pos.current_r(bar["close"]))

            # FIX Bug 6: R3 time-exit (was dead code before)
            if pos.strategy == "R3_CAL_MOMENTUM":
                if self._manage_r3_position(pos, bar["close"], current_time, bar):
                    continue  # Position closed by time-exit — skip to next

            # SL / TP check
            closed, exit_price, reason = self.exec_sim.check_sl_tp(pos, bar)
            if closed:
                self._close_position(pos, exit_price, current_time, reason)  # FIX Bug 2 & 4
                continue

            # Partial exit at 2.0R
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
                    # NOTE: do NOT call _update_lane_flags here — position still open

            # BE activation at 1.5R
            be_r = getattr(config, "BE_ACTIVATION_R", 1.5)
            if not pos.be_activated and self.exec_sim.check_be_activation(pos, bar, be_r):
                pos.current_sl   = pos.entry_price
                pos.be_activated = True
                logger.debug(f"BE activated: {pos.strategy} {pos.direction} @ {pos.entry_price:.2f}")

            # ATR trailing stop (after BE)
            if pos.be_activated:
                trail_mult = getattr(config, "ATR_TRAIL_MULTIPLIER", 2.5)
                new_sl = self.exec_sim.compute_atr_trail(pos, bar, self._last_atr_m15, trail_mult)
                if new_sl is not None:
                    pos.current_sl = new_sl

            still_open.append(pos)

        self.open_positions = still_open

    # =========================================================================
    # STRATEGY EVALUATION
    # =========================================================================

    def _evaluate_strategies(self, current_time: datetime) -> None:
        """Evaluate all enabled strategies for new signals."""
        daily_loss_limit = getattr(config, "KS3_DAILY_LOSS_LIMIT_PCT", -0.04)
        if self.state.balance > 0:
            if self.state.daily_pnl / self.state.balance <= daily_loss_limit:
                return

        new_orders: list[SimOrder] = []

        if "S1_LONDON_BRK"    in self.strategies:
            new_orders.extend(_evaluate_s1(self.bar_buffer, self.state, current_time, self._last_atr_h1_raw))
        if "S2_MEAN_REV"       in self.strategies:
            new_orders.extend(_evaluate_s2(self.bar_buffer, self.state, current_time,
                                           self._last_atr_h1_raw, self._last_ema20_h1, self._last_rsi_h1))
        if "S3_STOP_HUNT_REV" in self.strategies:
            new_orders.extend(_evaluate_s3(self.bar_buffer, self.state, current_time, self._last_atr_h1_raw))
        if "S6_ASIAN_BRK"     in self.strategies:
            new_orders.extend(_evaluate_s6(self.bar_buffer, self.state, current_time, self._last_atr_m15))
        if "S7_DAILY_STRUCT"  in self.strategies:
            new_orders.extend(_evaluate_s7(self.bar_buffer, self.state, current_time, self._last_daily_atr))

        for order in new_orders:
            self.pending_orders.append(order)
            logger.debug(f"New order: {order.strategy} {order.direction} {order.order_type} @ {order.price:.2f}")
            # Placed-today flags
            if order.strategy == "S7_DAILY_STRUCT": self.state.s7_placed_today = True
            elif order.strategy == "S6_ASIAN_BRK":  self.state.s6_placed_today = True

    # =========================================================================
    # TRADE RECORDING
    # =========================================================================

    def _record_trade(self, trade: TradeRecord) -> None:
        """Record a closed trade and update account state."""
        self.closed_trades.append(trade)
        self.state.balance    += trade.pnl
        self.state.daily_pnl  += trade.pnl
        self.state.daily_trades += 1
        if trade.pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses  = 0
        logger.debug(
            f"Trade closed: {trade.strategy} {trade.direction} "
            f"P&L=${trade.pnl:+.2f} R={trade.r_multiple:+.2f} ({trade.exit_reason})"
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
            self._close_position(pos, price, time, reason)  # FIX: use helper so lane flags clear
        self.open_positions = []
