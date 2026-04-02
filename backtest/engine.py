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
from backtest.data_feed import (
    HistoricalDataFeed, BarBuffer, HistoricalSpreadFeed, HistoricalEventFeed,
)
from backtest.execution_simulator import ExecutionSimulator

logger = logging.getLogger("backtest.engine")

# Add parent dir to path for config imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    import config
except ImportError:
    config = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION (reused from engines/regime_engine.py)
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
    super_thresh = getattr(config, "ATR_PCT_SUPER_THRESHOLD", 55)
    ks2_spread_mult = getattr(config, "KS2_SPREAD_MULTIPLIER", 2.5)

    # Hard NO_TRADE conditions
    if atr_pct_h1 > no_trade_thresh:
        return "NO_TRADE", 0.0
    if has_upcoming_event:
        return "NO_TRADE", 0.0
    if session == "OFF_HOURS":
        return "NO_TRADE", 0.0
    if spread_ratio > ks2_spread_mult:
        return "NO_TRADE", 0.0

    # UNSTABLE
    if atr_pct_h1 > unstable_thresh or adx_h4 > 55 or 18 <= adx_h4 <= 20:
        return "UNSTABLE", 0.4

    # RANGING
    if adx_h4 < 18:
        return "RANGING_CLEAR", 0.7

    # TRENDING — session multiplier
    session_mult = (
        1.0 if session in ("LONDON_NY_OVERLAP", "LONDON", "NY") else 0.7
    )

    if adx_h4 > 35 and atr_pct_h1 > super_thresh:
        # SUPER_TRENDING (without DXY check in backtest — simplified)
        return "NORMAL_TRENDING", round(1.0 * session_mult, 3)
    elif adx_h4 > 26:
        return "NORMAL_TRENDING", round(1.0 * session_mult, 3)
    else:
        return "WEAK_TRENDING", round(0.8 * session_mult, 3)


def compute_atr_percentile(atr_series: pd.Series, current_atr: float) -> float:
    """
    Compute ATR percentile using EWMA weighting (λ=0.94).
    Reuses logic from engines/regime_engine._ewma_percentile().
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
# STRATEGY IMPLEMENTATIONS (simplified for backtest)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_asian_range(
    bar_buffer: BarBuffer, current_time: datetime
) -> Optional[dict]:
    """
    Compute the Asian session range (00:00–07:00 UTC) from M5 bars.
    Used by S1 (London breakout) and S6 (Asian breakout).

    Returns dict with range_high, range_low, range_size, breakout_dist, hunt_threshold
    or None if insufficient data.
    """
    m5_df = bar_buffer.get_series("M5")
    if m5_df.empty:
        return None

    # Get today's date
    today = current_time.date()

    # Filter M5 bars for Asian session today (00:00–07:00 UTC)
    mask = (
        (m5_df["time"].dt.date == today) &
        (m5_df["time"].dt.hour >= 0) &
        (m5_df["time"].dt.hour < 7)
    )
    asian_bars = m5_df[mask]

    if len(asian_bars) < 12:  # Need at least 1 hour of data
        return None

    rh = float(asian_bars["high"].max())
    rl = float(asian_bars["low"].min())
    rs = rh - rl

    min_range = getattr(config, "MIN_RANGE_SIZE_PTS", 10)
    if rs < min_range:
        return None

    breakout_dist_pct = getattr(config, "BREAKOUT_DIST_PCT", 0.12)
    hunt_threshold_pct = getattr(config, "HUNT_THRESHOLD_PCT", 0.08)

    return {
        "range_high": rh,
        "range_low": rl,
        "range_size": rs,
        "breakout_dist": rs * breakout_dist_pct,
        "hunt_threshold": rs * hunt_threshold_pct,
    }


def _get_prev_day_hl(bar_buffer: BarBuffer) -> Optional[dict]:
    """
    Get previous day's high and low from D1 completed bars.
    Returns dict with high, low, range or None.
    """
    d1_df = bar_buffer.get_series("D1")
    if len(d1_df) < 2:
        return None

    prev = d1_df.iloc[-1]  # Last completed D1 bar
    return {
        "high": float(prev["high"]),
        "low": float(prev["low"]),
        "range": float(prev["high"] - prev["low"]),
    }


def _evaluate_s1(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    current_time: datetime,
    atr_h1: float,
) -> list[SimOrder]:
    """
    S1: London Range Breakout.

    Signal: M15 close > range_high + breakout_dist → LONG
            M15 close < range_low  - breakout_dist → SHORT
    Entry:  BUY_STOP / SELL_STOP at breakout level.
    Stop:   ATR-based buffer below/above range.
    TP:     2.5R from entry.

    Only fires during LONDON or LONDON_NY_OVERLAP sessions.
    """
    orders: list[SimOrder] = []

    # Session gate
    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders

    # Regime gate — S1 needs trending
    if state.current_regime not in (
        "SUPER_TRENDING", "NORMAL_TRENDING", "WEAK_TRENDING"
    ):
        return orders

    # Attempt limit
    max_attempts = getattr(config, "MAX_S1_FAMILY_ATTEMPTS", 4)
    if state.s1_family_attempts_today >= max_attempts:
        return orders

    # Time kill: London 16:30
    london_tz = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    if london_time.hour > 16 or (london_time.hour == 16 and london_time.minute >= 30):
        return orders

    # Need range data
    if not state.range_computed:
        return orders

    rh = state.range_high
    rl = state.range_low
    rs = state.range_size

    if rs < getattr(config, "MIN_RANGE_SIZE_PTS", 10):
        return orders

    breakout_dist = rs * getattr(config, "BREAKOUT_DIST_PCT", 0.12)

    # Check last M15 bar
    last_m15 = bar_buffer.get_last_bar("M15")
    if last_m15 is None:
        return orders

    close = last_m15["close"]

    # ATR-based stop buffer
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
        stop = round(rl - stop_buffer, 3)
    else:
        entry = round(rl - breakout_dist, 3)
        stop = round(rh + stop_buffer, 3)

    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return orders

    # Lot sizing: risk-based
    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)

    # TP at 2.5R
    if direction == "LONG":
        tp = round(entry + stop_dist * 2.5, 3)
    else:
        tp = round(entry - stop_dist * 2.5, 3)

    # Time kill expiry
    expiry_london = london_time.replace(hour=16, minute=30, second=0)
    expiry_utc = expiry_london.astimezone(pytz.utc)

    order = SimOrder(
        strategy="S1_LONDON_BRK",
        direction=direction,
        order_type="BUY_STOP" if direction == "LONG" else "SELL_STOP",
        price=entry,
        sl=stop,
        tp=tp,
        lots=lots,
        expiry=expiry_utc,
        placed_time=current_time,
    )
    orders.append(order)
    return orders


def _evaluate_s2(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    current_time: datetime,
    atr_h1: float,
    ema20_h1: Optional[float],
    rsi_h1: Optional[float],
) -> list[SimOrder]:
    """
    S2: Mean Reversion.

    Signal: Price deviates 1.5× ATR from EMA20 H1 + RSI filter.
    Entry:  LIMIT order at 1.5× ATR from EMA20.
    Stop:   2.0× ATR from EMA20.
    TP:     EMA20 (mean reversion target).

    Only fires in RANGING_CLEAR regime.
    """
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

    price = last_m15["close"]
    deviation = abs(price - ema20_h1)
    threshold = atr_h1 * 1.5

    if deviation < threshold:
        return orders

    # RSI filter: oversold for LONG, overbought for SHORT
    if price < ema20_h1 and rsi_h1 < 35:
        direction = "LONG"
    elif price > ema20_h1 and rsi_h1 > 65:
        direction = "SHORT"
    else:
        return orders

    # Entry at current deviation level (limit order)
    entry = round(price, 3)
    stop_dist = atr_h1 * 2.0

    if direction == "LONG":
        stop = round(entry - stop_dist, 3)
        tp = round(ema20_h1, 3)
    else:
        stop = round(entry + stop_dist, 3)
        tp = round(ema20_h1, 3)

    lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)

    # London time kill
    london_tz = pytz.timezone("Europe/London")
    london_time = current_time.astimezone(london_tz)
    expiry_london = london_time.replace(hour=16, minute=30, second=0)
    expiry_utc = expiry_london.astimezone(pytz.utc)

    order = SimOrder(
        strategy="S2_MEAN_REV",
        direction=direction,
        order_type="MARKET",
        price=entry,
        sl=stop,
        tp=tp,
        lots=lots,
        expiry=expiry_utc,
        placed_time=current_time,
    )
    orders.append(order)
    return orders


def _evaluate_s3(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    current_time: datetime,
    atr_h1: float,
) -> list[SimOrder]:
    """
    S3: Stop Hunt Reversal.

    Pattern: Price sweeps below range_low by 0.3× ATR, then reclaims
    within 3 M15 bars. Entry: BUY STOP 2pts above reclaim candle high.

    Simplified for backtest — checks last few M15 bars for sweep+reclaim.
    """
    orders: list[SimOrder] = []

    if state.current_regime in ("NO_TRADE", "UNSTABLE"):
        return orders

    if state.current_session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return orders

    if not state.range_computed or atr_h1 <= 0:
        return orders

    sweep_thresh = getattr(config, "S3_SWEEP_THRESHOLD_ATR", 0.3)
    stop_mult = getattr(config, "S3_STOP_ATR_MULT", 0.5)
    reclaim_offset = getattr(config, "S3_RECLAIM_OFFSET_PTS", 2.0)
    window = getattr(config, "S3_WINDOW_CANDLES", 3)

    m15_df = bar_buffer.get_series("M15")
    if len(m15_df) < window + 1:
        return orders

    recent = m15_df.iloc[-(window + 1):]
    rl = state.range_low
    rh = state.range_high

    # Check for sweep below range_low
    sweep_low = float(recent["low"].min())
    sweep_depth = rl - sweep_low

    if sweep_depth >= atr_h1 * sweep_thresh:
        # Check if price reclaimed above range_low
        last_bar = recent.iloc[-1]
        if last_bar["close"] > rl:
            # Sweep + reclaim detected — LONG entry
            entry = round(float(last_bar["high"]) + reclaim_offset, 3)
            stop = round(sweep_low - atr_h1 * stop_mult, 3)
            stop_dist = abs(entry - stop)

            if stop_dist > 0:
                lots = _calculate_lot_size(
                    state.balance, stop_dist, state.size_multiplier
                )
                tp = round(entry + stop_dist * 2.0, 3)

                order = SimOrder(
                    strategy="S3_STOP_HUNT_REV",
                    direction="LONG",
                    order_type="BUY_STOP",
                    price=entry,
                    sl=stop,
                    tp=tp,
                    lots=lots,
                    expiry=current_time + timedelta(hours=2),
                    placed_time=current_time,
                )
                orders.append(order)

    # Check for sweep above range_high (SHORT)
    sweep_high = float(recent["high"].max())
    sweep_depth_high = sweep_high - rh

    if sweep_depth_high >= atr_h1 * sweep_thresh:
        last_bar = recent.iloc[-1]
        if last_bar["close"] < rh:
            entry = round(float(last_bar["low"]) - reclaim_offset, 3)
            stop = round(sweep_high + atr_h1 * stop_mult, 3)
            stop_dist = abs(entry - stop)

            if stop_dist > 0:
                lots = _calculate_lot_size(
                    state.balance, stop_dist, state.size_multiplier
                )
                tp = round(entry - stop_dist * 2.0, 3)

                order = SimOrder(
                    strategy="S3_STOP_HUNT_REV",
                    direction="SHORT",
                    order_type="SELL_STOP",
                    price=entry,
                    sl=stop,
                    tp=tp,
                    lots=lots,
                    expiry=current_time + timedelta(hours=2),
                    placed_time=current_time,
                )
                orders.append(order)

    return orders


def _evaluate_s6(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    current_time: datetime,
    atr_m15: float,
) -> list[SimOrder]:
    """
    S6: Asian Range Breakout (dual pending orders).

    Places BUY STOP above Asian high and SELL STOP below Asian low.
    Fires during Asian session (00:00–05:30 UTC).
    """
    orders: list[SimOrder] = []

    if state.s6_placed_today:
        return orders

    if state.current_regime in ("NO_TRADE",):
        return orders

    # S6 places during Asian session
    utc_hour = current_time.hour
    utc_min = current_time.minute
    if not (0 <= utc_hour < 5 or (utc_hour == 5 and utc_min <= 30)):
        return orders

    if not state.range_computed:
        return orders

    rh = state.range_high
    rl = state.range_low
    rs = state.range_size

    min_range = getattr(config, "S6_MIN_RANGE_PTS", 8.0)
    if rs < min_range:
        return orders

    dist_pts = getattr(config, "S6_BREAKOUT_DIST_PTS", 2.0)
    stop_mult = getattr(config, "S6_STOP_ATR_MULT", 0.5)

    stop_dist = max(atr_m15 * stop_mult, 5.0) if atr_m15 > 0 else rs * 0.3

    # BUY STOP
    buy_entry = round(rh + dist_pts, 3)
    buy_stop = round(buy_entry - stop_dist, 3)
    buy_lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    buy_tp = round(buy_entry + stop_dist * 2.0, 3)

    # SELL STOP
    sell_entry = round(rl - dist_pts, 3)
    sell_stop = round(sell_entry + stop_dist, 3)
    sell_lots = _calculate_lot_size(state.balance, stop_dist, state.size_multiplier)
    sell_tp = round(sell_entry - stop_dist * 2.0, 3)

    # Expiry at London open (08:00 UTC)
    expiry = current_time.replace(hour=8, minute=0, second=0)
    if expiry <= current_time:
        expiry += timedelta(days=1)

    orders.append(SimOrder(
        strategy="S6_ASIAN_BRK",
        direction="LONG",
        order_type="BUY_STOP",
        price=buy_entry,
        sl=buy_stop,
        tp=buy_tp,
        lots=buy_lots,
        expiry=expiry,
        placed_time=current_time,
        tag="s6_buy_leg",
        linked_tag="s6_sell_leg",
    ))
    orders.append(SimOrder(
        strategy="S6_ASIAN_BRK",
        direction="SHORT",
        order_type="SELL_STOP",
        price=sell_entry,
        sl=sell_stop,
        tp=sell_tp,
        lots=sell_lots,
        expiry=expiry,
        placed_time=current_time,
        tag="s6_sell_leg",
        linked_tag="s6_buy_leg",
    ))

    return orders


def _evaluate_s7(
    bar_buffer: BarBuffer,
    state: SimulatedState,
    current_time: datetime,
    daily_atr: Optional[float],
) -> list[SimOrder]:
    """
    S7: Daily Structure Breakout.

    Places BUY STOP above prev day high and SELL STOP below prev day low.
    Placed at midnight UTC. On fill, opposite leg is cancelled.
    """
    orders: list[SimOrder] = []

    if state.s7_placed_today:
        return orders

    if state.current_regime == "NO_TRADE":
        return orders

    # S7 places at midnight (00:00–00:15 UTC)
    if current_time.hour != 0 or current_time.minute > 15:
        return orders

    prev_day = _get_prev_day_hl(bar_buffer)
    if prev_day is None:
        return orders

    ph = prev_day["high"]
    pl = prev_day["low"]
    prev_range = prev_day["range"]

    # Inside day filter
    min_ratio = getattr(config, "S7_MIN_RANGE_ATR_RATIO", 0.75)
    if daily_atr and prev_range < daily_atr * min_ratio:
        return orders

    entry_offset = getattr(config, "S7_ENTRY_OFFSET_PTS", 5.0)
    size_mult = getattr(config, "S7_SIZE_MULTIPLIER", 0.5)

    # ATR-based stop (EXP-6 fix)
    stop_dist = daily_atr * 0.5 if daily_atr else prev_range * 0.5

    # BUY STOP
    buy_entry = round(ph + entry_offset, 3)
    buy_stop = round(buy_entry - stop_dist, 3)
    buy_lots = _calculate_lot_size(
        state.balance, stop_dist, state.size_multiplier * size_mult
    )

    # SELL STOP
    sell_entry = round(pl - entry_offset, 3)
    sell_stop = round(sell_entry + stop_dist, 3)
    sell_lots = _calculate_lot_size(
        state.balance, stop_dist, state.size_multiplier * size_mult
    )

    # Expiry: end of day (23:59 UTC)
    expiry = current_time.replace(hour=23, minute=59, second=0)

    orders.append(SimOrder(
        strategy="S7_DAILY_STRUCT",
        direction="LONG",
        order_type="BUY_STOP",
        price=buy_entry,
        sl=buy_stop,
        lots=buy_lots,
        expiry=expiry,
        placed_time=current_time,
        tag="s7_buy_leg",
        linked_tag="s7_sell_leg",
    ))
    orders.append(SimOrder(
        strategy="S7_DAILY_STRUCT",
        direction="SHORT",
        order_type="SELL_STOP",
        price=sell_entry,
        sl=sell_stop,
        lots=sell_lots,
        expiry=expiry,
        placed_time=current_time,
        tag="s7_sell_leg",
        linked_tag="s7_buy_leg",
    ))

    return orders


def _calculate_lot_size(
    balance: float,
    stop_distance: float,
    size_multiplier: float,
    base_risk: float = None,
) -> float:
    """
    Risk-based lot sizing for XAUUSD.

    Formula: lots = (balance * risk_pct * size_mult) / (stop_dist * contract_size)
    Contract size = 100 oz/lot.
    """
    if base_risk is None:
        base_risk = getattr(config, "BASE_RISK_PHASE_1", 0.01)

    if stop_distance <= 0 or size_multiplier <= 0:
        return 0.01

    risk_amount = balance * base_risk * size_multiplier
    lots = risk_amount / (stop_distance * 100.0)

    # Clamp to valid range
    lot_cap = getattr(config, "V1_LOT_HARD_CAP", 0.50)
    lots = max(0.01, min(round(lots, 2), lot_cap))
    return lots


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
            initial_balance=10000.0,
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
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.cache_dir = cache_dir

        # Strategies to evaluate (default: S1 + S7 for Phase 1)
        self.strategies = strategies or [
            "S1_LONDON_BRK", "S2_MEAN_REV", "S3_STOP_HUNT_REV",
            "S6_ASIAN_BRK", "S7_DAILY_STRUCT",
        ]

        # Components
        self.data_feed = HistoricalDataFeed(
            start_date, end_date, cache_dir=cache_dir
        )
        self.bar_buffer = BarBuffer()
        self.exec_sim = ExecutionSimulator(slippage_points=slippage_points)
        self.spread_feed: Optional[HistoricalSpreadFeed] = None
        self.event_feed: Optional[HistoricalEventFeed] = None

        # State
        self.state = SimulatedState(
            balance=initial_balance,
            equity=initial_balance,
            peak_equity=initial_balance,
        )

        # Tracking
        self.pending_orders: list[SimOrder] = []
        self.open_positions: list[SimPosition] = []
        self.closed_trades: list[TradeRecord] = []
        self.equity_curve: list[EquityPoint] = []

        # Indicator cache
        self._last_adx_h4: float = 25.0
        self._last_atr_pct_h1: float = 50.0
        self._last_atr_h1_raw: float = 20.0
        self._last_atr_m15: float = 5.0
        self._last_ema20_h1: Optional[float] = None
        self._last_ema20_m15: Optional[float] = None
        self._last_rsi_h1: Optional[float] = None
        self._last_daily_atr: Optional[float] = None

        # Day tracking
        self._current_day: Optional[int] = None
        self._bars_processed: int = 0
        self._last_regime_update: Optional[datetime] = None

    def run(self) -> "BacktestResults":
        """
        Execute the backtest. Main replay loop.

        Returns BacktestResults object with all analytics.
        """
        from backtest.results import BacktestResults

        logger.info(
            f"Starting backtest: {self.start_date} to {self.end_date}, "
            f"balance=${self.initial_balance:,.2f}, "
            f"strategies={self.strategies}"
        )

        # Load data
        m5_df = self.data_feed.load()
        self.spread_feed = HistoricalSpreadFeed(m5_df)
        self.event_feed = HistoricalEventFeed(self.start_date, self.end_date)

        # Record initial equity
        self.equity_curve.append(EquityPoint(
            timestamp=self.start_date,
            equity=self.initial_balance,
        ))

        total_bars = len(m5_df)
        log_interval = max(total_bars // 20, 1)  # Log progress ~20 times

        # ── MAIN LOOP: iterate M5 bars ──────────────────────────────────────
        for bar in self.data_feed.iter_m5_bars():
            self._bars_processed += 1
            current_time = bar["time"]

            # Progress logging
            if self._bars_processed % log_interval == 0:
                pct = (self._bars_processed / total_bars) * 100
                logger.info(
                    f"Progress: {pct:.0f}% ({self._bars_processed}/{total_bars}) "
                    f"| Equity: ${self.state.equity:,.2f} "
                    f"| Trades: {len(self.closed_trades)}"
                )

            # Step 1: Update bar buffer (builds higher TF candles)
            completed = self.bar_buffer.add_m5(bar)

            # Step 2: Daily reset check
            self._check_daily_reset(current_time)

            # Step 3: Get current spread
            spread = self.spread_feed.get_spread_at(current_time)

            # Step 4: Update session
            self._update_session(current_time)

            # Step 5: Update indicators when higher TF bars complete
            if completed.get("H1"):
                self._update_h1_indicators()
            if completed.get("H4"):
                self._update_h4_indicators()
            if completed.get("D1"):
                self._update_d1_indicators()
            if completed.get("M15"):
                self._update_m15_indicators()

            # Step 6: Update regime (every 15 min or on H4 completion)
            if completed.get("M15") or completed.get("H4"):
                self._update_regime(current_time, spread)

            # Step 7: Process pending orders (check fills)
            self._process_pending_orders(bar, spread, current_time)

            # Step 8: Manage open positions (SL/TP, partial, BE, ATR trail)
            self._manage_positions(bar, current_time)

            # Step 9: Compute Asian range when session transitions
            if completed.get("M15"):
                self._update_range_data(current_time)

            # Step 10: Evaluate strategies for new signals
            if completed.get("M15"):
                self._evaluate_strategies(current_time)

            # Step 11: Record equity curve point (every M15)
            if completed.get("M15"):
                self._record_equity(current_time, bar["close"])

        # ── END OF LOOP ─────────────────────────────────────────────────────

        # Close any remaining positions at last bar's close
        if self.open_positions:
            last_bar = self.bar_buffer.get_last_bar("M5")
            if last_bar:
                self._close_all_positions(
                    last_bar["close"], last_bar["time"], "BACKTEST_END"
                )

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

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _check_daily_reset(self, current_time: datetime) -> None:
        """Reset daily counters at midnight UTC."""
        day = current_time.timetuple().tm_yday
        if self._current_day is not None and day != self._current_day:
            self.state.s1_family_attempts_today = 0
            self.state.s1f_attempts_today = 0
            self.state.s7_placed_today = False
            self.state.s6_placed_today = False
            self.state.s2_fired_today = False
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.range_computed = False
            logger.debug(f"Daily reset at {current_time}")
        self._current_day = day

    def _update_session(self, current_time: datetime) -> None:
        """Update current session based on timestamp."""
        from utils.session import get_session_for_datetime
        self.state.current_session = get_session_for_datetime(current_time)

    def _update_h1_indicators(self) -> None:
        """Recompute H1 indicators: ATR(14), EMA(20), RSI(14)."""
        h1_df = self.bar_buffer.get_series("H1")
        if len(h1_df) < 20:
            return

        # ATR(14) H1 — RMA (Wilder's)
        atr_mode = getattr(config, "ATR_MAMODE", "RMA")
        atr_period = getattr(config, "ATR_PERIOD", 14)
        atr_series = ta.atr(
            h1_df["high"], h1_df["low"], h1_df["close"],
            length=atr_period, mamode=atr_mode,
        )
        if atr_series is not None and not atr_series.empty:
            last_atr = atr_series.dropna()
            if len(last_atr) > 0:
                self._last_atr_h1_raw = float(last_atr.iloc[-1])
                self.state.last_atr_h1_raw = self._last_atr_h1_raw

                # ATR percentile
                self._last_atr_pct_h1 = compute_atr_percentile(
                    atr_series, self._last_atr_h1_raw
                )
                self.state.last_atr_pct_h1 = self._last_atr_pct_h1

        # EMA(20) H1
        ema_series = ta.ema(h1_df["close"], length=20)
        if ema_series is not None and not ema_series.empty:
            last_ema = ema_series.dropna()
            if len(last_ema) > 0:
                self._last_ema20_h1 = float(last_ema.iloc[-1])

        # RSI(14) H1
        rsi_series = ta.rsi(h1_df["close"], length=14)
        if rsi_series is not None and not rsi_series.empty:
            last_rsi = rsi_series.dropna()
            if len(last_rsi) > 0:
                self._last_rsi_h1 = float(last_rsi.iloc[-1])

    def _update_h4_indicators(self) -> None:
        """Recompute H4 indicators: ADX(14)."""
        h4_df = self.bar_buffer.get_series("H4")
        if len(h4_df) < 30:
            return

        adx_df = ta.adx(h4_df["high"], h4_df["low"], h4_df["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            col = "ADX_14"
            if col in adx_df.columns:
                val = adx_df[col].dropna()
                if len(val) > 0:
                    self._last_adx_h4 = float(val.iloc[-1])
                    self.state.last_adx_h4 = self._last_adx_h4

    def _update_d1_indicators(self) -> None:
        """Recompute D1 indicators: ATR(14) for S7."""
        d1_df = self.bar_buffer.get_series("D1")
        if len(d1_df) < 16:
            return

        atr_mode = getattr(config, "ATR_MAMODE", "RMA")
        atr_series = ta.atr(
            d1_df["high"], d1_df["low"], d1_df["close"],
            length=14, mamode=atr_mode,
        )
        if atr_series is not None and not atr_series.empty:
            last_atr = atr_series.dropna()
            if len(last_atr) > 0:
                self._last_daily_atr = float(last_atr.iloc[-1])

    def _update_m15_indicators(self) -> None:
        """Recompute M15 indicators: ATR(14), EMA(20)."""
        m15_df = self.bar_buffer.get_series("M15")
        if len(m15_df) < 20:
            return

        # ATR(14) M15
        atr_mode = getattr(config, "ATR_MAMODE", "RMA")
        atr_series = ta.atr(
            m15_df["high"], m15_df["low"], m15_df["close"],
            length=14, mamode=atr_mode,
        )
        if atr_series is not None and not atr_series.empty:
            last_atr = atr_series.dropna()
            if len(last_atr) > 0:
                self._last_atr_m15 = float(last_atr.iloc[-1])
                self.state.last_atr_m15 = self._last_atr_m15

        # EMA(20) M15
        ema_series = ta.ema(m15_df["close"], length=20)
        if ema_series is not None and not ema_series.empty:
            last_ema = ema_series.dropna()
            if len(last_ema) > 0:
                self._last_ema20_m15 = float(last_ema.iloc[-1])

    def _update_regime(self, current_time: datetime, spread: float) -> None:
        """Update regime classification."""
        avg_spread = self.spread_feed.get_avg_spread_24h() if self.spread_feed else 25.0
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0

        has_event = False
        if self.event_feed:
            has_event = self.event_feed.has_upcoming_event(current_time)

        regime, mult = classify_regime_backtest(
            adx_h4=self._last_adx_h4,
            atr_pct_h1=self._last_atr_pct_h1,
            session=self.state.current_session,
            has_upcoming_event=has_event,
            spread_ratio=spread_ratio,
        )

        self.state.current_regime = regime
        self.state.size_multiplier = mult
        self._last_regime_update = current_time

    def _update_range_data(self, current_time: datetime) -> None:
        """Compute Asian range for S1/S6 strategies."""
        # Only compute after Asian session ends (07:00 UTC)
        if current_time.hour < 7:
            return

        if self.state.range_computed:
            return

        range_data = _compute_asian_range(self.bar_buffer, current_time)
        if range_data:
            self.state.range_high = range_data["range_high"]
            self.state.range_low = range_data["range_low"]
            self.state.range_size = range_data["range_size"]
            self.state.range_computed = True
            logger.debug(
                f"Asian range computed: {range_data['range_low']:.2f} - "
                f"{range_data['range_high']:.2f} (size={range_data['range_size']:.2f})"
            )

    def _process_pending_orders(
        self, bar: dict, spread: float, current_time: datetime
    ) -> None:
        """Process pending orders — check for fills."""
        if not self.pending_orders:
            return

        filled, remaining = self.exec_sim.process_pending_orders(
            self.pending_orders, bar, spread, current_time
        )

        # Handle OCO (one-cancels-other) for S6/S7 dual pending
        for pos in filled:
            pos.regime_at_entry = self.state.current_regime
            self.open_positions.append(pos)

            # Cancel linked OCO leg
            filled_tags = set()
            for order in self.pending_orders:
                if order in remaining:
                    continue
                if order.tag:
                    filled_tags.add(order.tag)

            # Remove linked orders
            new_remaining = []
            for order in remaining:
                if order.linked_tag and order.linked_tag in filled_tags:
                    logger.debug(
                        f"OCO cancel: {order.strategy} {order.tag} "
                        f"(linked to filled {order.linked_tag})"
                    )
                    continue
                new_remaining.append(order)
            remaining = new_remaining

            # Update strategy counters
            if pos.strategy == "S1_LONDON_BRK":
                self.state.s1_family_attempts_today += 1
            elif pos.strategy == "S7_DAILY_STRUCT":
                self.state.s7_placed_today = True
            elif pos.strategy == "S6_ASIAN_BRK":
                self.state.s6_placed_today = True
            elif pos.strategy == "S2_MEAN_REV":
                self.state.s2_fired_today = True

        self.pending_orders = remaining

    def _manage_positions(self, bar: dict, current_time: datetime) -> None:
        """Manage open positions: SL/TP, partial exit, BE, ATR trail."""
        still_open: list[SimPosition] = []

        for pos in self.open_positions:
            # Update max favorable excursion
            if pos.direction == "LONG":
                pos.max_favorable = max(pos.max_favorable, bar["high"] - pos.entry_price)
            else:
                pos.max_favorable = max(pos.max_favorable, pos.entry_price - bar["low"])

            r = pos.current_r(bar["close"])
            pos.max_r = max(pos.max_r, r)

            # Check SL/TP
            closed, exit_price, reason = self.exec_sim.check_sl_tp(pos, bar)
            if closed:
                trade = self.exec_sim.close_position(
                    pos, exit_price, current_time, reason,
                    regime_at_exit=self.state.current_regime,
                )
                self._record_trade(trade)
                continue

            # Partial exit at 2.0R
            partial_r = getattr(config, "PARTIAL_EXIT_R", 2.0)
            should_partial, partial_price = self.exec_sim.check_partial_exit(
                pos, bar, partial_r
            )
            if should_partial:
                # Close half the position
                half_lots = round(pos.lots / 2, 2)
                if half_lots >= 0.01:
                    partial_trade = self.exec_sim.close_position(
                        SimPosition(
                            strategy=pos.strategy,
                            direction=pos.direction,
                            entry_price=pos.entry_price,
                            entry_time=pos.entry_time,
                            lots=half_lots,
                            stop_price_original=pos.stop_price_original,
                            current_sl=pos.current_sl,
                            tp=pos.tp,
                            regime_at_entry=pos.regime_at_entry,
                        ),
                        partial_price, current_time, "PARTIAL",
                        regime_at_exit=self.state.current_regime,
                    )
                    self._record_trade(partial_trade)
                    pos.lots = round(pos.lots - half_lots, 2)
                    pos.partial_done = True

            # BE activation at 1.5R
            be_r = getattr(config, "BE_ACTIVATION_R", 1.5)
            if self.exec_sim.check_be_activation(pos, bar, be_r):
                pos.current_sl = pos.entry_price
                pos.be_activated = True
                logger.debug(
                    f"BE activated: {pos.strategy} {pos.direction} "
                    f"@ {pos.entry_price:.2f}"
                )

            # ATR trailing stop (only after BE)
            if pos.be_activated:
                trail_mult = getattr(config, "ATR_TRAIL_MULTIPLIER", 2.5)
                new_sl = self.exec_sim.compute_atr_trail(
                    pos, bar, self._last_atr_m15, trail_mult
                )
                if new_sl is not None:
                    pos.current_sl = new_sl

            still_open.append(pos)

        self.open_positions = still_open

    def _evaluate_strategies(self, current_time: datetime) -> None:
        """Evaluate all enabled strategies for new signals."""
        # Daily loss limit check
        daily_loss_limit = getattr(config, "KS3_DAILY_LOSS_LIMIT_PCT", -0.04)
        if self.state.balance > 0:
            daily_loss_pct = self.state.daily_pnl / self.state.balance
            if daily_loss_pct <= daily_loss_limit:
                return  # Daily loss limit hit — no new trades

        new_orders: list[SimOrder] = []

        if "S1_LONDON_BRK" in self.strategies:
            new_orders.extend(_evaluate_s1(
                self.bar_buffer, self.state, current_time, self._last_atr_h1_raw
            ))

        if "S2_MEAN_REV" in self.strategies:
            new_orders.extend(_evaluate_s2(
                self.bar_buffer, self.state, current_time,
                self._last_atr_h1_raw, self._last_ema20_h1, self._last_rsi_h1,
            ))

        if "S3_STOP_HUNT_REV" in self.strategies:
            new_orders.extend(_evaluate_s3(
                self.bar_buffer, self.state, current_time, self._last_atr_h1_raw
            ))

        if "S6_ASIAN_BRK" in self.strategies:
            new_orders.extend(_evaluate_s6(
                self.bar_buffer, self.state, current_time, self._last_atr_m15
            ))

        if "S7_DAILY_STRUCT" in self.strategies:
            new_orders.extend(_evaluate_s7(
                self.bar_buffer, self.state, current_time, self._last_daily_atr
            ))

        # Add new orders to pending
        for order in new_orders:
            self.pending_orders.append(order)
            logger.debug(
                f"New order: {order.strategy} {order.direction} "
                f"{order.order_type} @ {order.price:.2f}"
            )

        # Update strategy counters for placed orders
        for order in new_orders:
            if order.strategy == "S7_DAILY_STRUCT":
                self.state.s7_placed_today = True
            elif order.strategy == "S6_ASIAN_BRK":
                self.state.s6_placed_today = True

    def _record_trade(self, trade: TradeRecord) -> None:
        """Record a closed trade and update state."""
        self.closed_trades.append(trade)
        self.state.balance += trade.pnl
        self.state.daily_pnl += trade.pnl
        self.state.daily_trades += 1

        if trade.pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        logger.debug(
            f"Trade closed: {trade.strategy} {trade.direction} "
            f"P&L=${trade.pnl:+.2f} R={trade.r_multiple:+.2f} "
            f"({trade.exit_reason})"
        )

    def _record_equity(self, current_time: datetime, current_price: float) -> None:
        """Record equity curve point."""
        # Equity = balance + unrealized P&L
        unrealized = sum(
            pos.unrealized_pnl(current_price) for pos in self.open_positions
        )
        equity = self.state.balance + unrealized
        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)

        dd_pct = 0.0
        if self.state.peak_equity > 0:
            dd_pct = (self.state.peak_equity - equity) / self.state.peak_equity

        self.equity_curve.append(EquityPoint(
            timestamp=current_time,
            equity=equity,
            drawdown_pct=dd_pct,
        ))

    def _close_all_positions(
        self, price: float, time: datetime, reason: str
    ) -> None:
        """Force-close all open positions."""
        for pos in self.open_positions:
            trade = self.exec_sim.close_position(
                pos, price, time, reason,
                regime_at_exit=self.state.current_regime,
            )
            self._record_trade(trade)
        self.open_positions = []
