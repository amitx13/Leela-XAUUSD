"""
backtest/models.py — Data classes for the backtesting framework.

SimOrder:    Represents a pending order in the simulation.
SimPosition: Represents an open position being tracked.
TradeRecord: Immutable record of a completed (closed) trade.

All timestamps are timezone-aware UTC (pytz.utc).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SimOrder:
    """
    A pending order waiting to be filled by the execution simulator.

    order_type values:
        BUY_STOP, SELL_STOP, BUY_LIMIT, SELL_LIMIT, MARKET

    direction values:
        LONG, SHORT
    """
    strategy: str
    direction: str          # "LONG" or "SHORT"
    order_type: str         # "BUY_STOP", "SELL_STOP", "BUY_LIMIT", "SELL_LIMIT", "MARKET"
    price: float
    sl: float
    tp: Optional[float] = None
    lots: float = 0.01
    expiry: Optional[datetime] = None
    placed_time: Optional[datetime] = None
    # Metadata for tracking
    tag: str = ""           # e.g. "s7_buy_leg", "s7_sell_leg" for OCO pairs
    linked_tag: str = ""    # tag of the opposite OCO leg to cancel on fill


@dataclass
class SimPosition:
    """
    An open position being tracked through the simulation.

    Tracks entry details, current stop level, and management state
    (partial exit, breakeven activation, trailing stop).
    """
    strategy: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    lots: float
    stop_price_original: float
    current_sl: float
    tp: Optional[float] = None
    regime_at_entry: str = ""
    partial_done: bool = False
    be_activated: bool = False
    # Tracking metrics
    max_r: float = 0.0
    max_favorable: float = 0.0

    @property
    def stop_distance(self) -> float:
        """Original risk distance in points."""
        return abs(self.entry_price - self.stop_price_original)

    def current_r(self, price: float) -> float:
        """Current R-multiple at given price."""
        sd = self.stop_distance
        if sd == 0:
            return 0.0
        if self.direction == "LONG":
            return (price - self.entry_price) / sd
        else:
            return (self.entry_price - price) / sd

    def unrealized_pnl(self, price: float) -> float:
        """Unrealized P&L in USD at given price (XAUUSD: 100 oz/lot)."""
        if self.direction == "LONG":
            return (price - self.entry_price) * self.lots * 100.0
        else:
            return (self.entry_price - price) * self.lots * 100.0


@dataclass
class TradeRecord:
    """
    Immutable record of a completed trade.

    pnl is NET of commission ($7.00/lot round trip).
    r_multiple = pnl_gross / (stop_distance * lots * 100).
    """
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lots: float
    pnl: float              # net of commission
    pnl_gross: float         # before commission
    r_multiple: float
    exit_reason: str         # "SL", "TP", "PARTIAL", "BE", "ATR_TRAIL", "TIME_KILL", "SESSION_CLOSE"
    regime_at_entry: str
    regime_at_exit: str
    stop_original: float
    commission: float        # total commission for this trade


@dataclass
class EquityPoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity: float
    drawdown_pct: float = 0.0


@dataclass
class SimulatedState:
    """
    Aggregated simulation state passed through the backtest loop.

    Mirrors the live system's `state` dict but as a structured object
    for clarity. The engine converts this to a dict when needed by
    strategy evaluation functions.
    """
    balance: float = 10000.0
    equity: float = 10000.0
    peak_equity: float = 10000.0

    # Regime
    current_regime: str = "NO_TRADE"
    size_multiplier: float = 0.0
    consecutive_regime_readings: int = 0
    pending_regime_state: Optional[str] = None

    # ATR / ADX cached values
    last_adx_h4: float = 0.0
    last_atr_pct_h1: float = 50.0
    last_atr_h1_raw: float = 20.0
    last_atr_m15: float = 5.0

    # Session tracking
    current_session: str = "OFF_HOURS"

    # Strategy counters (daily reset)
    s1_family_attempts_today: int = 0
    s1f_attempts_today: int = 0
    s7_placed_today: bool = False
    s6_placed_today: bool = False
    s2_fired_today: bool = False

    # Position management
    trend_family_strategy: Optional[str] = None

    # Daily loss tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0

    # Range data for S1
    range_high: float = 0.0
    range_low: float = 0.0
    range_size: float = 0.0
    range_computed: bool = False

    # S7 prev day data
    s7_prev_day_high: float = 0.0
    s7_prev_day_low: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for compatibility with strategy evaluation functions."""
        return {
            "balance": self.balance,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "current_regime": self.current_regime,
            "size_multiplier": self.size_multiplier,
            "consecutive_regime_readings": self.consecutive_regime_readings,
            "pending_regime_state": self.pending_regime_state,
            "last_adx_h4": self.last_adx_h4,
            "last_atr_pct_h1": self.last_atr_pct_h1,
            "last_atr_h1_raw": self.last_atr_h1_raw,
            "last_atr_m15": self.last_atr_m15,
            "current_session": self.current_session,
            "s1_family_attempts_today": self.s1_family_attempts_today,
            "s1f_attempts_today": self.s1f_attempts_today,
            "s7_placed_today": self.s7_placed_today,
            "s6_placed_today": self.s6_placed_today,
            "s2_fired_today": self.s2_fired_today,
            "trend_family_strategy": self.trend_family_strategy,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "range_size": self.range_size,
            "range_computed": self.range_computed,
            "s7_prev_day_high": self.s7_prev_day_high,
            "s7_prev_day_low": self.s7_prev_day_low,
        }
