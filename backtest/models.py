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
    stop_price: float
    tp_price: Optional[float] = None
    lot_size: float = 0.01
    expiry: Optional[datetime] = None
    placed_time: Optional[datetime] = None
    # Metadata for tracking
    tag: str = ""           # e.g. "s7_buy_leg", "s7_sell_leg" for OCO pairs
    linked_tag: str = ""    # tag of the opposite OCO leg to cancel on fill
    metadata: dict = field(default_factory=dict)


@dataclass
class SimPosition:
    """
    An open position being tracked through the simulation.

    Tracks entry details, current stop level, and management state
    (partial exit, breakeven activation, trailing stop).

    Fields aligned with execution_simulator._try_fill_updated() exactly.
    """
    strategy: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    lot_size: float
    stop_price: float       # current stop level (moves to BE, then trails)
    # Optional fields
    tp_price: Optional[float] = None
    ticket: str = ""        # unique identifier e.g. "BT_42_1234567890.0"
    tag: str = ""           # OCO tag
    regime_at_entry: str = ""
    # Position management flags (stored in metadata dict below)
    metadata: dict = field(default_factory=dict)
    # Tracking metrics
    max_r: float = 0.0
    max_favorable: float = 0.0

    @property
    def stop_price_original(self) -> float:
        """Original stop level stored in metadata (for R-multiple calc)."""
        return self.metadata.get('original_stop', self.stop_price)

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
        """Unrealized P&L in USD at given price (XAUUSD: $100/point/lot)."""
        if self.direction == "LONG":
            return (price - self.entry_price) * self.lot_size * 100.0
        else:
            return (self.entry_price - price) * self.lot_size * 100.0


@dataclass
class TradeRecord:
    """
    Immutable record of a completed trade.

    pnl_net_dollars is NET of commission ($7.00/lot round trip).
    r_multiple = pnl_gross_dollars / (stop_distance * lot_size * 100).

    Fields aligned with execution_simulator._create_trade_record() and
    engine._finalize_backtest() exactly.
    """
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lot_size: float
    pnl_net_dollars: float          # net of commission
    pnl_gross_dollars: float        # before commission
    pnl_points: float               # raw price-point P&L
    r_multiple: float
    exit_type: str                  # "SL_HIT", "TP_HIT", "PARTIAL_EXIT", "ATR_TRAIL",
                                    # "TIME_EXIT", "FORCED_CLOSE"
    stop_price: float               # stop level at time of close
    commission: float               # total commission for this trade
    # Optional / metadata
    ticket: str = ""
    tp_price: Optional[float] = None
    regime_at_entry: str = ""
    regime_at_exit: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON export."""
        return {
            "ticket": self.ticket,
            "strategy": self.strategy,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "lot_size": self.lot_size,
            "pnl_net_dollars": self.pnl_net_dollars,
            "pnl_gross_dollars": self.pnl_gross_dollars,
            "pnl_points": self.pnl_points,
            "r_multiple": self.r_multiple,
            "exit_type": self.exit_type,
            "stop_price": self.stop_price,
            "tp_price": self.tp_price,
            "commission": self.commission,
            "regime_at_entry": self.regime_at_entry,
            "regime_at_exit": self.regime_at_exit,
        }


@dataclass
class EquityPoint:
    """Single point on the equity curve."""
    time: datetime
    equity: float
    balance: float = 0.0
    open_positions: int = 0
    regime: str = "NO_TRADE"        # FIX: was commented out; engine passes regime= and to_dict() reads self.regime
    drawdown_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "time": str(self.time),
            "equity": self.equity,
            "balance": self.balance,
            "open_positions": self.open_positions,
            "regime": self.regime,
            "drawdown_pct": self.drawdown_pct,
        }


@dataclass
class SimulatedState:
    """
    ENHANCED backtest state matching live system state.py exactly.

    Tracks all strategy state, risk management, and position data.
    Mirrors the live system's state structure for perfect parity.

    BUG-4 FIX: to_dict() now exposes 'regime' and 'session' as aliases for
        current_regime and current_session so all strategy state.get()
        calls work regardless of which key name they use.

    BUG-5 FIX: size_multiplier initialised to 0.5 (RANGING floor) instead of
        0.0 to prevent zero-multiplier lot sizes on bar-0.

    REMAINING-6 FIX: removed stale duplicate `regime` field; only
        current_regime is canonical.  to_dict() already exposes 'regime'
        as an alias.
    """
    # Account metrics
    balance: float = 10000.0
    equity: float = 10000.0
    peak_equity: float = 10000.0

    # Regime state (from live regime_engine.py)
    current_regime: str = "NO_TRADE"
    size_multiplier: float = 0.5
    consecutive_regime_readings: int = 0
    pending_regime_state: Optional[str] = None
    last_adx_h4: float = 0.0
    last_atr_pct_h1: float = 0.0
    last_atr_h1_raw: float = 0.0
    last_atr_m15: float = 0.0
    last_di_plus_h4: Optional[float] = None
    last_di_minus_h4: Optional[float] = None
    current_session: str = "OFF_HOURS"

    # S1 family state
    s1_family_attempts_today: int = 0
    s1f_attempts_today: int = 0
    s1b_pending_ticket: Optional[int] = None
    s1d_ema_touched_today: bool = False
    s1d_fired_today: bool = False
    s1e_pyramid_done: bool = False
    s1f_post_tk_active: bool = False
    s3_sweep_candle_time: Optional[datetime] = None
    s3_sweep_low: float = 0.0
    s3_fired_today: bool = False
    s3_sweep_direction: Optional[str] = None
    s4_fired_today: bool = False
    s4_ema_touched: bool = False
    s5_fired_today: bool = False
    s6_placed_today: bool = False
    s7_placed_today: bool = False
    s8_fired_today: bool = False
    s8_armed: bool = False
    s8_arm_time: Optional[datetime] = None
    s8_spike_high: float = 0.0
    s8_spike_low: float = 0.0
    s8_spike_direction: Optional[str] = None
    s8_confirmation_passed: bool = False
    s2_fired_today: bool = False

    # S1d / S1e pyramid tracking
    s1d_pyramid_count: int = 0
    s1e_pyramid_count: int = 0

    # S1f post-time-kill re-entry tracking
    s1f_reentered_today: bool = False
    s1f_killed_position_profitable: bool = False
    s1f_original_size: float = 0.0

    # R3 state (independent lane)
    r3_armed: bool = False
    r3_arm_time: Optional[datetime] = None
    r3_direction: Optional[str] = None
    r3_fired_today: bool = False
    ks7_pre_event_price: float = 0.0

    # Position state
    trend_family_occupied: bool = False
    trend_family_strategy: Optional[str] = None
    trend_trade_direction: Optional[str] = None
    reversal_family_occupied: bool = False
    open_position: Optional[int] = None
    entry_price: float = 0.0
    stop_price_original: float = 0.0
    stop_price_current: float = 0.0
    original_lot_size: float = 0.0
    open_trade_id: Optional[str] = None
    open_campaign_id: Optional[str] = None
    last_s1_direction: Optional[str] = None
    last_s1_max_r: float = 0.0
    position_partial_done: bool = False
    position_be_activated: bool = False
    position_pyramid_done: bool = False
    position_m5_count: int = 0

    # Independent position lanes (R3, S8)
    s8_open_ticket: Optional[int] = None
    s8_entry_price: float = 0.0
    s8_stop_price_original: float = 0.0
    s8_stop_price_current: float = 0.0
    s8_trade_direction: Optional[str] = None
    s8_be_activated: bool = False
    s8_open_time_utc: Optional[str] = None

    r3_open_ticket: Optional[int] = None
    r3_open_time: Optional[datetime] = None
    r3_entry_price: float = 0.0
    r3_stop_price: float = 0.0
    r3_tp_price: float = 0.0

    # Pending orders
    s1_pending_buy_ticket: Optional[int] = None
    s1_pending_sell_ticket: Optional[int] = None
    s6_pending_buy_ticket: Optional[int] = None
    s6_pending_sell_ticket: Optional[int] = None
    s7_pending_buy_ticket: Optional[int] = None
    s7_pending_sell_ticket: Optional[int] = None

    # Risk management state
    trading_enabled: bool = True
    shutdown_reason: Optional[str] = None
    ks4_reduced_trades_remaining: int = 0
    failed_breakout_flag: bool = False
    failed_breakout_direction: Optional[str] = None
    stop_hunt_detected: bool = False
    ks7_active: bool = False
    ks7_pre_event_atr: float = 0.0

    # Daily / weekly tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    daily_trades: int = 0
    daily_commission_paid: float = 0.0
    consecutive_m5_losses: int = 0
    consecutive_losses: int = 0

    # Range and structural data
    range_high: float = 0.0
    range_low: float = 0.0
    range_size: float = 0.0
    range_computed: bool = False
    s7_prev_day_high: float = 0.0
    s7_prev_day_low: float = 0.0

    # Performance tracking
    total_closed_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl_gross: float = 0.0
    total_commission: float = 0.0

    # Portfolio correlation throttle (CRIT-3)
    corr_throttle_active: bool = False
    corr_throttle_pairs: list = field(default_factory=list)
    _corr_check_trade_counter: int = field(default=0, repr=False)

    # Extra structural state (set dynamically by engine)
    open_positions: dict = field(default_factory=dict)
    independent_lanes_occupied: dict = field(default_factory=dict)
    pre_london_range: Optional[dict] = None
    asian_range: Optional[dict] = None
    prev_day_ohlc: Optional[dict] = None
    r3_pre_event_price: float = 0.0
    r3_pre_event_atr: float = 0.0
    conviction_level: str = "STANDARD"
    severity_multiplier: float = 1.0
    vol_scalar: float = 1.0

    def to_dict(self) -> dict:
        """
        Convert to dict for strategy evaluation functions.

        BUG-4/13 FIX: Exposes 'regime' and 'session' as aliases for
        current_regime and current_session so all strategy state.get()
        calls work regardless of which key name they use.
        """
        return {
            # Account
            "balance": self.balance,
            "equity": self.equity,
            "peak_equity": self.peak_equity,

            # Regime — expose BOTH canonical and alias names
            "current_regime": self.current_regime,
            "regime": self.current_regime,          # BUG-4 alias
            "size_multiplier": self.size_multiplier,
            "consecutive_regime_readings": self.consecutive_regime_readings,
            "pending_regime_state": self.pending_regime_state,
            "last_adx_h4": self.last_adx_h4,
            "last_atr_pct_h1": self.last_atr_pct_h1,
            "last_atr_h1_raw": self.last_atr_h1_raw,
            "last_atr_m15": self.last_atr_m15,
            "last_di_plus_h4": self.last_di_plus_h4,
            "last_di_minus_h4": self.last_di_minus_h4,
            "current_session": self.current_session,
            "session": self.current_session,        # BUG-13 alias

            # S1 family
            "s1_family_attempts_today": self.s1_family_attempts_today,
            "s1f_attempts_today": self.s1f_attempts_today,
            "s1_attempts_today": self.s1_family_attempts_today,   # legacy alias
            "s1b_pending_ticket": self.s1b_pending_ticket,
            "s1d_ema_touched_today": self.s1d_ema_touched_today,
            "s1d_fired_today": self.s1d_fired_today,
            "s1e_pyramid_done": self.s1e_pyramid_done,
            "s1f_post_tk_active": self.s1f_post_tk_active,
            "s1b_fired_today": self.s1d_fired_today,              # legacy alias
            "s3_sweep_candle_time": self.s3_sweep_candle_time,
            "s3_sweep_low": self.s3_sweep_low,
            "s3_fired_today": self.s3_fired_today,
            "s3_sweep_direction": self.s3_sweep_direction,
            "s4_fired_today": self.s4_fired_today,
            "s4_ema_touched": self.s4_ema_touched,
            "s5_fired_today": self.s5_fired_today,
            "s5_compression_confirmed": self.range_computed,      # reused flag
            "s6_placed_today": self.s6_placed_today,
            "s7_placed_today": self.s7_placed_today,
            "s8_fired_today": self.s8_fired_today,
            "s8_armed": self.s8_armed,
            "s8_arm_time": self.s8_arm_time,
            "s8_spike_high": self.s8_spike_high,
            "s8_spike_low": self.s8_spike_low,
            "s8_spike_direction": self.s8_spike_direction,
            "s8_confirmation_passed": self.s8_confirmation_passed,
            "s2_fired_today": self.s2_fired_today,
            "r3_fired_today": self.r3_fired_today,
            "s1d_pyramid_count": self.s1d_pyramid_count,
            "s1e_pyramid_count": self.s1e_pyramid_count,
            "s1f_reentered_today": self.s1f_reentered_today,
            "s1f_killed_position_profitable": self.s1f_killed_position_profitable,
            "s1f_original_size": self.s1f_original_size,

            # R3
            "r3_armed": self.r3_armed,
            "r3_arm_time": self.r3_arm_time,
            "r3_direction": self.r3_direction,
            "ks7_pre_event_price": self.ks7_pre_event_price,

            # Position state
            "trend_family_occupied": self.trend_family_occupied,
            "trend_family_strategy": self.trend_family_strategy,
            "trend_trade_direction": self.trend_trade_direction,
            "reversal_family_occupied": self.reversal_family_occupied,
            "open_position": self.open_position,
            "entry_price": self.entry_price,
            "stop_price_original": self.stop_price_original,
            "stop_price_current": self.stop_price_current,
            "original_lot_size": self.original_lot_size,
            "open_trade_id": self.open_trade_id,
            "open_campaign_id": self.open_campaign_id,
            "last_s1_direction": self.last_s1_direction,
            "last_s1_max_r": self.last_s1_max_r,
            "position_partial_done": self.position_partial_done,
            "position_be_activated": self.position_be_activated,
            "position_pyramid_done": self.position_pyramid_done,
            "position_m5_count": self.position_m5_count,

            # Independent lanes
            "s8_open_ticket": self.s8_open_ticket,
            "s8_entry_price": self.s8_entry_price,
            "s8_stop_price_original": self.s8_stop_price_original,
            "s8_stop_price_current": self.s8_stop_price_current,
            "s8_trade_direction": self.s8_trade_direction,
            "s8_be_activated": self.s8_be_activated,
            "s8_open_time_utc": self.s8_open_time_utc,
            "r3_open_ticket": self.r3_open_ticket,
            "r3_open_time": self.r3_open_time,
            "r3_entry_price": self.r3_entry_price,
            "r3_stop_price": self.r3_stop_price,
            "r3_tp_price": self.r3_tp_price,

            # Pending orders
            "s1_pending_buy_ticket": self.s1_pending_buy_ticket,
            "s1_pending_sell_ticket": self.s1_pending_sell_ticket,
            "s6_pending_buy_ticket": self.s6_pending_buy_ticket,
            "s6_pending_sell_ticket": self.s6_pending_sell_ticket,
            "s7_pending_buy_ticket": self.s7_pending_buy_ticket,
            "s7_pending_sell_ticket": self.s7_pending_sell_ticket,

            # Risk management
            "trading_enabled": self.trading_enabled,
            "shutdown_reason": self.shutdown_reason,
            "ks4_reduced_trades_remaining": self.ks4_reduced_trades_remaining,
            "failed_breakout_flag": self.failed_breakout_flag,
            "failed_breakout_direction": self.failed_breakout_direction,
            "stop_hunt_detected": self.stop_hunt_detected,
            "ks7_active": self.ks7_active,
            "ks7_pre_event_atr": self.ks7_pre_event_atr,

            # Daily tracking
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "daily_trades": self.daily_trades,
            "daily_commission_paid": self.daily_commission_paid,
            "consecutive_m5_losses": self.consecutive_m5_losses,
            "consecutive_losses": self.consecutive_losses,

            # Range data
            "range_high": self.range_high,
            "range_low": self.range_low,
            "range_size": self.range_size,
            "range_computed": self.range_computed,
            "s7_prev_day_high": self.s7_prev_day_high,
            "s7_prev_day_low": self.s7_prev_day_low,

            # Performance
            "total_closed_trades": self.total_closed_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_pnl_gross": self.total_pnl_gross,
            "total_commission": self.total_commission,

            # Correlation kill (CRIT-3)
            "corr_throttle_active": self.corr_throttle_active,
            "corr_throttle_pairs": self.corr_throttle_pairs,
        }
