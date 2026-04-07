"""
backtest/engine_updated.py — Complete Backtest Engine Rebuild

Fully rebuilt backtest engine to match current live system:
- Updated regime engine with 6 states and hysteresis
- All 10 strategies with current logic and parameters
- Updated risk management with current thresholds
- Enhanced execution simulator with ATR-based stops
- Portfolio risk controls and correlation management
- Independent lane tracking (R3, S8)
- Phase-based progression and conviction levels
- Economic calendar integration with event filtering
- Multi-timeframe data processing (M5, M15, H1, H4)

Matches live system exactly for accurate backtesting results.
"""
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

import pytz

# Import updated components
from backtest.indicators import calculate_atr, calculate_rsi, calculate_ema, calculate_adx_h4
from backtest.data_feed import (
    HistoricalDataFeed, HistoricalEventFeed, HistoricalSpreadFeed, BarBuffer, HistoricalDXYFeed
)
from backtest.strategies import ALL_STRATEGIES, STRATEGY_CONFIGS, get_strategy_family
from backtest.strategies import evaluate_strategy
from backtest.regime_engine import RegimeClassifier, RegimeState, get_atr_percentile_h1
from backtest.risk.kill_switches import KillSwitchChecker, check_ks6_drawdown
from backtest.ks6_auto_reset import should_enable_ks6_auto_reset
from backtest.risk.position_sizing import PositionSizer, PortfolioRiskChecker
from backtest.execution_simulator import ExecutionSimulator
from backtest.models import SimulatedState, SimOrder, SimPosition, TradeRecord, EquityPoint
from backtest.ks6_auto_reset import (
    handle_ks6_trigger_backtest, check_ks6_cooldown, add_ks6_to_final_results
)

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger("backtest.engine")

# Trading constants for XAUUSD
POINT_VALUE = 100.0  # XAUUSD: $1 price move × 100 oz/lot = $100 per standard lot
COMMISSION_PER_LOT_RT = 7.00  # USD round-trip per standard lot
SWAP_PER_LOT_PER_NIGHT = -7.0  # USD per standard lot per overnight hold (XAUUSD long typical)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Complete backtest engine matching live system.
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        slippage_points: float = 0.7,
        strategies: Optional[List[str]] = None
    ):
        self.start_date = start_date.replace(tzinfo=pytz.utc)
        self.end_date = end_date.replace(tzinfo=pytz.utc)
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.strategies = strategies or ALL_STRATEGIES
        
        # Initialize components
        self.data_feed = HistoricalDataFeed(start_date, end_date)
        self.event_feed = HistoricalEventFeed(start_date, end_date)
        self.spread_feed = HistoricalSpreadFeed(start_date, end_date)
        self.dxy_feed = HistoricalDXYFeed(start_date, end_date)
        self.bar_buffer = BarBuffer()
        
        self.regime_classifier = RegimeClassifier()
        # Backtest event-blackout (KS7) control:
        # Only run KS7 when a real events CSV was loaded; hardcoded events
        # have no accurate timestamps and flood bars with false blackouts.
        self._ks7_enabled: bool = False          # set True after data load if CSV found
        self._last_ks7_log_time = None           # rate-limit warning spam
        self.kill_switch_checker = KillSwitchChecker()
        self.position_sizer = PositionSizer()
        self.portfolio_checker = PortfolioRiskChecker()
        self.execution_sim = ExecutionSimulator(slippage_points)
        
        # State tracking
        self.state: SimulatedState = SimulatedState()
        self.pending_orders: List[SimOrder] = []
        self.equity_curve: List[EquityPoint] = []
        self.trades: List[TradeRecord] = []
        
        # Performance tracking
        self.total_bars = 0
        self.processed_bars = 0

    def run(self) -> Dict[str, Any]:
        """
        Run complete backtest.
        """
        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Strategies: {', '.join(self.strategies)}")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")
        
        # Load data
        try:
            m5_data = self.data_feed.load()
            events = self.event_feed.load()
            spreads = self.spread_feed.load()

            # Load warm-up bars (100 H1 bars = ~4 days before start_date)
            warmup_start = self.start_date - timedelta(days=5)
            warmup_feed = HistoricalDataFeed(warmup_start, self.start_date)
            warmup_data = None
            try:
                warmup_data = warmup_feed.load()
                # Pre-populate bar buffer with warm-up data (don't generate signals)
                for bar in warmup_data.itertuples():
                    self.bar_buffer.add_m5_bar(bar._asdict())
                logger.info(f"Warm-up complete: {len(warmup_data)} bars pre-loaded")
            except Exception as e:
                logger.warning(f"Could not load warm-up data: {e}")

            # Enable KS7 only when a real events CSV was loaded.
            # Hardcoded events are not timestamp-accurate enough for bar-level blackouts.
            self._ks7_enabled = getattr(self.event_feed, 'loaded_from_csv', False)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {'error': str(e)}
        
        self.total_bars = len(m5_data)
        logger.info(f"Processing {self.total_bars} M5 bars")
        
        # Initialize state — must happen AFTER warm-up buffer is populated
        self._initialize_state()

        # ── WARM-UP RANGE BACKFILL ────────────────────────────────────────────
        # pre_london_range / asian_range / prev_day_ohlc start as None.
        # They are only updated inside _update_strategy_state() when the live
        # bar clock hits exact timestamps (07:55, 05:30, 00:00).  On a run
        # that starts mid-day those timestamps are never reached before the
        # first strategy evaluation, so S1/S6/S7/S4 always return None on
        # day-1 (and often day-2).  Fix: derive these ranges directly from the
        # warm-up bars that are already in the bar buffer.
        if warmup_data is not None and not warmup_data.empty:
            self._backfill_ranges_from_warmup(warmup_data)
        # ─────────────────────────────────────────────────────────────────────
        
        # Main processing loop
        start_time = time.time()
        
        for i, bar in enumerate(m5_data.itertuples(), 1):
            try:
                # Check KS6 cooldown first (if auto-reset enabled)
                if should_enable_ks6_auto_reset():
                    if check_ks6_cooldown(self.state.__dict__, i):
                        self.processed_bars = i
                        continue
                
                self._process_bar(bar, events, spreads)
                self.processed_bars = i
                
                # Progress logging
                if i % 1000 == 0:
                    progress_pct = (i / self.total_bars) * 100
                    eta_seconds = ((self.total_bars - i) * (time.time() - start_time)) / max(i, 1)
                    eta_minutes = eta_seconds / 60
                    logger.info(
                        f"Progress {i}/{self.total_bars} ({progress_pct:.1f}%) "
                        f"bar={bar.time} equity=${self.state.equity:,.2f} ETA={eta_minutes:.1f}m"
                    )
                
            except Exception as e:
                logger.error(f"Error processing bar {i}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # Finalize
        self._finalize_backtest()
        
        return self._generate_results()

    def _initialize_state(self) -> None:
        """Initialize backtest state."""
        self.state.balance = self.initial_balance
        self.state.equity = self.initial_balance
        self.state.peak_equity = self.initial_balance
        self.state.daily_pnl = 0.0
        self.state.weekly_pnl = 0.0
        self.state.trading_enabled = True
        self.state.consecutive_losses = 0
        self.state.ks4_reduced_trades_remaining = 0
        
        # Strategy-specific state
        self.state.s1_family_attempts_today = 0
        self.state.s1d_fired_today = False
        self.state.s1d_reentries_today = 0  # legacy alias — kept for safety
        self.state.s1f_reentered_today = False
        self.state.s4_fired_today = False
        self.state.s4_ema_touched = False
        self.state.s5_fired_today = False
        self.state.range_computed = False
        self.state.r3_fired_today = False
        self.state.s8_fired_today = False
        
        # Position tracking
        self.state.open_positions = {}
        self.state.trend_family_occupied = False
        self.state.reversal_family_occupied = False
        self.state.independent_lanes_occupied = {}

        self.kill_switch_checker.reset_daily(self.initial_balance)
        self.kill_switch_checker.reset_weekly(self.initial_balance)

    def _backfill_ranges_from_warmup(self, warmup_data: pd.DataFrame) -> None:
        """
        Derive pre_london_range, asian_range, and prev_day_ohlc from the
        warm-up DataFrame so strategies have valid state from bar-1 of the
        live window.

        Called once, after _initialize_state() and after the bar buffer has
        been populated with warm-up bars.
        """
        try:
            # Ensure times are tz-aware UTC
            wdf = warmup_data.copy()
            if wdf['time'].dt.tz is None:
                wdf['time'] = wdf['time'].dt.tz_localize(pytz.utc)

            # ── prev_day_ohlc ────────────────────────────────────────────────
            # Use the last full calendar day present in the warm-up window.
            wdf['_date'] = wdf['time'].dt.date
            dates = sorted(wdf['_date'].unique())
            if len(dates) >= 1:
                last_day = dates[-1]
                day_bars = wdf[wdf['_date'] == last_day]
                self.state.prev_day_ohlc = {
                    'high':  float(day_bars['high'].max()),
                    'low':   float(day_bars['low'].min()),
                    'open':  float(day_bars.iloc[0]['open']),
                    'close': float(day_bars.iloc[-1]['close']),
                }
                logger.info(
                    f"Backfill prev_day_ohlc from warm-up: "
                    f"H={self.state.prev_day_ohlc['high']:.2f} "
                    f"L={self.state.prev_day_ohlc['low']:.2f}"
                )

            # ── asian_range ──────────────────────────────────────────────────
            # Asian session = 22:00–07:00 UTC.  Use the last Asian window in
            # the warm-up data (bars where hour is 22-23 OR 0-6).
            asian_mask = (wdf['time'].dt.hour >= 22) | (wdf['time'].dt.hour < 7)
            asian_bars = wdf[asian_mask]
            if not asian_bars.empty:
                self.state.asian_range = {
                    'high': float(asian_bars['high'].max()),
                    'low':  float(asian_bars['low'].min()),
                }
                logger.info(
                    f"Backfill asian_range from warm-up: "
                    f"H={self.state.asian_range['high']:.2f} "
                    f"L={self.state.asian_range['low']:.2f}"
                )

            # ── pre_london_range ─────────────────────────────────────────────
            # Pre-London range = Asian session bars of the LAST warm-up day
            # (hour 0-6 UTC on the final date, which is the day before live
            # trading starts).
            if len(dates) >= 1:
                last_date = dates[-1]
                pre_london_mask = (
                    (wdf['_date'] == last_date) &
                    (wdf['time'].dt.hour < 7)
                )
                pre_london_bars = wdf[pre_london_mask]
                # Fallback: if the last warm-up day has no pre-07:00 bars
                # (e.g. data starts at 01:00), extend to the previous date.
                if pre_london_bars.empty and len(dates) >= 2:
                    prev_date = dates[-2]
                    pre_london_mask = (
                        (wdf['_date'] == prev_date) &
                        (wdf['time'].dt.hour < 7)
                    )
                    pre_london_bars = wdf[pre_london_mask]

                if not pre_london_bars.empty:
                    self.state.pre_london_range = {
                        'high': float(pre_london_bars['high'].max()),
                        'low':  float(pre_london_bars['low'].min()),
                    }
                    logger.info(
                        f"Backfill pre_london_range from warm-up: "
                        f"H={self.state.pre_london_range['high']:.2f} "
                        f"L={self.state.pre_london_range['low']:.2f}"
                    )
                else:
                    # Last resort: use the full Asian range as pre-London range
                    self.state.pre_london_range = self.state.asian_range
                    logger.info("Backfill pre_london_range: used asian_range as fallback")

        except Exception as exc:
            logger.warning(f"_backfill_ranges_from_warmup failed (non-fatal): {exc}")
            # Leave fields as None — strategies will skip gracefully

    def _process_bar(self, bar, events: List[Dict[str, Any]], spreads: pd.DataFrame) -> None:
        """
        Process single M5 bar.
        """
        # Extract bar_time first; add to buffer
        bar_time = bar.time
        self.bar_buffer.add_m5_bar(bar._asdict())

        # Apply overnight swap at midnight
        if bar_time.hour == 0 and bar_time.minute == 0:
            self._apply_overnight_swap(bar_time)

        session = self._get_session(bar_time)
        self.state.current_session = session  # BUG-4: keep SimulatedState in sync

        # Get current spread
        current_spread = self._get_current_spread(bar_time, spreads)
        median_spread  = self.spread_feed.get_24h_median_spread(bar_time)
        
        # Update state with current data
        self.state.current_time  = bar_time
        self.state.current_price = bar.close
        self.state.current_spread = current_spread
        self.state.median_spread_24h = median_spread
        
        # Calculate indicators
        context = self._calculate_indicators(bar_time)
        
        # Update regime
        upcoming_events = self.event_feed.get_upcoming_events(bar_time, minutes_ahead=60)
        has_upcoming_event = len(upcoming_events) > 0
        
        # KS7: Event Blackout
        if self._ks7_enabled and has_upcoming_event:
            if self._last_ks7_log_time is None or (bar_time - self._last_ks7_log_time).total_seconds() > 4500:
                logger.info(f"KS7: Event blackout active at {bar_time}")
                self._last_ks7_log_time = bar_time
            self.state.trading_enabled = False
        
        # BUG-8 FIX: correct classify_regime call signature — dxy_macro is 4th arg
        dxy_macro = self.dxy_feed.get_daily_change(bar_time)  # Returns None if no data
        regime, size_multiplier = self.regime_classifier.classify_regime(
            context.get('adx_h4', 20.0),
            context.get('atr_pct_h1', 50.0),
            session,
            dxy_macro,           # dxy_macro — retrieved from feed
            has_upcoming_event,
            current_spread / median_spread if median_spread > 0 else 1.0
        )
        
        self.state.current_regime = str(regime.value) if hasattr(regime, 'value') else str(regime)
        self.state.size_multiplier = size_multiplier
        
        # Run kill switches — pass empty events list when KS7 is disabled
        # (hardcoded events have no accurate timestamps; CSV not present)
        ks_events = upcoming_events if self._ks7_enabled else []
        ks_results = self.kill_switch_checker.check_all_kill_switches(
            self.state.to_dict(), bar_time, ks_events
        )
        
        # Update trading enabled status
        self.state.trading_enabled = ks_results['trading_enabled']
        
        if ks_results['triggered_switches']:
            logger.warning(f"Kill switches triggered: {ks_results['triggered_switches']}")
        
        # Process existing positions — BUG-6 FIX: pass bar as dict, not namedtuple
        closed_trades = self.execution_sim.manage_open_positions(
            bar._asdict(), context.get('atr_m15', 15.0), self.state.to_dict()
        )
        
        for trade in closed_trades:
            self.trades.append(trade)
            self._update_state_after_trade(trade)
        
        # Generate new signals
        if self.state.trading_enabled:
            # BUG-4 FIX: pass state.to_dict() so strategies get 'regime' and 'session' aliases
            new_orders = self._generate_signals(bar, context, upcoming_events)
            if not self.state.s4_ema_touched:
                ema20_m15 = context.get('ema20_m15')
                bar_dict = bar._asdict()
                if ema20_m15 and bar_dict['low'] <= ema20_m15 <= bar_dict['high']:
                    self.state.s4_ema_touched = True

            
            # Portfolio risk check
            filtered_orders = []
            for order in new_orders:
                portfolio_result = self.portfolio_checker.check_portfolio_risk(
                    order.__dict__, self.state.__dict__
                )
                if portfolio_result[0]:  # Permitted
                    filtered_orders.append(order)
                else:
                    logger.debug(f"Portfolio risk blocked: {order.strategy} - {portfolio_result[1]}")
            
            # Position sizing
            valid_orders = []
            for order in filtered_orders:
                lot_size, sizing_details = self.position_sizer.calculate_lot_size(
                    abs(order.stop_price - order.price),
                    self.state.size_multiplier,
                    self.state.__dict__,
                    conviction_level=getattr(self.state, 'conviction_level', 'STANDARD'),
                    severity_multiplier=getattr(self.state, 'severity_multiplier', 1.0),
                    spread_multiplier=current_spread / median_spread if median_spread > 0 else 1.0,
                    vol_scalar=getattr(self.state, 'vol_scalar', 1.0)
                )
                
                # Check if position sizing blocked this order
                if lot_size > 0:
                    order.lot_size = lot_size
                    order.metadata.update(sizing_details)
                    valid_orders.append(order)
                else:
                    logger.debug(f"Position sizing blocked: {order.strategy} - {sizing_details.get('reason', 'unknown')}")
            
            # Process orders - combine new valid orders with any pending orders from previous bars
            all_orders = self.pending_orders + valid_orders
            filled_positions, remaining_orders = self.execution_sim.process_pending_orders(
                all_orders, bar._asdict(), current_spread, bar_time
            )
            
            self.pending_orders = remaining_orders
            
            # Update position tracking
            for position in filled_positions:
                self._update_state_after_fill(position)
        
        # Update equity curve
        self._update_equity_curve(bar_time)

    def _calculate_indicators(self, bar_time: datetime) -> Dict[str, Any]:
        """
        Calculate all required indicators.
        BUG-7 FIX: every key is seeded with a safe default so downstream
        code never receives None.
        """
        # Safe defaults (BUG-7)
        context: Dict[str, Any] = {
            'ema20_m5': None,
            'ema20_m15': None,
            'ema20_h1': None,
            'atr_m15': 15.0,
            'atr_h1': 20.0,
            'rsi_h1': 50.0,
            'atr_pct_h1': 50.0,
            'adx_h4': 20.0,
            'di_plus': 0.0,
            'di_minus': 0.0,
            'di_plus_h4': 0.0,   # BUG-14 alias used by S4
            'di_minus_h4': 0.0,  # BUG-14 alias used by S4
            'adx_increasing': False,
        }
        
        m5_df  = self.bar_buffer.get_dataframe('M5',  100)
        m15_df = self.bar_buffer.get_dataframe('M15',  50)
        h1_df  = self.bar_buffer.get_dataframe('H1',  100)
        h4_df  = self.bar_buffer.get_dataframe('H4',  100)

        if not m5_df.empty:
            context['ema20_m5'] = calculate_ema(m5_df, 20).iloc[-1] if len(m5_df) >= 20 else None

        if not m15_df.empty and len(m15_df) >= 14:
            # BUG-7 FIX: only overwrite the default if data is sufficient
            atr_m15_series = calculate_atr(m15_df, 14)
            val = atr_m15_series.iloc[-1]
            if val is not None and not (isinstance(val, float) and val != val):  # not NaN
                context['atr_m15'] = float(val)
            context['ema20_m15'] = calculate_ema(m15_df, 20).iloc[-1] if len(m15_df) >= 20 else None  # BUG-15

        if not h1_df.empty and len(h1_df) >= 14:
            context['ema20_h1'] = calculate_ema(h1_df, 20).iloc[-1] if len(h1_df) >= 20 else None
            atr_h1_series = calculate_atr(h1_df, 14)
            val = atr_h1_series.iloc[-1]
            if val is not None and not (isinstance(val, float) and val != val):
                context['atr_h1'] = float(val)
            rsi_val = calculate_rsi(h1_df, 14).iloc[-1]
            if rsi_val is not None and not (isinstance(rsi_val, float) and rsi_val != rsi_val):
                context['rsi_h1'] = float(rsi_val)
            context['atr_pct_h1'] = get_atr_percentile_h1(atr_h1_series, context['atr_h1'])

        if not h4_df.empty:
            adx_result = calculate_adx_h4(h4_df, 14)
            if adx_result:
                context['adx_h4']      = adx_result['adx']
                context['di_plus']     = adx_result['di_plus']
                context['di_minus']    = adx_result['di_minus']
                context['di_plus_h4']  = adx_result['di_plus']   # BUG-14 alias for S4
                context['di_minus_h4'] = adx_result['di_minus']  # BUG-14 alias for S4
                context['adx_increasing'] = adx_result['increasing']
        
        # Store recent bars for strategies
        context['recent_m5_bars']  = self.bar_buffer.get_latest_bars('M5',  10)
        context['recent_m15_bars'] = self.bar_buffer.get_latest_bars('M15', 20)
        context['london_bars']     = self.bar_buffer.get_latest_bars('M5',  48)
        return context

    def _get_session(self, bar_time: datetime) -> str:
        """Get trading session for bar time."""
        hour = bar_time.hour
        
        if 22 <= hour or hour < 7:
            return "ASIAN"
        elif 7 <= hour < 13:
            return "LONDON"
        elif 13 <= hour < 17:
            return "LONDON_NY_OVERLAP"
        elif 17 <= hour < 21:
            return "NY"
        else:
            return "OFF_HOURS"

    def _get_current_spread(self, bar_time: datetime, spreads: pd.DataFrame) -> float:
        """Get current spread from spread data (BUG-12: handles tz-naive spread times)."""
        if spreads is None or spreads.empty:
            return 2.0

        # Ensure spread times are tz-aware UTC to match bar_time
        if spreads['time'].dt.tz is None:
            spreads = spreads.copy()
            import pytz
            spreads['time'] = spreads['time'].dt.tz_localize(pytz.utc)

        closest_idx = (spreads['time'] - bar_time).abs().idxmin()
        return float(spreads.loc[closest_idx, 'spread'])

    def _generate_signals(
        self,
        bar,
        context: Dict[str, Any],
        upcoming_events: List[Dict[str, Any]]
    ) -> List[SimOrder]:
        """Generate trading signals for enabled strategies."""
        orders = []

        # Update strategy-specific state
        self._update_strategy_state(bar, context, upcoming_events)

        # BUG-4 FIX: use to_dict() so strategies get 'regime' / 'session' aliases
        state_dict = self.state.to_dict()

        for strategy in self.strategies:
            try:
                signal = evaluate_strategy(strategy, state_dict, bar._asdict(), context)

                if signal:
                    if isinstance(signal, list):
                        for sig in signal:
                            orders.append(self._create_order(sig, bar.time))
                    else:
                        orders.append(self._create_order(signal, bar.time))

                    # Mark one-shot strategies as placed for today
                    if strategy == 'S6_ASIAN_BRK':
                        self.state.s6_placed_today = True
                    elif strategy == 'S7_DAILY_STRUCT':
                        self.state.s7_placed_today = True

            except Exception as e:
                logger.error(f"Error evaluating {strategy}: {e}")
                continue

        return orders

    def _create_order(self, signal: Dict[str, Any], bar_time: datetime) -> SimOrder:
        """Create SimOrder from signal."""
        return SimOrder(
            strategy=signal['strategy'],
            direction=signal['direction'],
            order_type=signal['entry_type'],
            price=signal['entry_price'],
            stop_price=signal['stop_price'],
            tp_price=signal.get('tp_price'),
            lot_size=signal.get('lot_size', 0.0),
            expiry=signal.get('expiry'),
            tag=signal.get('tag'),
            linked_tag=signal.get('linked_tag'),
            placed_time=bar_time,
            metadata=signal.get('metadata', {})
        )

    def _update_strategy_state(
        self,
        bar,
        context: Dict[str, Any],
        upcoming_events: List[Dict[str, Any]]
    ) -> None:
        """Update strategy-specific state variables."""
        bar_time = bar.time
        session = self._get_session(bar_time)
        
        # Update session-based state
        if session == "LONDON":
            # Calculate pre-London range at 07:55 UTC
            if bar_time.hour == 7 and bar_time.minute == 55:
                # Need bars from previous Asian session (which is up to ~8 hours ago = 96 bars)
                m5_bars = self.bar_buffer.get_latest_bars('M5', 96)
                asian_bars = [b for b in m5_bars if b['time'].hour < 7]
                if asian_bars:
                    asian_high = max(b['high'] for b in asian_bars)
                    asian_low = min(b['low'] for b in asian_bars)
                    self.state.pre_london_range = {'high': asian_high, 'low': asian_low}
        
        elif session == "ASIAN":
            # Update Asian range
            if bar_time.hour == 5 and bar_time.minute == 30:
                m5_bars = self.bar_buffer.get_latest_bars('M5', 72)
                asian_bars = [b for b in m5_bars if b['time'].hour >= 22 or b['time'].hour < 7]
                if asian_bars:
                    asian_high = max(b['high'] for b in asian_bars)
                    asian_low = min(b['low'] for b in asian_bars)
                    self.state.asian_range = {'high': asian_high, 'low': asian_low}
        
        # Update daily state at midnight
        if bar_time.hour == 0 and bar_time.minute == 0:
            # Previous day OHLC
            m15_bars = self.bar_buffer.get_latest_bars('M15', 96)
            if len(m15_bars) >= 96:  # 24 hours of M15
                prev_high = max(b['high'] for b in m15_bars)
                prev_low = min(b['low'] for b in m15_bars)
                self.state.prev_day_ohlc = {
                    'high': prev_high, 'low': prev_low,
                    'open': m15_bars[0]['open'],
                    'close': m15_bars[-1]['close']
                }

            # KS6: Drawdown circuit breaker (checked daily at midnight)
            ks6_triggered, ks6_reason = check_ks6_drawdown(
                self.state.equity, self.state.peak_equity, self.state.to_dict()
            )
            if ks6_triggered:
                if should_enable_ks6_auto_reset() and "AUTO_RESET" in ks6_reason:
                    # BUG-10 FIX: capture returned dict before reconstructing state
                    updated_state_dict = handle_ks6_trigger_backtest(
                        self.state.to_dict(),
                        self.processed_bars,
                        bar._asdict(),
                        (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
                    )
                    # Rebuild SimulatedState from the *returned* dict, not self.state
                    new_state = SimulatedState()
                    for k, v in updated_state_dict.items():
                        if hasattr(new_state, k):
                            setattr(new_state, k, v)
                    self.state = new_state
                    self.execution_sim.open_positions.clear()
                    self.pending_orders.clear()
                else:
                    self.state.trading_enabled = False
                    self.state.shutdown_reason = ks6_reason

            # Reset daily counters and kill-switch baselines
            self.state.s1_family_attempts_today = 0
            self.state.s1d_fired_today = False
            self.state.s1d_pyramid_count = 0
            self.state.s1e_pyramid_count = 0
            self.state.s1f_reentered_today = False
            self.state.s1f_post_tk_active = False
            self.state.s4_fired_today = False
            self.state.s4_ema_touched = False
            self.state.s1d_ema_touched_today = False
            self.state.s5_fired_today = False
            self.state.range_computed = False  # s5_compression_confirmed alias
            self.state.r3_fired_today = False
            self.state.s8_fired_today = False
            self.state.s2_fired_today = False
            self.state.s3_fired_today = False
            self.state.s6_placed_today = False
            self.state.s7_placed_today = False
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_commission_paid = 0.0
            self.kill_switch_checker.reset_daily(self.state.balance)
            self.portfolio_checker.reset_daily()

            if bar_time.weekday() == 0:  # Monday
                self.kill_switch_checker.reset_weekly(self.state.balance)
                self.state.weekly_pnl = 0.0
        
        # Update event-related state
        if upcoming_events:
            for event in upcoming_events:
                event_time = event.get('time')
                if event_time and (bar_time - event_time).total_seconds() <= 0:
                    self.state.r3_pre_event_price = bar.close
                    self.state.r3_pre_event_atr = context.get('atr_h1', 20)

    def _update_state_after_fill(self, position: SimPosition) -> None:
        """Update state after position fill."""
        strategy_family = get_strategy_family(position.strategy)
        
        if strategy_family == "trend":
            self.state.trend_family_occupied = True
        elif strategy_family == "reversal":
            self.state.reversal_family_occupied = True
        elif strategy_family == "independent":
            lane = position.strategy
            self.state.independent_lanes_occupied[lane] = True
        
        # Add to open positions
        self.state.open_positions[position.ticket] = position

        self.portfolio_checker.add_position({
            "strategy": position.strategy,
            "direction": position.direction,
            "lot_size": position.lot_size,
            "ticket": position.ticket,
            "atr_h1": self.state.__dict__.get("atr_h1", 20.0)
        })

    def _update_state_after_trade(self, trade: TradeRecord) -> None:
        """Update state after trade close."""
        # Update P&L
        self.state.daily_pnl += trade.pnl_net_dollars
        self.state.weekly_pnl += trade.pnl_net_dollars
        
        # Update equity
        self.state.equity += trade.pnl_net_dollars
        self.state.balance = self.state.equity
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        
        # Update consecutive losses
        if trade.pnl_net_dollars < 0:
            self.kill_switch_checker.update_trade_result(-1)
        else:
            self.kill_switch_checker.update_trade_result(1)
        
        if self.kill_switch_checker.ks4_reduced_trades_remaining > 0:
            self.kill_switch_checker.ks4_reduced_trades_remaining -= 1
            self.state.ks4_reduced_trades_remaining = self.kill_switch_checker.ks4_reduced_trades_remaining
        
        # Update position tracking
        strategy_family = get_strategy_family(trade.strategy)
        
        if trade.ticket in self.state.open_positions:
            position = self.state.open_positions[trade.ticket]

            self.portfolio_checker.remove_position({
                "strategy": position.strategy,
                "direction": position.direction,
                "lot_size": position.lot_size,
                "ticket": position.ticket,
                "atr_h1": self.state.__dict__.get("atr_h1", 20.0)
            })
            
            # Check if position was in trend family
            if strategy_family == "trend":
                self.state.trend_family_occupied = False
            elif strategy_family == "reversal":
                self.state.reversal_family_occupied = False
            elif strategy_family == "independent":
                lane = trade.strategy
                if lane in self.state.independent_lanes_occupied:
                    del self.state.independent_lanes_occupied[lane]
            
            # Store last S1 trade for S1b
            if trade.strategy == "S1_LONDON_BRK":
                self.state.last_s1_trade = {
                    'ticket': trade.ticket,
                    'direction': trade.direction,
                    'r_multiple': trade.r_multiple,
                    'stop_price': trade.stop_price
                }
                self.state.last_s1_direction = trade.direction
                self.state.last_s1_max_r = trade.r_multiple
            
            del self.state.open_positions[trade.ticket]

    def _apply_overnight_swap(self, bar_time: datetime) -> None:
        """Deduct overnight swap cost for all open positions."""
        total_swap = 0.0
        for ticket, position in self.execution_sim.open_positions.items():
            # Only charge swap if position was open before today
            if position.entry_time.date() < bar_time.date():
                swap_cost = SWAP_PER_LOT_PER_NIGHT * position.lot_size
                total_swap += swap_cost

        if total_swap != 0.0:
            self.state.equity += total_swap
            self.state.balance = self.state.equity
            self.state.daily_pnl += total_swap
            logger.debug(f"Overnight swap applied: ${total_swap:.2f} at {bar_time.date()}")

    def _update_equity_curve(self, bar_time: datetime) -> None:
        """Update equity curve."""
        equity_point = EquityPoint(
            time=bar_time,
            equity=self.state.equity,
            balance=self.state.balance,
            open_positions=len(self.state.open_positions),
            regime=str(self.state.current_regime)
        )
        self.equity_curve.append(equity_point)

    def _finalize_backtest(self) -> None:
        """Finalize backtest and close any remaining positions."""
        logger.info("Finalizing backtest...")
        
        # Close any remaining positions at last price
        if self.state.open_positions:
            last_bar = self.bar_buffer.get_latest_bars('M5', 1)[0]
            last_price = last_bar['close']
            
            # Iterate over a copy to avoid "dictionary changed size during iteration"
            for ticket, position in list(self.state.open_positions.items()):
                if position.direction == "LONG":
                    pnl_points = last_price - position.entry_price
                else:
                    pnl_points = position.entry_price - last_price

                pnl_usd    = pnl_points * POINT_VALUE * position.lot_size
                commission = COMMISSION_PER_LOT_RT * position.lot_size
                net_pnl    = pnl_usd - commission

                # BUG-19 FIX: guard against zero stop distance
                stop_dist = abs(position.entry_price - position.stop_price)
                r_mult    = pnl_points / stop_dist if stop_dist > 0 else 0.0

                final_trade = TradeRecord(
                    ticket=ticket,
                    strategy=position.strategy,
                    direction=position.direction,
                    lot_size=position.lot_size,
                    entry_price=position.entry_price,
                    entry_time=position.entry_time,
                    exit_price=last_price,
                    exit_time=self.state.current_time,
                    exit_type="FORCED_CLOSE",
                    stop_price=position.stop_price,
                    tp_price=position.tp_price,
                    pnl_points=pnl_points,
                    pnl_gross_dollars=pnl_usd,
                    commission=commission,
                    pnl_net_dollars=net_pnl,
                    r_multiple=r_mult,
                    metadata=dict(position.metadata),
                )

                self.trades.append(final_trade)
                self._update_state_after_trade(final_trade)

    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Calculate performance metrics
        # Separate full closes from partial exits
        full_trades = [t for t in self.trades if t.exit_type != 'PARTIAL_EXIT']
        partial_exits = [t for t in self.trades if t.exit_type == 'PARTIAL_EXIT']

        total_trades = len(full_trades)       # Report only full closes as "trades"
        winning_trades = len([t for t in full_trades if t.pnl_net_dollars > 0])
        losing_trades = len([t for t in full_trades if t.pnl_net_dollars <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # For P&L, expectancy, and R-multiple stats — use ALL trades (full + partial)
        all_closed_trades = self.trades
        total_pnl = sum(t.pnl_net_dollars for t in all_closed_trades)
        total_commission = sum(t.commission for t in all_closed_trades)

        # Calculate expectancy
        avg_win = sum(t.pnl_net_dollars for t in full_trades if t.pnl_net_dollars > 0)
        avg_win = avg_win / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(abs(t.pnl_net_dollars) for t in full_trades if t.pnl_net_dollars <= 0)
        avg_loss = avg_loss / losing_trades if losing_trades > 0 else 0

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Calculate drawdown
        equity_values = [ep.equity for ep in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Calculate Sharpe ratio — on DAILY returns, annualized
        # Group equity_curve by calendar date, take end-of-day (last) equity per day
        sharpe = 0.0
        try:
            daily_equity = {}
            for ep in self.equity_curve:
                day_key = ep.time.date()
                daily_equity[day_key] = ep.equity  # overwrites with last bar of each day

            sorted_days = sorted(daily_equity.keys())
            if len(sorted_days) >= 30:
                daily_equities = [daily_equity[d] for d in sorted_days]
                daily_returns = [
                    daily_equities[i] / daily_equities[i-1] - 1
                    for i in range(1, len(daily_equities))
                ]
                avg_daily = np.mean(daily_returns)
                std_daily = np.std(daily_returns)
                # Annualize: multiply by sqrt(252 trading days)
                sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0
            if len(sorted_days) > 0 and len(sorted_days) < 30:
                logger.warning(
                    f"Sharpe ratio skipped: only {len(sorted_days)} trading days in run "
                    f"(minimum 30 required for statistical validity). Set to 0."
                )
        except Exception as e:
            logger.warning(f"Sharpe calculation failed: {e}")
            sharpe = 0.0
        
        # Strategy breakdown
        strategy_performance = {}
        for strategy in ALL_STRATEGIES:
            strategy_trades = [t for t in self.trades if t.strategy == strategy]
            if strategy_trades:
                strategy_pnl = sum(t.pnl_net_dollars for t in strategy_trades)
                strategy_wins = len([t for t in strategy_trades if t.pnl_net_dollars > 0])
                strategy_performance[strategy] = {
                    'trades': len(strategy_trades),
                    'pnl': strategy_pnl,
                    'win_rate': strategy_wins / len(strategy_trades),
                    'avg_r': np.mean([t.r_multiple for t in strategy_trades])
                }
        
        # Base results
        results = {
            'success': True,
            'summary': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'initial_balance': self.initial_balance,
                'final_balance': self.state.equity,
                'total_pnl': total_pnl,
                'total_commission': total_commission,
                'pnl_pct': (total_pnl / self.initial_balance) * 100,
                'total_trades': total_trades,
                'partial_exits': len(partial_exits),
                'total_closures': len(self.trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate * 100,
                'expectancy': expectancy,
                'max_drawdown_pct': max_dd * 100,
                'sharpe_ratio': sharpe,
                'processed_bars': self.processed_bars,
                'total_bars': self.total_bars
            },
            'strategy_performance': strategy_performance,
            'trades': [t.to_dict() for t in self.trades],
            'equity_curve': [ep.to_dict() for ep in self.equity_curve]
        }
        
        # Add KS6 analysis if auto-reset is enabled
        if should_enable_ks6_auto_reset():
            results = add_ks6_to_final_results(results, self.state.__dict__)
        
        return results

# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATION HELPERS MOVED TO backtest.indicators
# ─────────────────────────────────────────────────────────────────────────────
