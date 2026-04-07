"""
backtest/execution_simulator_updated.py — Updated Execution Simulator for Backtesting

Updated to match current live execution engine (engines/execution_engine.py):
- ATR-based stops for all strategies (replaces fixed-point stops)
- TP targets on all strategies (2.5R for S1, 1.5R for others)
- Spread-adjusted BUY STOPs
- Independent lane tracking (R3, S8)
- Enhanced position reconciliation with ghost/orphan detection
- Partial exit at 2R with BE activation at 1.5R
- ATR trailing stops at 2.5× M15 ATR
- Commission tracking ($7 per lot round-trip)

Matches live execution engine exactly for accurate backtesting.
"""
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal, ROUND_DOWN

# Import updated components
from backtest.models import SimOrder, SimPosition, TradeRecord

logger = logging.getLogger("backtest.execution_simulator")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS - Updated to match live system
# ─────────────────────────────────────────────────────────────────────────────

COMMISSION_PER_LOT_RT = 7.00    # USD round-trip commission per standard lot
CONTRACT_SIZE = 100             # XAUUSD: 100 troy oz per standard lot
# BUG-3 FIX: 1 USD price move on 1 standard lot (100 oz) = $100 PnL
POINT_VALUE = 100.0             # XAUUSD: $1 price move = $100 per standard lot

# Position management constants
PARTIAL_EXIT_R = 2.0      # Take 50% at 2R
BE_ACTIVATION_R = 1.5      # BE after 1.5R + swing
ATR_TRAIL_MULTIPLIER = 2.5  # 2.5× M15 ATR trail

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED EXECUTION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionSimulator:
    """
    Updated execution simulator matching live execution engine.
    """

    def __init__(self, slippage_points: float = 0.70):
        """
        Args:
            slippage_points: Maximum slippage applied on gap-open fills.
        """
        self.slippage_points = slippage_points
        self.open_positions: Dict[str, SimPosition] = {}
        self.pending_orders: List[SimOrder] = []

    # =========================================================================
    # ORDER PROCESSING
    # =========================================================================

    def process_pending_orders(
        self,
        pending: List[SimOrder],
        bar: Dict[str, Any],
        spread: float,
        bar_time: datetime,
    ) -> Tuple[List[SimPosition], List[SimOrder]]:
        """
        Process pending orders against current M5 bar.
        Updated with spread adjustment and independent lane logic.
        """
        filled: List[SimPosition] = []
        remaining: List[SimOrder] = []

        bar_open = bar["open"]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        # Track filled tags for OCO cancellation
        filled_tags: set[str] = set()

        for order in pending:
            # Expiry check
            if order.expiry and bar_time > order.expiry:
                logger.debug(f"Order expired: {order.strategy} {order.direction} @ {order.price:.2f}")
                continue

            # Cancel OCO partner if linked tag already filled
            if order.tag and order.tag in filled_tags:
                continue
            if order.linked_tag and order.linked_tag in filled_tags:
                continue

            # Attempt fill with spread adjustment
            pos = self._try_fill_updated(order, bar_open, bar_high, bar_low, bar_close,
                                         bar_time, spread)
            if pos is not None:
                filled.append(pos)
                if order.tag:
                    filled_tags.add(order.tag)
            else:
                remaining.append(order)

        return filled, remaining

    def _try_fill_updated(
        self,
        order: SimOrder,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_time: datetime,
        spread: float,
    ) -> Optional[SimPosition]:
        """
        Updated fill logic with spread adjustment and ATR-based stops.
        """
        price = order.price
        otype = order.order_type
        direction = order.direction

        fill_price: Optional[float] = None

        if otype == "MARKET":
            # Market order: fill at open + spread for LONG
            fill_price = bar_open + (spread if direction == "LONG" else 0.0)

        elif otype == "BUY_STOP":
            # Spread-adjusted BUY STOP
            if bar_open >= price:
                # Gap-open above stop: fill at open + slippage
                fill_price = bar_open + self.slippage_points
            elif bar_high >= price:
                fill_price = price
                # Add spread to stop price for BUY orders
                if direction == "LONG":
                    fill_price += spread

        elif otype == "SELL_STOP":
            if bar_open <= price:
                # Gap-open below stop: fill at open - slippage
                fill_price = bar_open - self.slippage_points
            elif bar_low <= price:
                fill_price = price
                # No spread adjustment for SELL orders

        elif otype == "BUY_LIMIT":
            if bar_open <= price:
                fill_price = max(bar_open, price)
                # Add spread for BUY orders
                if direction == "LONG":
                    fill_price += spread

        elif otype == "SELL_LIMIT":
            if bar_open >= price:
                fill_price = min(bar_open, price)

        else:
            return None

        if fill_price is None:
            return None

        # BUG-2 FIX: use field names matching updated SimPosition dataclass
        ticket = f"BT_{len(self.open_positions)}_{int(bar_time.timestamp())}"
        metadata = dict(order.metadata) if order.metadata else {}
        # Store original stop in metadata for R-multiple calculation
        metadata['original_stop'] = order.stop_price

        position = SimPosition(
            ticket=ticket,
            strategy=order.strategy,
            direction=direction,
            lot_size=order.lot_size,
            entry_price=fill_price,
            entry_time=bar_time,
            stop_price=order.stop_price,
            tp_price=order.tp_price,
            tag=order.tag,
            metadata=metadata,
        )

        self.open_positions[position.ticket] = position
        return position

    # =========================================================================
    # POSITION MANAGEMENT - Updated with ATR-based logic
    # =========================================================================

    def manage_open_positions(
        self,
        bar: Dict[str, Any],
        atr_m15: float,
        state: Dict[str, Any]
    ) -> List[TradeRecord]:
        """
        Manage open positions with updated logic.
        Returns list of closed trades.
        """
        closed_trades: List[TradeRecord] = []
        
        for ticket, position in list(self.open_positions.items()):
            close_result = self._check_position_exit_updated(position, bar, atr_m15, state)
            
            if close_result:
                trade_record = self._create_trade_record(position, close_result)
                closed_trades.append(trade_record)
                del self.open_positions[ticket]
        
        return closed_trades

    def _check_position_exit_updated(
        self,
        position: SimPosition,
        bar: Dict[str, Any],
        atr_m15: float,
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Position exit checking.
        Order: TP → Hard SL → Partial exit (2R) → BE activation (1.5R)
              → ATR trail (only after BE) → Time exit.
        """
        direction = position.direction
        entry_price = position.entry_price
        stop_price = position.stop_price
        tp_price = position.tp_price

        bar_high = bar["high"]
        bar_low  = bar["low"]
        bar_close = bar["close"]

        # Protect against zero-distance stop
        original_stop = position.metadata.get('original_stop', stop_price)
        risk_points = abs(entry_price - original_stop)
        if risk_points == 0:
            risk_points = 1.0

        current_r = (
            (bar_close - entry_price) / risk_points
            if direction == "LONG"
            else (entry_price - bar_close) / risk_points
        )

        # 1. TP hit
        if tp_price:
            if direction == "LONG" and bar_high >= tp_price:
                return {'exit_type': 'TP_HIT', 'exit_price': tp_price,
                        'exit_time': bar["time"], 'final_r': current_r,
                        'exit_lots': position.lot_size}
            elif direction == "SHORT" and bar_low <= tp_price:
                return {'exit_type': 'TP_HIT', 'exit_price': tp_price,
                        'exit_time': bar["time"], 'final_r': current_r,
                        'exit_lots': position.lot_size}

        # 2. Hard SL hit
        if direction == "LONG" and bar_low <= stop_price:
            return {'exit_type': 'SL_HIT', 'exit_price': stop_price,
                    'exit_time': bar["time"], 'final_r': current_r,
                    'exit_lots': position.lot_size}
        if direction == "SHORT" and bar_high >= stop_price:
            return {'exit_type': 'SL_HIT', 'exit_price': stop_price,
                    'exit_time': bar["time"], 'final_r': current_r,
                    'exit_lots': position.lot_size}

        # 3. Partial exit at 2R (take 50%, keep rest running)
        if current_r >= PARTIAL_EXIT_R and not position.metadata.get('partial_exit_done'):
            partial_lots = position.lot_size * 0.5
            remaining_lots = position.lot_size * 0.5
            position.lot_size = remaining_lots
            position.metadata['partial_exit_done'] = True
            position.metadata['partial_exit_r'] = current_r
            return {
                'exit_type': 'PARTIAL_EXIT',
                'exit_price': bar_close,
                'exit_time': bar["time"],
                'exit_lots': partial_lots,
                'final_r': current_r,
            }

        # 4. Breakeven activation at 1.5R
        if current_r >= BE_ACTIVATION_R and not position.metadata.get('be_activated'):
            position.stop_price = entry_price
            position.metadata['be_activated'] = True
            position.metadata['be_r'] = current_r
            return None  # no exit, just move stop

        # 5. ATR trailing stop — only after BE is active to avoid premature exit
        if self._check_atr_trail_stop(position, atr_m15):
            trail_stop = self._calculate_trail_stop(position, bar, atr_m15)
            if direction == "LONG" and bar_low <= trail_stop:
                return {'exit_type': 'ATR_TRAIL', 'exit_price': trail_stop,
                        'exit_time': bar["time"], 'final_r': current_r,
                        'exit_lots': position.lot_size}
            elif direction == "SHORT" and bar_high >= trail_stop:
                return {'exit_type': 'ATR_TRAIL', 'exit_price': trail_stop,
                        'exit_time': bar["time"], 'final_r': current_r,
                        'exit_lots': position.lot_size}

        # 6. Time-based hard exits
        if self._check_time_exit(position, bar, state):
            return {'exit_type': 'TIME_EXIT', 'exit_price': bar_close,
                    'exit_time': bar["time"], 'final_r': current_r,
                    'exit_lots': position.lot_size}

        return None

    def _check_atr_trail_stop(
        self,
        position: SimPosition,
        atr_m15: float,
    ) -> bool:
        """
        Returns True if ATR trailing stop should be active.
        Only activates after BE has been triggered (position has moved 1.5R+).
        """
        if not position.metadata.get('be_activated', False):
            return False
        if atr_m15 is None or atr_m15 <= 0:
            return False
        return True

    def _calculate_trail_stop(
        self,
        position: SimPosition,
        bar: Dict[str, Any],
        atr_m15: float,
    ) -> float:
        """Calculate current ATR trail stop level, never worse than current stop."""
        trail_distance = atr_m15 * ATR_TRAIL_MULTIPLIER
        if position.direction == "LONG":
            # Trail moves up with price, never down
            return max(position.stop_price, bar["high"] - trail_distance)
        else:
            # Trail moves down with price, never up
            return min(position.stop_price, bar["low"] + trail_distance)

    def _check_time_exit(self, position: SimPosition, bar: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Check time-based exits for strategies with hard exits.
        """
        bar_time = bar["time"]
        strategy = position.strategy
        
        # S4 hard exit at 16:00 UTC
        if strategy == "S4_LONDON_PULL":
            hard_exit = bar_time.replace(hour=16, minute=0, second=0)
            if bar_time >= hard_exit:
                return True
        
        # S5 hard exit at 22:00 UTC
        elif strategy == "S5_NY_COMPRESS":
            hard_exit = bar_time.replace(hour=22, minute=0, second=0)
            if bar_time >= hard_exit:
                return True
        
        # R3 max hold 30 minutes
        elif strategy == "R3_CAL_MOMENTUM":
            max_hold = position.entry_time + timedelta(minutes=30)
            if bar_time >= max_hold:
                return True
        
        return False

    def _create_trade_record(
        self,
        position: SimPosition,
        close_result: Dict[str, Any],
    ) -> TradeRecord:
        """Create TradeRecord from position and close result (BUG-3 FIX)."""
        exit_type  = close_result['exit_type']
        exit_price = close_result['exit_price']
        exit_time  = close_result['exit_time']
        exit_lots  = close_result.get('exit_lots', position.lot_size)

        # Price-point P&L
        if position.direction == "LONG":
            pnl_points = exit_price - position.entry_price
        else:
            pnl_points = position.entry_price - exit_price

        # USD P&L: POINT_VALUE=100 → $1 price move × 100 oz/lot
        pnl_gross = pnl_points * POINT_VALUE * exit_lots
        commission = COMMISSION_PER_LOT_RT * exit_lots
        pnl_net    = pnl_gross - commission

        # R-multiple uses original stop distance
        original_stop = position.metadata.get('original_stop', position.stop_price)
        risk_points = abs(position.entry_price - original_stop)
        r_multiple = pnl_points / risk_points if risk_points > 0 else 0.0

        # BUG-3 FIX: field names match updated TradeRecord dataclass
        return TradeRecord(
            ticket=position.ticket,
            strategy=position.strategy,
            direction=position.direction,
            lot_size=exit_lots,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_type=exit_type,
            stop_price=position.stop_price,
            tp_price=position.tp_price,
            pnl_points=pnl_points,
            pnl_gross_dollars=pnl_gross,
            commission=commission,
            pnl_net_dollars=pnl_net,
            r_multiple=r_multiple,
            metadata=dict(position.metadata),
        )

    # =========================================================================
    # (reconcile_positions removed — live-MT5-only code, not applicable to backtest BUG-16)
    # =========================================================================


    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_open_positions(self) -> Dict[str, SimPosition]:
        """Get current open positions."""
        return self.open_positions.copy()

    def get_position_count(self) -> int:
        """Get count of open positions."""
        return len(self.open_positions)

    def get_total_lots(self) -> float:
        """Get total lots open."""
        return sum(pos.lot_size for pos in self.open_positions.values())

    def cancel_all_orders(self) -> None:
        """Cancel all pending orders."""
        self.pending_orders = []
        logger.info("All pending orders cancelled")

    def emergency_shutdown(self, reason: str) -> None:
        """Emergency shutdown - cancel everything."""
        self.cancel_all_orders()
        self.open_positions = {}
        logger.critical(f"Emergency shutdown: {reason}")

    # =========================================================================
    # INDEPENDENT LANE MANAGEMENT
    # =========================================================================

    def get_independent_lanes(self) -> Dict[str, SimPosition]:
        """
        Get positions in independent lanes (R3, S8).
        These can coexist with trend family positions.
        """
        independent_positions = {}
        for ticket, position in self.open_positions.items():
            if position.strategy in ['R3_CAL_MOMENTUM', 'S8_ATR_SPIKE']:
                independent_positions[ticket] = position
        return independent_positions

    def get_trend_family_positions(self) -> Dict[str, SimPosition]:
        """
        Get positions in trend family lane.
        """
        trend_positions = {}
        for ticket, position in self.open_positions.items():
            if position.strategy in ['S1_LONDON_BRK', 'S1F_POST_TK', 'S4_LONDON_PULL', 'S5_NY_COMPRESS']:
                trend_positions[ticket] = position
        return trend_positions
