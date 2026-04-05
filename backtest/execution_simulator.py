"""
backtest/execution_simulator.py — Order fill and position management simulation.

ExecutionSimulator handles:
  - Pending order fill checks (STOP, LIMIT, MARKET)
  - SL/TP hit detection
  - Partial exit simulation
  - Breakeven activation
  - ATR trailing stop

Contract spec (XAUUSD defaults):
  point = 0.01, tick_size = 0.01, tick_value = 1.0, contract_size = 100
  Commission: $3.50/lot/side = $7.00 round trip

FIXES IN THIS REVISION:
  HIGH-1  Same-bar SL/TP conflict: use proximity-to-open tie-break instead
          of always picking SL. Whichever level is closer to the bar open
          is assumed to have been touched first (matches MT5/QuantConnect
          probabilistic model). Eliminates systematic win-rate understatement.
  HIGH-2  ATR trailing stop now anchors to bar extreme (high for LONG,
          low for SHORT) instead of bar close, matching the live
          manage_s1_position() implementation.
"""
import logging
from datetime import datetime
from typing import Optional

from backtest.models import SimOrder, SimPosition, TradeRecord

logger = logging.getLogger("backtest.execution")

# XAUUSD contract defaults (used if config.CONTRACT_SPEC is empty)
POINT = 0.01
CONTRACT_SIZE = 100.0  # 100 oz per lot
COMMISSION_PER_LOT_ROUND_TRIP = 7.00  # $3.50/side


class ExecutionSimulator:
    """
    Simulates order fills and position management on historical bars.

    Slippage model:
      - STOP orders: fill at order price ± slippage (adverse)
      - LIMIT orders: fill at order price (no slippage — favorable)
      - MARKET orders: fill at bar open ± half_spread ± slippage

    SL/TP model:
      - Checks bar high/low against SL and TP levels.
      - HIGH-1 FIX: If both SL and TP could be hit in the same bar,
        the level CLOSER to bar open is assumed to have been touched
        first and wins. This is the standard probabilistic tie-break
        used by MT5, QuantConnect, and similar frameworks.
        Previous behaviour (always pick SL) systematically understated
        win rate by treating every same-bar case as a loss.
    """

    def __init__(self, slippage_points: float = 0.7):
        """
        Args:
            slippage_points: Slippage in price points (0.7 = $0.70 for XAUUSD).
        """
        self.slippage = slippage_points

    def process_pending_orders(
        self,
        orders: list[SimOrder],
        bar: dict,
        spread: float,
        current_time: datetime,
    ) -> tuple[list[SimPosition], list[SimOrder]]:
        """
        Check if pending orders would have filled during this bar.

        Args:
            orders: List of pending SimOrder objects.
            bar: Current M5 bar dict with open, high, low, close, time.
            spread: Current spread in points.
            current_time: Current simulation timestamp.

        Returns:
            (filled_positions, remaining_orders)
            filled_positions: List of new SimPosition objects from filled orders.
            remaining_orders: Orders that didn't fill and haven't expired.
        """
        filled: list[SimPosition] = []
        remaining: list[SimOrder] = []
        half_spread = spread * POINT * 0.5

        for order in orders:
            # Check expiry first
            if order.expiry is not None and current_time >= order.expiry:
                logger.debug(
                    f"Order expired: {order.strategy} {order.order_type} "
                    f"@ {order.price}"
                )
                continue

            fill_price = self._check_fill(order, bar, half_spread)

            if fill_price is not None:
                pos = SimPosition(
                    strategy=order.strategy,
                    direction=order.direction,
                    entry_price=fill_price,
                    entry_time=current_time,
                    lots=order.lots,
                    stop_price_original=order.sl,
                    current_sl=order.sl,
                    tp=order.tp,
                )
                filled.append(pos)
                logger.debug(
                    f"Order filled: {order.strategy} {order.direction} "
                    f"@ {fill_price:.2f} (order: {order.price:.2f})"
                )
            else:
                remaining.append(order)

        return filled, remaining

    def _check_fill(
        self, order: SimOrder, bar: dict, half_spread: float
    ) -> Optional[float]:
        """
        Check if an order would fill on this bar. Returns fill price or None.

        Fill rules:
          BUY_STOP:  fills if bar.high >= order.price → fill at order.price + slippage
          SELL_STOP: fills if bar.low  <= order.price → fill at order.price - slippage
          BUY_LIMIT: fills if bar.low  <= order.price → fill at order.price (no slippage)
          SELL_LIMIT:fills if bar.high >= order.price → fill at order.price (no slippage)
          MARKET:    fills at bar.open + half_spread + slippage (BUY)
                     or bar.open - half_spread - slippage (SELL)
        """
        ot = order.order_type

        if ot == "BUY_STOP":
            if bar["high"] >= order.price:
                return round(order.price + self.slippage, 3)

        elif ot == "SELL_STOP":
            if bar["low"] <= order.price:
                return round(order.price - self.slippage, 3)

        elif ot == "BUY_LIMIT":
            if bar["low"] <= order.price:
                return round(order.price, 3)

        elif ot == "SELL_LIMIT":
            if bar["high"] >= order.price:
                return round(order.price, 3)

        elif ot == "MARKET":
            if order.direction == "LONG":
                return round(bar["open"] + half_spread + self.slippage, 3)
            else:
                return round(bar["open"] - half_spread - self.slippage, 3)

        return None

    def check_sl_tp(
        self, position: SimPosition, bar: dict
    ) -> tuple[bool, float, str]:
        """
        Check if SL or TP was hit during this bar.

        Returns:
            (closed, exit_price, reason)
            closed: True if position should be closed.
            exit_price: Price at which position exits.
            reason: "SL", "TP", or "" if not closed.

        HIGH-1 FIX — Same-bar SL/TP tie-break:
            When both SL and TP are within the bar range, we cannot know
            from OHLC data alone which was touched first. The previous
            code always chose SL ("conservative worst-case"), which
            systematically understated win rate.

            Fix: compare the DISTANCE from bar open to each level.
            The level closer to bar open is assumed to have been reached
            first. This is the standard probabilistic model used by MT5
            built-in backtester and QuantConnect.

            For LONG:
              dist_to_sl = open - SL    (SL is below open)
              dist_to_tp = TP - open    (TP is above open)
              if dist_to_sl <= dist_to_tp → SL hit first
              else                        → TP hit first

            For SHORT:
              dist_to_sl = SL - open    (SL is above open)
              dist_to_tp = open - TP    (TP is below open)
              if dist_to_sl <= dist_to_tp → SL hit first
              else                        → TP hit first
        """
        sl = position.current_sl
        tp = position.tp
        bar_open = bar["open"]

        if position.direction == "LONG":
            sl_hit = bar["low"] <= sl if sl else False
            tp_hit = bar["high"] >= tp if tp else False

            if sl_hit and tp_hit:
                # HIGH-1 FIX: proximity-to-open tie-break
                dist_to_sl = abs(bar_open - sl)   # SL is below open for LONG
                dist_to_tp = abs(tp - bar_open)   # TP is above open for LONG
                if dist_to_sl <= dist_to_tp:
                    return True, sl, "SL"
                else:
                    return True, tp, "TP"
            elif sl_hit:
                return True, sl, "SL"
            elif tp_hit:
                return True, tp, "TP"

        else:  # SHORT
            sl_hit = bar["high"] >= sl if sl else False
            tp_hit = bar["low"] <= tp if tp else False

            if sl_hit and tp_hit:
                # HIGH-1 FIX: proximity-to-open tie-break
                dist_to_sl = abs(sl - bar_open)   # SL is above open for SHORT
                dist_to_tp = abs(bar_open - tp)   # TP is below open for SHORT
                if dist_to_sl <= dist_to_tp:
                    return True, sl, "SL"
                else:
                    return True, tp, "TP"
            elif sl_hit:
                return True, sl, "SL"
            elif tp_hit:
                return True, tp, "TP"

        return False, 0.0, ""

    def check_partial_exit(
        self,
        position: SimPosition,
        bar: dict,
        partial_r: float = 2.0,
    ) -> tuple[bool, float]:
        """
        Check if partial exit condition is met (R-multiple threshold).

        Returns:
            (should_partial, current_price)
        """
        if position.partial_done:
            return False, 0.0

        price = bar["close"]
        r = position.current_r(price)

        if r >= partial_r:
            return True, price

        return False, 0.0

    def check_be_activation(
        self,
        position: SimPosition,
        bar: dict,
        be_r: float = 1.5,
    ) -> bool:
        """
        Check if breakeven activation condition is met.
        Moves stop to entry price when R-multiple exceeds threshold.

        Returns True if BE should be activated.
        """
        if position.be_activated:
            return False

        price = bar["close"]
        r = position.current_r(price)

        return r >= be_r

    def compute_atr_trail(
        self,
        position: SimPosition,
        bar: dict,
        atr_m15: float,
        trail_mult: float = 2.5,
    ) -> Optional[float]:
        """
        Compute ATR trailing stop level.

        HIGH-2 FIX — Anchor to bar EXTREME, not bar close:
            Previous code used bar["close"] as the anchor point:
              LONG:  new_sl = bar["close"] - atr * mult
              SHORT: new_sl = bar["close"] + atr * mult

            This lags behind on strong trending candles because close < high
            (LONG) or close > low (SHORT), causing the trail to sit further
            away from the current price than intended and leaving more open
            risk than the live system.

            The live manage_s1_position() anchors to the bar EXTREME:
              LONG:  new_sl = bar["high"]  - atr * mult
              SHORT: new_sl = bar["low"]   + atr * mult

            Using the extreme matches the logic that "price reached that high
            (or low) during this bar, so we can trail the stop up to that
            level minus the buffer". The stop only moves in the favorable
            direction (never widens).

        Args:
            position: Open SimPosition being managed.
            bar:      Current M5 bar dict with open, high, low, close.
            atr_m15:  Current ATR on M15 timeframe (in price points).
            trail_mult: Multiplier applied to ATR for the buffer distance.

        Returns:
            New SL level (float) if the stop should be moved, else None.
        """
        if atr_m15 <= 0:
            return None

        trail_dist = atr_m15 * trail_mult

        if position.direction == "LONG":
            # HIGH-2 FIX: anchor to bar high (extreme) not bar close
            new_sl = round(bar["high"] - trail_dist, 3)
            if new_sl > position.current_sl:
                return new_sl
        else:
            # HIGH-2 FIX: anchor to bar low (extreme) not bar close
            new_sl = round(bar["low"] + trail_dist, 3)
            if new_sl < position.current_sl:
                return new_sl

        return None

    @staticmethod
    def compute_trade_pnl(
        direction: str,
        entry_price: float,
        exit_price: float,
        lots: float,
    ) -> tuple[float, float, float]:
        """
        Compute P&L for a closed trade.

        Returns:
            (pnl_gross, commission, pnl_net)
        """
        if direction == "LONG":
            pnl_gross = (exit_price - entry_price) * lots * CONTRACT_SIZE
        else:
            pnl_gross = (entry_price - exit_price) * lots * CONTRACT_SIZE

        commission = lots * COMMISSION_PER_LOT_ROUND_TRIP
        pnl_net = pnl_gross - commission

        return round(pnl_gross, 2), round(commission, 2), round(pnl_net, 2)

    @staticmethod
    def compute_r_multiple(
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_original: float,
    ) -> float:
        """Compute R-multiple for a closed trade."""
        stop_dist = abs(entry_price - stop_original)
        if stop_dist == 0:
            return 0.0

        if direction == "LONG":
            return round((exit_price - entry_price) / stop_dist, 3)
        else:
            return round((entry_price - exit_price) / stop_dist, 3)

    def close_position(
        self,
        position: SimPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        regime_at_exit: str = "",
    ) -> TradeRecord:
        """
        Close a position and create a TradeRecord.

        Args:
            position: The SimPosition to close.
            exit_price: Price at which position exits.
            exit_time: Timestamp of exit.
            exit_reason: Reason for exit (SL, TP, ATR_TRAIL, TIME_KILL, etc.)
            regime_at_exit: Current regime state string.

        Returns:
            TradeRecord with all fields populated.
        """
        pnl_gross, commission, pnl_net = self.compute_trade_pnl(
            position.direction, position.entry_price, exit_price, position.lots
        )
        r_mult = self.compute_r_multiple(
            position.direction, position.entry_price, exit_price,
            position.stop_price_original
        )

        return TradeRecord(
            strategy=position.strategy,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            lots=position.lots,
            pnl=pnl_net,
            pnl_gross=pnl_gross,
            r_multiple=r_mult,
            exit_reason=exit_reason,
            regime_at_entry=position.regime_at_entry,
            regime_at_exit=regime_at_exit,
            stop_original=position.stop_price_original,
            commission=commission,
        )
