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
  OCO-FIX After any order fills, its linked OCO counterpart (identified by
          linked_tag / tag pairing) is immediately cancelled from the
          remaining-orders list. Prevents S6/S7 double-fills where both legs
          of the same OCO pair would otherwise independently fill on
          whipsaw bars.
  GAP-3   check_partial_exit() and check_be_activation() now use the
          intrabar EXTREME (bar high for LONG, bar low for SHORT) to
          evaluate R-multiple thresholds, instead of bar close.
          The live system evaluates these on every tick — the intrabar
          extreme is the closest available OHLC approximation and
          eliminates the systematic miss where price touched the
          partial/BE level mid-bar but pulled back before close.
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

        OCO-FIX: After processing fills, any pending order whose `tag` matches
        a filled order's `linked_tag` is immediately cancelled. This implements
        the OCO (One-Cancels-Other) mechanic for S6/S7 paired orders so that
        when one leg fills its counterpart is removed rather than also filling
        on the same or a subsequent whipsaw bar.
        """
        filled: list[SimPosition] = []
        remaining: list[SimOrder] = []
        half_spread = spread * POINT * 0.5

        # Track which linked_tags were triggered by fills this bar
        cancelled_tags: set[str] = set()
        # First pass: determine fills
        fill_results: list[tuple[SimOrder, Optional[float]]] = []
        for order in orders:
            # Check expiry first
            if order.expiry is not None and current_time >= order.expiry:
                logger.debug(
                    f"Order expired: {order.strategy} {order.order_type} "
                    f"@ {order.price}"
                )
                continue
            fill_price = self._check_fill(order, bar, half_spread)
            fill_results.append((order, fill_price))

        # Second pass: commit fills, collect cancelled OCO tags
        for order, fill_price in fill_results:
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
                # OCO-FIX: mark the linked counterpart for cancellation
                if order.linked_tag:
                    cancelled_tags.add(order.linked_tag)
            else:
                remaining.append(order)

        # OCO-FIX: purge any remaining order whose tag is in cancelled_tags
        if cancelled_tags:
            kept: list[SimOrder] = []
            for order in remaining:
                if order.tag and order.tag in cancelled_tags:
                    logger.debug(
                        f"OCO cancel: {order.strategy} {order.direction} "
                        f"tag={order.tag} cancelled because linked leg filled"
                    )
                else:
                    kept.append(order)
            remaining = kept

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

        GAP-3 FIX: Use intrabar EXTREME instead of bar close.
        For LONG positions the favourable extreme is bar["high"];
        for SHORT positions it is bar["low"]. The live system evaluates
        partial exit on every tick — using the bar extreme is the best
        OHLC approximation and eliminates false negatives where price
        touched the partial level intrabar but closed below it.

        Returns:
            (should_partial, exit_price)
            exit_price is the partial_r level itself (not the extreme)
            so that the simulated fill is realistic, not at the extreme.
        """
        if position.partial_done:
            return False, 0.0

        # GAP-3 FIX: evaluate on intrabar extreme, not close
        if position.direction == "LONG":
            eval_price = bar["high"]
        else:
            eval_price = bar["low"]

        r = position.current_r(eval_price)

        if r >= partial_r:
            # Fill at the exact partial_r price level, not the extreme
            stop_dist = position.stop_distance
            if position.direction == "LONG":
                partial_price = round(position.entry_price + stop_dist * partial_r, 3)
            else:
                partial_price = round(position.entry_price - stop_dist * partial_r, 3)
            return True, partial_price

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

        GAP-3 FIX: Use intrabar EXTREME instead of bar close.
        For LONG positions the favourable extreme is bar["high"];
        for SHORT positions it is bar["low"]. This matches the live
        system's tick-level BE activation — price only needs to reach
        the BE threshold intrabar for BE to activate, regardless of
        where price closes.

        Returns True if BE should be activated.
        """
        if position.be_activated:
            return False

        # GAP-3 FIX: evaluate on intrabar extreme, not close
        if position.direction == "LONG":
            eval_price = bar["high"]
        else:
            eval_price = bar["low"]

        r = position.current_r(eval_price)

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
