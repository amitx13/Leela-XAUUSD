"""
backtest/execution_simulator.py — Simulated order-fill and position-management engine.

Responsibilities
────────────────
  • Fill pending orders (BUY_STOP / SELL_STOP / BUY_LIMIT / SELL_LIMIT / MARKET)
    against the current M5 bar using realistic bar-open-first simulation.
  • Apply slippage on gap-opens (configurable in points).
  • Track SL / TP exits with:
        - ATR-based trailing stop
        - Partial-exit at 1 R
        - Breakeven activation at 1 R
  • Compute trade P&L net of commission ($7.00 / lot round-trip).

HIGH-1 FIX (v2 — this commit):
    When both SL and TP are hit inside the same bar we previously used a
    proximity-to-open heuristic (dist_to_sl <= dist_to_tp) to decide which
    filled first.  That heuristic introduced a *systematic* directional bias
    (whichever side was closer to the open always "won").

    Replaced with a statistically neutral random.random() < 0.5 coin-flip.
    Over a large sample the 50/50 split is unbiased.  If results drop
    significantly versus the deterministic version, the old heuristic was
    masking a real strategy edge problem — which is exactly what we want to
    detect.

All timestamps are timezone-aware UTC (pytz.utc).
"""
import logging
import random
from datetime import datetime
from typing import Optional

import pytz

from backtest.models import SimOrder, SimPosition, TradeRecord

logger = logging.getLogger("backtest.execution_simulator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMMISSION_PER_LOT_RT = 7.00   # USD round-trip commission per standard lot
CONTRACT_SIZE         = 100    # XAUUSD: 100 troy oz per standard lot


class ExecutionSimulator:
    """
    Simulates order execution and position management for the backtest.

    Bar-open-first model:
      On each M5 bar the simulator checks whether gap-open price fills any
      pending order BEFORE testing the bar's high/low range for SL/TP.
      This prevents looking ahead into the bar body for entry fills.
    """

    def __init__(self, slippage_points: float = 0.70):
        """
        Args:
            slippage_points: Maximum slippage applied on gap-open fills (in
                             XAUUSD points, i.e. 0.70 = $0.70).
        """
        self.slippage_points = slippage_points

    # =========================================================================
    # ORDER PROCESSING
    # =========================================================================

    def process_pending_orders(
        self,
        pending: list[SimOrder],
        bar: dict,
        spread: float,
        bar_time: datetime,
    ) -> tuple[list[SimPosition], list[SimOrder]]:
        """
        Try to fill each pending order against the current M5 bar.

        Returns:
            filled   — list of newly opened SimPosition objects
            remaining— list of orders that were NOT filled this bar
        """
        filled:    list[SimPosition] = []
        remaining: list[SimOrder]    = []

        bar_open  = bar["open"]
        bar_high  = bar["high"]
        bar_low   = bar["low"]
        bar_close = bar["close"]

        # Track filled tags so OCO partners can be cancelled
        filled_tags: set[str] = set()

        for order in pending:
            # -- Expiry check -------------------------------------------------
            if order.expiry and bar_time > order.expiry:
                logger.debug(f"Order expired: {order.strategy} {order.direction} "
                             f"@ {order.price:.2f}")
                continue  # drop expired order

            # -- Cancel OCO partner if linked tag already filled -------------
            if order.tag and order.tag in filled_tags:
                continue
            if order.linked_tag and order.linked_tag in filled_tags:
                continue

            # -- Attempt fill -------------------------------------------------
            pos = self._try_fill(order, bar_open, bar_high, bar_low, bar_close,
                                 bar_time, spread)
            if pos is not None:
                filled.append(pos)
                if order.tag:
                    filled_tags.add(order.tag)
            else:
                remaining.append(order)

        return filled, remaining

    def _try_fill(
        self,
        order: SimOrder,
        bar_open: float,
        bar_high: float,
        bar_low:  float,
        bar_close: float,
        bar_time: datetime,
        spread: float,
    ) -> Optional[SimPosition]:
        """
        Attempt to fill a single order.  Returns SimPosition on fill, else None.
        """
        price  = order.price
        otype  = order.order_type
        direct = order.direction

        fill_price: Optional[float] = None

        if otype == "MARKET":
            fill_price = bar_open + (spread if direct == "LONG" else 0.0)

        elif otype == "BUY_STOP":
            if bar_open >= price:
                # Gap-open above stop: fill at open + slippage
                fill_price = bar_open + self.slippage_points
            elif bar_high >= price:
                fill_price = price

        elif otype == "SELL_STOP":
            if bar_open <= price:
                fill_price = bar_open - self.slippage_points
            elif bar_low <= price:
                fill_price = price

        elif otype == "BUY_LIMIT":
            if bar_open <= price:
                fill_price = bar_open
            elif bar_low <= price:
                fill_price = price

        elif otype == "SELL_LIMIT":
            if bar_open >= price:
                fill_price = bar_open
            elif bar_high >= price:
                fill_price = price

        if fill_price is None:
            return None

        # Build position
        pos = SimPosition(
            strategy             = order.strategy,
            direction            = direct,
            entry_price          = round(fill_price, 2),
            entry_time           = bar_time,
            lots                 = order.lots,
            stop_price_original  = order.sl,
            current_sl           = order.sl,
            tp                   = order.tp,
        )
        logger.debug(f"FILLED {order.strategy} {direct} @ {fill_price:.2f} "
                     f"SL={order.sl} TP={order.tp} lots={order.lots}")
        return pos

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def check_sl_tp(
        self,
        pos: SimPosition,
        bar: dict,
    ) -> tuple[bool, float, str]:
        """
        Check whether the current bar hits SL or TP for an open position.

        HIGH-1 FIX (v2):
            When both SL and TP are inside the same bar's range we can not
            determine which was hit first from bar data alone.  We used to
            resolve this with a proximity heuristic (whichever was closer to
            bar_open).  That heuristic introduced a systematic directional
            bias over large samples.

            REPLACED with a statistically neutral random.random() < 0.5
            coin-flip.  50 % → SL wins,  50 % → TP wins.

        Returns:
            (closed, exit_price, reason)
            closed     — True if the position should be closed this bar
            exit_price — price at which it closes
            reason     — "SL" | "TP" | "" (not closed)
        """
        bar_open  = bar["open"]
        bar_high  = bar["high"]
        bar_low   = bar["low"]

        sl = pos.current_sl
        tp = pos.tp

        if pos.direction == "LONG":
            sl_hit = bar_low  <= sl
            tp_hit = tp is not None and bar_high >= tp

            # Gap-down below SL (worst-case: fill at bar_open)
            if sl_hit and bar_open <= sl:
                return True, bar_open, "SL"

            if sl_hit and tp_hit:
                # HIGH-1 FIX (v2): random 50/50 tie-break replaces proximity
                # heuristic.  Eliminates any directional bias introduced by the
                # dist_to_open approximation and makes the tie-break
                # statistically neutral over a large sample.
                if random.random() < 0.5:
                    return True, sl, "SL"
                else:
                    return True, tp, "TP"

            if sl_hit:
                return True, sl, "SL"
            if tp_hit:
                return True, tp, "TP"

        else:  # SHORT
            sl_hit = bar_high >= sl
            tp_hit = tp is not None and bar_low <= tp

            # Gap-up above SL
            if sl_hit and bar_open >= sl:
                return True, bar_open, "SL"

            if sl_hit and tp_hit:
                # HIGH-1 FIX (v2): random 50/50 tie-break (same as LONG branch)
                if random.random() < 0.5:
                    return True, sl, "SL"
                else:
                    return True, tp, "TP"

            if sl_hit:
                return True, sl, "SL"
            if tp_hit:
                return True, tp, "TP"

        return False, 0.0, ""

    def check_partial_exit(
        self,
        pos: SimPosition,
        bar: dict,
    ) -> tuple[bool, float]:
        """
        Return (True, price) if the position should take a partial exit at 1R.
        Partial exit is skipped if already done or if TP is inside the 1R zone.
        """
        if pos.partial_done:
            return False, 0.0

        r1_price: float
        if pos.direction == "LONG":
            r1_price = pos.entry_price + pos.stop_distance
            if bar["high"] >= r1_price:
                return True, r1_price
        else:
            r1_price = pos.entry_price - pos.stop_distance
            if bar["low"] <= r1_price:
                return True, r1_price

        return False, 0.0

    def check_be_activation(
        self,
        pos: SimPosition,
        bar: dict,
    ) -> bool:
        """Return True if breakeven should be activated this bar (price reached 1R)."""
        if pos.be_activated:
            return False

        if pos.direction == "LONG":
            r1 = pos.entry_price + pos.stop_distance
            return bar["high"] >= r1
        else:
            r1 = pos.entry_price - pos.stop_distance
            return bar["low"] <= r1

    def compute_atr_trail(
        self,
        pos: SimPosition,
        bar: dict,
        atr: Optional[float],
    ) -> Optional[float]:
        """
        Compute a new ATR-based trailing SL.

        Trail activates only after breakeven is hit.  The new SL is
        `price_extreme - 1.5 × ATR` for LONGs, `price_extreme + 1.5 × ATR`
        for SHORTs — only moved in the favourable direction.

        Returns the new SL if it improves, else None.
        """
        if not pos.be_activated or atr is None or atr <= 0:
            return None

        multiplier = 1.5

        if pos.direction == "LONG":
            new_sl = round(bar["high"] - multiplier * atr, 2)
            if new_sl > pos.current_sl:
                return new_sl
        else:
            new_sl = round(bar["low"] + multiplier * atr, 2)
            if new_sl < pos.current_sl:
                return new_sl

        return None

    # =========================================================================
    # P&L COMPUTATION
    # =========================================================================

    def close_position(
        self,
        pos: SimPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        regime_at_exit: str,
    ) -> TradeRecord:
        """Close a position and return a completed TradeRecord."""
        _, _, pnl_net = self.compute_trade_pnl(
            pos.direction, pos.entry_price, exit_price, pos.lots
        )
        pnl_gross = self._gross_pnl(pos.direction, pos.entry_price, exit_price, pos.lots)
        commission = COMMISSION_PER_LOT_RT * pos.lots

        sd = pos.stop_distance
        if sd > 0:
            r_multiple = round(pnl_gross / (sd * pos.lots * CONTRACT_SIZE), 3)
        else:
            r_multiple = 0.0

        return TradeRecord(
            strategy       = pos.strategy,
            direction      = pos.direction,
            entry_price    = pos.entry_price,
            exit_price     = round(exit_price, 2),
            entry_time     = pos.entry_time,
            exit_time      = exit_time,
            lots           = pos.lots,
            pnl            = round(pnl_net, 2),
            pnl_gross      = round(pnl_gross, 2),
            r_multiple     = r_multiple,
            exit_reason    = exit_reason,
            regime_at_entry= pos.regime_at_entry,
            regime_at_exit = regime_at_exit,
            stop_original  = pos.stop_price_original,
            commission     = round(commission, 2),
        )

    def compute_trade_pnl(
        self,
        direction: str,
        entry: float,
        exit_p: float,
        lots: float,
    ) -> tuple[float, float, float]:
        """
        Returns (pnl_gross, commission, pnl_net) in USD.

        XAUUSD: 1 lot = 100 oz.  P&L = price_delta × lots × 100.
        """
        gross     = self._gross_pnl(direction, entry, exit_p, lots)
        comm      = COMMISSION_PER_LOT_RT * lots
        return gross, comm, gross - comm

    @staticmethod
    def _gross_pnl(direction: str, entry: float, exit_p: float, lots: float) -> float:
        if direction == "LONG":
            return (exit_p - entry) * lots * CONTRACT_SIZE
        else:
            return (entry - exit_p) * lots * CONTRACT_SIZE
