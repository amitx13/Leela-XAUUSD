"""
Enhanced Execution Simulator

Realistic order execution with deterministic price-cross fill logic.

Fix (v2):
  _simulate_fill_decision() replaced with deterministic price-cross logic.
  STOP/LIMIT orders fill only when bar high/low actually crosses the order
  price — not via random probability. MARKET orders always fill.
  Engine passes (bar_high, bar_low) into check_fill() for accuracy.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from backtest.models import SimOrder, SimPosition

logger = logging.getLogger("backtest.enhanced_execution_simulator")


@dataclass
class FillResult:
    """Result of order fill simulation."""
    filled: bool
    fill_price: float
    fill_time: datetime
    fill_volume: float
    slippage: float
    partial_fill: bool = False
    phantom_order: bool = False
    rejection_reason: Optional[str] = None


class EnhancedExecutionSimulator:
    """
    Execution simulator with deterministic price-cross fill logic.

    Fill rules (mirrors MT5 behaviour):
      MARKET    — always fills at current_price ± slippage
      BUY_STOP  — fills when bar_high  >= order.price
      SELL_STOP — fills when bar_low   <= order.price
      BUY_LIMIT — fills when bar_low   <= order.price
      SELL_LIMIT— fills when bar_high  >= order.price
    """

    def __init__(self, slippage_points: float = 0.7):
        self.slippage_points = slippage_points
        self.pending_orders:  List[SimOrder]       = []
        self.order_history:   List[Dict[str, Any]] = []
        logger.info(f"Enhanced execution simulator initialised (slippage={slippage_points})")

    # ------------------------------------------------------------------ #
    # PUBLIC
    # ------------------------------------------------------------------ #

    def submit_order(self, order: SimOrder) -> None:
        self.pending_orders.append(order)
        self.order_history.append({
            "order":         order,
            "submit_time":   datetime.utcnow(),
            "status":        "PENDING",
            "fill_attempts": 0,
        })
        logger.debug(f"Order submitted: {order.strategy} {order.direction} @ {order.price}")

    def check_fill(
        self,
        order:         SimOrder,
        current_price: float,
        current_time:  datetime,
        bar_high:      Optional[float] = None,
        bar_low:       Optional[float] = None,
    ) -> FillResult:
        """
        Deterministic fill check.

        Args:
            order:         The pending order to evaluate.
            current_price: Bar close price (used for MARKET fills & fallback).
            current_time:  Current simulation time.
            bar_high:      Bar high  — required for BUY_STOP / SELL_LIMIT.
            bar_low:       Bar low   — required for SELL_STOP / BUY_LIMIT.
        """
        meta = self._get_meta(order)
        if not meta:
            return FillResult(False, 0.0, current_time, 0.0, 0.0,
                              rejection_reason="ORDER_NOT_FOUND")

        if meta["status"] == "FILLED":
            return FillResult(
                True,
                meta["fill_price"],
                meta["fill_time"],
                meta["fill_volume"],
                meta["slippage"],
                partial_fill=meta.get("partial_fill", False),
            )

        if order.expiry and current_time >= order.expiry:
            meta["status"] = "EXPIRED"
            return FillResult(False, 0.0, current_time, 0.0, 0.0,
                              rejection_reason="ORDER_EXPIRED")

        meta["fill_attempts"] += 1

        # ── Determine fill price using price-cross logic ─────────────── #
        fill_price = self._price_cross_fill(
            order, current_price, bar_high, bar_low
        )

        if fill_price is None:
            return FillResult(False, 0.0, current_time, 0.0, 0.0,
                              rejection_reason="PRICE_NOT_REACHED")

        # Apply slippage (adverse: costs the trader)
        slippage = random.uniform(0.0, self.slippage_points)
        if order.direction == "LONG":
            fill_price += slippage
        else:
            fill_price -= slippage

        fill_volume = order.lots

        meta.update({
            "status":       "FILLED",
            "fill_time":    current_time,
            "fill_price":   fill_price,
            "fill_volume":  fill_volume,
            "slippage":     slippage,
            "partial_fill": False,
            "phantom_order": False,
        })
        if order in self.pending_orders:
            self.pending_orders.remove(order)

        logger.info(
            f"Filled: {order.strategy} {order.direction} {order.order_type} "
            f"@ {fill_price:.2f} (slippage {slippage:.2f})"
        )
        return FillResult(
            True, fill_price, current_time, fill_volume, slippage
        )

    # ------------------------------------------------------------------ #
    # INTERNAL FILL LOGIC
    # ------------------------------------------------------------------ #

    def _price_cross_fill(
        self,
        order:         SimOrder,
        current_price: float,
        bar_high:      Optional[float],
        bar_low:       Optional[float],
    ) -> Optional[float]:
        """
        Return the fill price if the bar crosses the order level, else None.

        Uses bar_high / bar_low when available (accurate); falls back to
        current_price (bar close) only for MARKET orders or when hi/lo are
        not provided.
        """
        ot = order.order_type
        op = order.price

        if ot == "MARKET":
            return current_price

        high = bar_high if bar_high is not None else current_price
        low  = bar_low  if bar_low  is not None else current_price

        if ot == "BUY_STOP":
            # Triggers when price rises to or above order level
            if high >= op:
                return op
        elif ot == "SELL_STOP":
            # Triggers when price falls to or below order level
            if low <= op:
                return op
        elif ot == "BUY_LIMIT":
            # Triggers when price falls to or below order level (buy cheaper)
            if low <= op:
                return op
        elif ot == "SELL_LIMIT":
            # Triggers when price rises to or above order level (sell higher)
            if high >= op:
                return op
        elif ot in ("BUY", "SELL"):
            # Immediate market-style orders
            return current_price

        return None

    def _get_meta(self, order: SimOrder) -> Optional[Dict[str, Any]]:
        for m in self.order_history:
            if m["order"] is order:
                return m
        return None

    # ------------------------------------------------------------------ #
    # STATS / RESET
    # ------------------------------------------------------------------ #

    def get_pending_orders(self) -> List[SimOrder]:
        return self.pending_orders.copy()

    def get_order_history(self) -> List[Dict[str, Any]]:
        return self.order_history.copy()

    def reset(self) -> None:
        self.pending_orders.clear()
        self.order_history.clear()
        logger.info("Execution simulator reset")

    def get_execution_stats(self) -> Dict[str, Any]:
        if not self.order_history:
            return {}
        filled   = [m for m in self.order_history if m["status"] == "FILLED"]
        rejected = [m for m in self.order_history if m["status"] == "REJECTED"]
        expired  = [m for m in self.order_history if m["status"] == "EXPIRED"]
        total    = len(self.order_history)
        return {
            "total_orders":     total,
            "filled_orders":    len(filled),
            "rejected_orders":  len(rejected),
            "expired_orders":   len(expired),
            "fill_rate":        len(filled) / total if total > 0 else 0,
            "avg_slippage":     sum(m.get("slippage", 0) for m in filled) / len(filled) if filled else 0,
        }

    def log_execution_summary(self) -> None:
        stats = self.get_execution_stats()
        logger.info("=== EXECUTION SIMULATOR SUMMARY ===")
        logger.info(f"Total Orders : {stats.get('total_orders', 0)}")
        logger.info(f"Filled       : {stats.get('filled_orders', 0)} ({stats.get('fill_rate', 0):.1%})")
        logger.info(f"Rejected     : {stats.get('rejected_orders', 0)}")
        logger.info(f"Expired      : {stats.get('expired_orders', 0)}")
        logger.info(f"Avg Slippage : {stats.get('avg_slippage', 0):.2f} pts")
        logger.info("=====================================")
