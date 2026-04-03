"""
Enhanced Execution Simulator

Realistic order execution with phantom order detection and enhanced fills.
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
    Enhanced execution simulator with realistic fill modeling.
    
    Features:
    - Realistic fill probabilities
    - Slippage modeling
    - Partial fills
    - Phantom order detection
    - Order expiry handling
    - Spread-aware execution
    - Latency simulation
    """
    
    def __init__(self, slippage_points: float = 0.7):
        self.slippage_points = slippage_points
        self.pending_orders: List[SimOrder] = []
        self.order_history: List[Dict[str, Any]] = []
        
        logger.info(f"Enhanced execution simulator initialized with slippage: {slippage_points}")
    
    def submit_order(self, order: SimOrder) -> None:
        """Submit order for execution."""
        order_with_metadata = {
            "order": order,
            "submit_time": datetime.utcnow(),
            "status": "PENDING",
            "fill_attempts": 0,
            "phantom_checks": 0
        }
        
        self.pending_orders.append(order)
        self.order_history.append(order_with_metadata)
        
        logger.debug(f"Order submitted: {order.strategy} {order.direction} @ {order.price}")
    
    def check_fill(self, order: SimOrder, current_price: float, current_time: datetime) -> FillResult:
        """Check if order should be filled with enhanced logic."""
        # Find order in pending list
        order_metadata = None
        for om in self.order_history:
            if om["order"] == order:
                order_metadata = om
                break
        
        if not order_metadata:
            return FillResult(
                filled=False,
                fill_price=0.0,
                fill_time=current_time,
                fill_volume=0.0,
                slippage=0.0,
                phantom_order=False,
                rejection_reason="ORDER_NOT_FOUND"
            )
        
        # Check if order already filled
        if order_metadata["status"] == "FILLED":
            return FillResult(
                filled=True,
                fill_price=order_metadata.get("fill_price", order.price),
                fill_time=order_metadata.get("fill_time", current_time),
                fill_volume=order_metadata.get("fill_volume", order.lots),
                slippage=order_metadata.get("slippage", 0.0),
                phantom_order=order_metadata.get("phantom_order", False),
                partial_fill=order_metadata.get("partial_fill", False)
            )
        
        # Check order expiry
        if order.expiry and current_time >= order.expiry:
            order_metadata["status"] = "EXPIRED"
            return FillResult(
                filled=False,
                fill_price=0.0,
                fill_time=current_time,
                fill_volume=0.0,
                slippage=0.0,
                phantom_order=False,
                rejection_reason="ORDER_EXPIRED"
            )
        
        # Increment fill attempts
        order_metadata["fill_attempts"] += 1
        
        # Simulate fill decision
        fill_decision = self._simulate_fill_decision(order, current_price, current_time)
        
        if fill_decision["fill"]:
            # Order fills
            fill_price = fill_decision["fill_price"]
            slippage = fill_decision["slippage"]
            fill_volume = fill_decision["fill_volume"]
            
            # Update order metadata
            order_metadata.update({
                "status": "FILLED",
                "fill_time": current_time,
                "fill_price": fill_price,
                "fill_volume": fill_volume,
                "slippage": slippage,
                "partial_fill": fill_decision["partial_fill"],
                "phantom_order": fill_decision["phantom_order"]
            })
            
            # Remove from pending orders
            if order in self.pending_orders:
                self.pending_orders.remove(order)
            
            logger.info(f"Order FILLED: {order.strategy} {order.direction} @ {fill_price} (slippage: {slippage:.1f})")
            
            return FillResult(
                filled=True,
                fill_price=fill_price,
                fill_time=current_time,
                fill_volume=fill_volume,
                slippage=slippage,
                phantom_order=fill_decision["phantom_order"],
                partial_fill=fill_decision["partial_fill"]
            )
        
        else:
            # Order rejected
            order_metadata["status"] = "REJECTED"
            
            # Remove from pending orders
            if order in self.pending_orders:
                self.pending_orders.remove(order)
            
            logger.warning(f"Order REJECTED: {order.strategy} {order.direction} - {fill_decision['rejection_reason']}")
            
            return FillResult(
                filled=False,
                fill_price=0.0,
                fill_time=current_time,
                fill_volume=0.0,
                slippage=0.0,
                phantom_order=False,
                rejection_reason=fill_decision["rejection_reason"]
            )
    
    def _simulate_fill_decision(self, order: SimOrder, current_price: float, current_time: datetime) -> Dict[str, Any]:
        """Simulate realistic fill decision logic."""
        # Calculate distance to order price
        price_distance = abs(current_price - order.price)
        
        # Base fill probability by order type
        if order.order_type == "MARKET":
            # Market orders: high fill probability
            base_fill_prob = 0.95
        else:
            # Pending orders: fill probability based on price distance
            if price_distance <= 0.5:  # Very close
                base_fill_prob = 0.8
            elif price_distance <= 1.0:  # Close
                base_fill_prob = 0.6
            elif price_distance <= 2.0:  # Moderate distance
                base_fill_prob = 0.3
            else:  # Far from price
                base_fill_prob = 0.1
        
        # Adjust for volatility (higher volatility = better fill chance)
        volatility_multiplier = 1.2  # Would be calculated from ATR in real system
        adjusted_fill_prob = min(base_fill_prob * volatility_multiplier, 0.98)
        
        # Simulate fill decision
        random_fill = random.random() < adjusted_fill_prob
        
        if not random_fill:
            return {
                "fill": False,
                "rejection_reason": "PRICE_NOT_REACHED"
            }
        
        # Calculate fill price with slippage
        if random_fill:
            # Slippage in favor of broker (worse for client)
            slippage_direction = 1 if order.direction == "LONG" else -1
            slippage_amount = random.uniform(0, self.slippage_points * 1.5)  # Variable slippage
            
            fill_price = order.price + (slippage_direction * slippage_amount)
            
            # Determine partial fill probability
            partial_fill_prob = 0.05  # 5% chance of partial fill
            is_partial = random.random() < partial_fill_prob
            
            if is_partial:
                fill_volume = order.lots * random.uniform(0.5, 0.9)  # Fill 50-90%
            else:
                fill_volume = order.lots
            
            return {
                "fill": True,
                "fill_price": fill_price,
                "fill_volume": fill_volume,
                "slippage": slippage_amount,
                "partial_fill": is_partial,
                "phantom_order": False
            }
        
        return {
            "fill": False,
            "rejection_reason": "SIMULATION_REJECTED"
        }
    
    def check_phantom_order(
        self,
        order: SimOrder,
        current_time: datetime,
        verification_delay_seconds: int = 30
    ) -> bool:
        """
        Check for phantom orders (accepted but never actually filled).
        
        This simulates enhanced phantom order detection from live system.
        """
        # Find order metadata
        order_metadata = None
        for om in self.order_history:
            if om["order"] == order:
                order_metadata = om
                break
        
        if not order_metadata:
            return False
        
        # Check if order was "filled" but we want to verify it's real
        if order_metadata["status"] != "FILLED":
            return False
        
        # Check if enough time has passed for verification
        fill_time = order_metadata.get("fill_time")
        if not fill_time:
            return False
        
        elapsed = (current_time - fill_time).total_seconds()
        if elapsed < verification_delay_seconds:
            return False
        
        # Simulate phantom order detection (5% chance)
        phantom_probability = 0.05
        is_phantom = random.random() < phantom_probability
        
        if is_phantom:
            logger.critical(f"PHANTOM ORDER DETECTED: {order.strategy} {order.direction} @ {order.price}")
            order_metadata["phantom_order"] = True
            
            # In live system, this would trigger emergency shutdown
            # For backtest, we'll just log it
            return True
        
        return False
    
    def simulate_order_latency(self, order: SimOrder) -> timedelta:
        """Simulate realistic order execution latency."""
        # Market orders: lower latency
        if order.order_type == "MARKET":
            base_latency_ms = 50
        else:
            # Pending orders: higher latency
            base_latency_ms = 150
        
        # Add random variation
        latency_ms = base_latency_ms + random.randint(-20, 50)
        return timedelta(milliseconds=latency_ms)
    
    def get_pending_orders(self) -> List[SimOrder]:
        """Get all currently pending orders."""
        return self.pending_orders.copy()
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get complete order execution history."""
        return self.order_history.copy()
    
    def reset(self) -> None:
        """Reset simulator state."""
        self.pending_orders.clear()
        self.order_history.clear()
        logger.info("Execution simulator reset")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for analysis."""
        if not self.order_history:
            return {}
        
        filled_orders = [om for om in self.order_history if om["status"] == "FILLED"]
        rejected_orders = [om for om in self.order_history if om["status"] == "REJECTED"]
        expired_orders = [om for om in self.order_history if om["status"] == "EXPIRED"]
        phantom_orders = [om for om in self.order_history if om.get("phantom_order", False)]
        
        total_orders = len(self.order_history)
        
        return {
            "total_orders": total_orders,
            "filled_orders": len(filled_orders),
            "rejected_orders": len(rejected_orders),
            "expired_orders": len(expired_orders),
            "phantom_orders": len(phantom_orders),
            "fill_rate": len(filled_orders) / total_orders if total_orders > 0 else 0,
            "phantom_rate": len(phantom_orders) / total_orders if total_orders > 0 else 0,
            "avg_slippage": sum(om.get("slippage", 0) for om in filled_orders) / len(filled_orders) if filled_orders else 0,
            "partial_fill_rate": sum(1 for om in filled_orders if om.get("partial_fill", False)) / len(filled_orders) if filled_orders else 0,
        }
    
    def log_execution_summary(self) -> None:
        """Log execution summary for debugging."""
        stats = self.get_execution_stats()
        
        logger.info("=== EXECUTION SIMULATOR SUMMARY ===")
        logger.info(f"Total Orders: {stats.get('total_orders', 0)}")
        logger.info(f"Filled Orders: {stats.get('filled_orders', 0)} ({stats.get('fill_rate', 0):.1%})")
        logger.info(f"Rejected Orders: {stats.get('rejected_orders', 0)}")
        logger.info(f"Expired Orders: {stats.get('expired_orders', 0)}")
        logger.info(f"Phantom Orders: {stats.get('phantom_orders', 0)} ({stats.get('phantom_rate', 0):.1%})")
        logger.info(f"Avg Slippage: {stats.get('avg_slippage', 0):.2f} points")
        logger.info(f"Partial Fill Rate: {stats.get('partial_fill_rate', 0):.1%}")
        logger.info("=====================================")
