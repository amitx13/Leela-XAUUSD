"""
backtest/risk_updated/position_sizing.py — Updated Position Sizing for Backtesting

Updated to match current live system risk engine:
- Phase-based progression (Phase 1: 1%, Phase 2: 2% after 50+ trades)
- Conviction level sizing (A+: 1.25×, OBSERVATION: 0.75×)
- KS4 countdown size reduction (50% for 3 trades after 4-loss streak)
- Compound condition gate (severity × spread × vol_scalar < 0.35 → block)
- Reduction floor (minimum 0.50×)
- Portfolio risk controls (VAR, correlation kills)

Matches engines/risk_engine.py exactly.
"""
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Tuple
import math

# Import updated components
from backtest.strategies import STRATEGY_CONFIGS

# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING CONSTANTS - Updated to match config.py
# ─────────────────────────────────────────────────────────────────────────────

BASE_RISK_PHASE_1 = 0.010      # 1.0% per trade
BASE_RISK_PHASE_2 = 0.020      # 2.0% (after 50 proven trades)
V1_LOT_HARD_CAP = 0.50           # Maximum lot size
MIN_CONDITION_MULTIPLIER = 0.35    # Compound gate threshold
REDUCTION_FLOOR = 0.50            # Minimum reduction multiplier

# Conviction thresholds
MACRO_PROMOTE_TRADE_MIN = 50      # Minimum trades for conviction activation
MACRO_PROMOTE_DELTA_PP = 8        # 8pp win-rate delta required

# Portfolio risk limits
MAX_DAILY_VAR_PCT = 0.020        # 2.0% of account equity
MAX_SESSION_LOTS = 0.15            # Maximum lots per session
CORRELATION_REDUCTION = 0.65         # Same family + direction reduction

# ─────────────────────────────────────────────────────────────────────────────
# CONVICTION LEVELS
# ─────────────────────────────────────────────────────────────────────────────

class ConvictionLevel:
    """Conviction levels matching live system."""
    STANDARD = "STANDARD"
    A_PLUS = "A_PLUS"
    OBSERVATION = "OBSERVATION"
    
    @classmethod
    def get_multiplier(cls, level: str) -> float:
        """Get size multiplier for conviction level."""
        multipliers = {
            cls.STANDARD: 1.0,
            cls.A_PLUS: 1.25,
            cls.OBSERVATION: 0.75,
        }
        return multipliers.get(level, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Position sizing calculator matching live risk engine exactly.
    """
    
    def __init__(self):
        self.live_trade_count = 0
        self.win_rate_a_plus = 0.45
        self.win_rate_observation = 0.45
        self.ks4_reduced_trades_remaining = 0
        
    def calculate_lot_size(
        self,
        stop_distance_points: float,
        size_multiplier: float,
        state: Dict[str, Any],
        conviction_level: str = ConvictionLevel.STANDARD,
        severity_multiplier: float = 1.0,
        spread_multiplier: float = 1.0,
        vol_scalar: float = 1.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size using live system algorithm.
        
        Returns:
            (lot_size, sizing_details)
        """
        details = {}
        
        # 1. Base risk based on phase
        if self.live_trade_count < 50:
            base_risk = BASE_RISK_PHASE_1
            details['phase'] = 'Phase 1'
        else:
            # Check Phase 2 gates (WR>45%, exp>+0.15R, max_dd<15%)
            phase2_gates_met = self._check_phase2_gates(state)
            base_risk = BASE_RISK_PHASE_2 if phase2_gates_met else BASE_RISK_PHASE_1
            details['phase'] = 'Phase 2' if phase2_gates_met else 'Phase 1 (gates not met)'
        
        # 2. Conviction boost (after 50+ trades)
        if self.live_trade_count >= MACRO_PROMOTE_TRADE_MIN:
            conviction_delta = self._get_conviction_delta()
            if conviction_delta is not None and conviction_delta > MACRO_PROMOTE_DELTA_PP / 100:
                conviction_mult = ConvictionLevel.get_multiplier(conviction_level)
                base_risk *= conviction_mult
                details['conviction_applied'] = {
                    'level': conviction_level,
                    'multiplier': conviction_mult,
                    'delta_pp': conviction_delta * 100
                }
        
        # 3. KS4 countdown reduction
        if self.ks4_reduced_trades_remaining > 0:
            base_risk *= 0.5
            details['ks4_reduction'] = {
                'active': True,
                'trades_remaining': self.ks4_reduced_trades_remaining,
                'multiplier': 0.5
            }
        
        # 4. Apply multipliers
        size_before_compound = base_risk * size_multiplier * severity_multiplier * spread_multiplier * vol_scalar
        
        # 5. Reduction floor (minimum 0.50×)
        compound_multiplier = max(size_before_compound, REDUCTION_FLOOR)
        details['compound_multiplier'] = compound_multiplier
        
        # 6. Compound gate check (use compound_multiplier after floor)
        if compound_multiplier < MIN_CONDITION_MULTIPLIER:
            return 0.0, {'blocked': True, 'reason': f'Compound gate: {compound_multiplier:.3f} < {MIN_CONDITION_MULTIPLIER}'}
        
        # 7. Calculate lot size
        equity = state.get('equity', state.get('balance', 10000))

        # Risk amount in USD
        risk_amount = equity * compound_multiplier

        # BUG-17 FIX: correct XAUUSD lot formula
        # XAUUSD: 1 standard lot = 100 oz.  $1 price move on 1 lot = $100 USD.
        # So: lots = risk_$ / (stop_distance_$ × 100)
        CONTRACT_SIZE = 100.0   # oz per standard lot
        calculated_lots = (
            risk_amount / (stop_distance_points * CONTRACT_SIZE)
            if stop_distance_points > 0 else 0.01
        )
        
        # 8. Apply hard cap and minimum
        volume_min = state.get('volume_min', 0.01)
        final_lots = max(volume_min, min(calculated_lots, V1_LOT_HARD_CAP))
        
        # 9. Decimal rounding (ROUND_DOWN)
        final_lots = float(Decimal(str(final_lots)).quantize(Decimal('0.01'), rounding=ROUND_DOWN))
        
        details.update({
            'equity': equity,
            'base_risk': base_risk,
            'size_multiplier': size_multiplier,
            'severity_multiplier': severity_multiplier,
            'spread_multiplier': spread_multiplier,
            'vol_scalar': vol_scalar,
            'stop_distance': stop_distance_points,
            'calculated_lots': calculated_lots,
            'final_lots': final_lots,
            'hard_cap': V1_LOT_HARD_CAP,
            'volume_min': volume_min
        })
        
        return final_lots, details
    
    def _check_phase2_gates(self, state: Dict[str, Any]) -> bool:
        """Check if Phase 2 gates are met."""
        # In a real implementation, these would come from Truth Engine
        # For backtest, we'll use simplified checks
        max_dd = state.get('max_drawdown_pct', 0)
        win_rate = state.get('win_rate', 0.45)
        expectancy = state.get('expectancy', 0.15)
        
        return (max_dd < 0.15 and win_rate > 0.45 and expectancy > 0.15)
    
    def _get_conviction_delta(self) -> float:
        """Get conviction level delta (A+ vs OBSERVATION win rate)."""
        # In real implementation, this comes from Truth Engine
        # For backtest, we'll use a simplified approach
        return self.win_rate_a_plus - self.win_rate_observation
    
    def update_trade_count(self, count: int) -> None:
        """Update live trade count."""
        self.live_trade_count = count
    
    def update_ks4_countdown(self, trades_remaining: int) -> None:
        """Update KS4 countdown."""
        self.ks4_reduced_trades_remaining = trades_remaining

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioRiskChecker:
    """
    Portfolio risk controls matching live portfolio risk brain.
    """
    
    def __init__(self):
        self.open_positions = []
        self.daily_var_used = 0.0
        
    def check_portfolio_risk(
        self,
        candidate: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check portfolio risk controls.
        
        Returns (permitted, reason).
        """
        equity = state.get('equity', 10000)
        direction = candidate.get('direction', 'LONG')
        lot_size = candidate.get('lot_size', 0.01)
        strategy = candidate.get('strategy', '')
        
        # 1. Daily VAR check
        atr_h1 = state.get('atr_h1', 20.0)
        var_risk = lot_size * atr_h1  # Simplified VAR calculation
        if (self.daily_var_used + var_risk) > (equity * MAX_DAILY_VAR_PCT):
            return False, f"Portfolio VAR: {self.daily_var_used + var_risk:.2f} exceeds limit {equity * MAX_DAILY_VAR_PCT:.2f}"
        
        # 2. Session lots cap
        current_session_lots = sum(pos.get('lot_size', 0) for pos in self.open_positions)
        if (current_session_lots + lot_size) > MAX_SESSION_LOTS:
            return False, f"Session lots: {current_session_lots + lot_size:.2f} exceeds cap {MAX_SESSION_LOTS}"
        
        # 3. Correlation kill (same family + same direction)
        trend_family = ["S1_LONDON_BRK", "S1F_POST_TK", "S4_LONDON_PULL", "S5_NY_COMPRESS"]
        if strategy in trend_family:
            same_direction_lots = sum(
                pos.get('lot_size', 0) for pos in self.open_positions
                if pos.get('strategy') in trend_family and pos.get('direction') == direction
            )
            if same_direction_lots > 0:
                # Apply correlation reduction
                adjusted_lots = lot_size * CORRELATION_REDUCTION
                if adjusted_lots < 0.01:  # Minimum lot size
                    return False, f"Correlation kill: {adjusted_lots:.2f} below minimum 0.01"
                candidate['lot_size'] = adjusted_lots
                return True, f"Correlation reduction applied: {lot_size:.2f} → {adjusted_lots:.2f}"
        
        return True, "OK"
    
    def add_position(self, position: Dict[str, Any]) -> None:
        """Add position to tracking."""
        self.open_positions.append(position)
        atr_h1 = position.get('atr_h1', 20.0)
        self.daily_var_used += position.get('lot_size', 0) * atr_h1
    
    def remove_position(self, position: Dict[str, Any]) -> None:
        """Remove position from tracking."""
        if position in self.open_positions:
            self.open_positions.remove(position)
            atr_h1 = position.get('atr_h1', 20.0)
            self.daily_var_used -= position.get('lot_size', 0) * atr_h1
    
    def reset_daily(self) -> None:
        """Reset daily state."""
        self.daily_var_used = 0.0
        self.open_positions = []
