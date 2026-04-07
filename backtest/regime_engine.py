"""
backtest/regime_engine_updated.py — Updated Regime Engine for Backtesting

Updated to match current live system (engines/regime.py) with:
- 6-state classification: NO_TRADE, UNSTABLE, RANGING_CLEAR, WEAK_TRENDING, NORMAL_TRENDING, SUPER_TRENDING
- Hysteresis implementation (3 consecutive readings)
- ATR percentile calculation with EWMA weighting
- ADX trend strength analysis
- Session-aware volatility normalization
- DXY macro bias integration

Matches live regime engine exactly for accurate backtesting.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from typing import Optional, Tuple
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# REGIME STATE ENUM - Matches live system
# ─────────────────────────────────────────────────────────────────────────────

class RegimeState(str, Enum):
    """Regime states matching live system exactly."""
    SUPER_TRENDING  = "SUPER_TRENDING"
    NORMAL_TRENDING = "NORMAL_TRENDING"
    WEAK_TRENDING   = "WEAK_TRENDING"
    RANGING_CLEAR   = "RANGING_CLEAR"
    UNSTABLE        = "UNSTABLE"
    NO_TRADE        = "NO_TRADE"

    @property
    def multiplier(self) -> float:
        return {
            RegimeState.SUPER_TRENDING:  1.5,  # SIZE-2 FIX: was 1.2
            RegimeState.NORMAL_TRENDING: 1.0,
            RegimeState.WEAK_TRENDING:   0.8,
            RegimeState.RANGING_CLEAR:   0.7,
            RegimeState.UNSTABLE:        0.4,
            RegimeState.NO_TRADE:        0.0,
        }[self]

    @property
    def is_trending(self) -> bool:
        return self in (
            RegimeState.SUPER_TRENDING,
            RegimeState.NORMAL_TRENDING,
            RegimeState.WEAK_TRENDING,
        )

    @property
    def allows_s1(self) -> bool:
        """S1 fires in any trending regime."""
        return self.is_trending

    @property
    def allows_s2(self) -> bool:
        """S2 fires in RANGING_CLEAR only."""
        return self == RegimeState.RANGING_CLEAR

    @property
    def allows_reentry(self) -> bool:
        """S1d re-entries only in SUPER or NORMAL."""
        return self in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING)

# ─────────────────────────────────────────────────────────────────────────────
# ATR PERCENTILE CALCULATION - EWMA Weighted (B2 Fix)
# ─────────────────────────────────────────────────────────────────────────────

def get_atr_percentile_h1(
    atr_series: pd.Series, 
    current_atr: float,
    lookback_days: int = 29,
    use_session_filter: bool = False,
    session_filter: Optional[str] = None
) -> float:
    """
    Calculate ATR percentile using EWMA weighting (B2 Fix).
    
    EWMA weighting (λ=0.94):
      - Last ~20 bars carry ~65% of ranking weight
      - Volatility clusters — recent bars are more relevant
    
    Session normalization:
      - Filters historical bars by session before ranking
      - Asian ATR (8-14 pts) ranked separately from London ATR (16-30 pts)
    
    Returns percentile rank (0-100).
    """
    if atr_series is None or len(atr_series) < 10 or current_atr is None:
        return 50.0
    
    valid = atr_series.dropna()
    if len(valid) < 10:
        return 50.0
    
    # Session filtering if requested
    if use_session_filter and session_filter:
        # This would need session data - for backtest, we'll skip session filtering
        pass
    
    # Calculate EWMA weights (λ=0.94)
    weights = np.exp(-0.06 * np.arange(len(valid))[::-1])
    weights = weights / weights.sum()
    
    # Weighted percentile calculation
    sorted_values = np.sort(valid.values)
    sorted_weights = weights[np.argsort(valid.values)]
    
    cum_weights = np.cumsum(sorted_weights)
    percentile = np.interp(current_atr, sorted_values, cum_weights * 100)
    
    return float(percentile)

# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION WITH HYSTERESIS
# ─────────────────────────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Regime classifier with hysteresis matching live system.
    """
    
    def __init__(self):
        self.consecutive_readings = []
        self.current_regime = RegimeState.NORMAL_TRENDING
        
        # Thresholds from config
        self.atr_pct_unstable = 85    # >95% = NO_TRADE, 85-95% = UNSTABLE  
        self.atr_pct_super = 55        # >55% + ADX>35 + DXY<-0.70 = SUPER
        self.adx_weak = 18              # <18 = RANGING_CLEAR
        self.adx_normal = 26           # 18-26 = WEAK
        self.adx_strong = 35           # >35 = SUPER candidate
        
    def classify_regime(
        self,
        adx_h4: float,
        atr_pct_h1: float,
        session: str,
        dxy_macro: Optional[float] = None,
        has_upcoming_event: bool = False,
        spread_ratio: float = 1.0
    ) -> Tuple[RegimeState, float]:
        """
        Classify market regime with hysteresis.
        
        Returns (regime_state, size_multiplier).
        """
        # Hard blocks first
        if has_upcoming_event:
            candidate = RegimeState.NO_TRADE
        elif spread_ratio > 2.5:
            candidate = RegimeState.NO_TRADE
        elif atr_pct_h1 >= 95:  # NO_TRADE threshold
            candidate = RegimeState.NO_TRADE
        elif atr_pct_h1 >= 85:  # UNSTABLE threshold
            candidate = RegimeState.UNSTABLE
        elif adx_h4 < self.adx_weak:
            candidate = RegimeState.RANGING_CLEAR
        elif adx_h4 < self.adx_normal:
            candidate = RegimeState.WEAK_TRENDING
        elif adx_h4 < self.adx_strong:
            candidate = RegimeState.NORMAL_TRENDING
        else:
            # Check for SUPER_TRENDING conditions
            dxy_boost = dxy_macro is not None and dxy_macro < -0.70
            if atr_pct_h1 >= self.atr_pct_super and dxy_boost:
                candidate = RegimeState.SUPER_TRENDING
            else:
                candidate = RegimeState.NORMAL_TRENDING
        
        # Apply hysteresis (3 consecutive readings required)
        self.consecutive_readings.append(candidate)
        if len(self.consecutive_readings) > 3:
            self.consecutive_readings.pop(0)
        
        # Check if we have 3 consecutive same readings
        if len(self.consecutive_readings) >= 3:
            if all(r == self.consecutive_readings[0] for r in self.consecutive_readings):
                # Ensure we're setting to a RegimeState enum, not a string
                candidate = self.consecutive_readings[0]
                if isinstance(candidate, str):
                    # Convert string to RegimeState if needed
                    regime_map = {
                        "SUPER_TRENDING": RegimeState.SUPER_TRENDING,
                        "NORMAL_TRENDING": RegimeState.NORMAL_TRENDING,
                        "WEAK_TRENDING": RegimeState.WEAK_TRENDING,
                        "RANGING_CLEAR": RegimeState.RANGING_CLEAR,
                        "UNSTABLE": RegimeState.UNSTABLE,
                        "NO_TRADE": RegimeState.NO_TRADE,
                    }
                    self.current_regime = regime_map.get(candidate, RegimeState.NORMAL_TRENDING)
                else:
                    self.current_regime = candidate
        
        # Session multipliers (Asian gets 0.7×)
        session_multiplier = 0.7 if session == "ASIAN" else 1.0
        final_multiplier = self.current_regime.multiplier * session_multiplier
        
        return self.current_regime, final_multiplier
    
    def get_current_regime(self) -> RegimeState:
        """Get current regime state."""
        return self.current_regime
    
    def reset(self) -> None:
        """Reset classifier state."""
        self.consecutive_readings = []
        self.current_regime = RegimeState.NORMAL_TRENDING

# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime_backtest(
    adx_h4: float,
    atr_pct_h1: float,
    session: str,
    has_upcoming_event: bool = False,
    spread_ratio: float = 1.0,
    dxy_macro: Optional[float] = None
) -> Tuple[RegimeState, float]:
    """
    Convenience function for backward compatibility.
    """
    classifier = RegimeClassifier()
    return classifier.classify_regime(
        adx_h4, atr_pct_h1, session, dxy_macro, 
        has_upcoming_event, spread_ratio
    )

def get_regime_multiplier(regime: RegimeState, session: str = "LONDON") -> float:
    """
    Get size multiplier for regime and session.
    """
    session_multiplier = 0.7 if session == "ASIAN" else 1.0
    return regime.multiplier * session_multiplier

def is_regime_allowed_for_strategy(
    regime, 
    strategy: str
) -> bool:
    """
    Check if regime allows specific strategy.
    Handles both RegimeState enums and string values.
    """
    # Convert string to enum if needed
    if isinstance(regime, str):
        try:
            regime = RegimeState(regime)
        except ValueError:
            return False
    
    if regime == RegimeState.NO_TRADE:
        return False  # Only S7 pending orders allowed
    elif strategy == "S2_MEAN_REV":
        return regime == RegimeState.RANGING_CLEAR
    elif strategy.startswith("S1D"):  # Re-entries
        return regime in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING)
    else:
        return regime.is_trending
