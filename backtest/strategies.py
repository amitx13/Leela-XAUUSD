"""
backtest/strategies_implemented.py — Complete Strategy Implementations for Backtesting

Updated implementations for all 10 strategies matching live system:
- S1 family with volume filters and ATR-based stops
- S2 mean reversion with RSI confirmation
- S3 stop hunt reversal with dynamic range
- S6/S7 with ADX trend filtering
- Phase 2 strategies: R3, S4, S5, S8 with independent lane logic
- All strategies use current parameters and thresholds

Matches engines/signal_engine.py and engines/signal_engine_phase2.py exactly.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from backtest.regime_engine import RegimeState, is_regime_allowed_for_strategy
from backtest.execution_simulator import POINT_VALUE, COMMISSION_PER_LOT_RT
from backtest.indicators import calculate_atr, calculate_rsi, calculate_ema


# Strategy definitions
ALL_STRATEGIES = [
    'S1_LONDON_BRK', 'S1B_FAILED_BRK', 'S1D_PYRAMID', 'S1E_PYRAMID', 'S1F_POST_TK',
    'S2_MEAN_REV', 'S3_STOP_HUNT_REV', 'S6_ASIAN_BRK', 'S7_DAILY_STRUCT',
    'R3_CAL_MOMENTUM', 'S4_LONDON_PULL', 'S5_NY_COMPRESS', 'S8_ATR_SPIKE'
]

# Strategy configurations (simplified for backtest)
STRATEGY_CONFIGS = {
    'S1_LONDON_BRK': {'max_attempts': 3, 'min_range': 10, 'breakout_pct': 0.12, 'stop_atr_mult': 1.5, 'stop_min_points': 15, 'tp_r_multiple': 2.5},
    'S1B_FAILED_BRK': {'reversal_distance': 20},
    'S1D_PYRAMID': {'max_reentries': {'SUPER_TRENDING': 8, 'NORMAL_TRENDING': 5}, 'stop_atr_mult': 1.0, 'stop_min_points': 15, 'expiry_minutes': 5},
    'S1E_PYRAMID': {'size_multiplier': 0.5},
    'S1F_POST_TK': {'expiry_minutes': 30},
    'S2_MEAN_REV': {'atr_mult': 1.5, 'rsi_overbought': 70, 'rsi_oversold': 30},
    'S3_STOP_HUNT_REV': {'sessions': ['LONDON', 'LONDON_NY_OVERLAP', 'NY'], 'sweep_atr_mult': 0.3, 'stop_atr_mult': 0.5},
    'S6_ASIAN_BRK': {'min_range': 8, 'breakout_pct': 0.1, 'expiry_utc': '08:05'},
    'S7_DAILY_STRUCT': {'min_range_atr_ratio': 0.75, 'breakout_points': 5},
    'R3_CAL_MOMENTUM': {'volatility_atr_mult': 0.3, 'max_hold_minutes': 30},
    'S4_LONDON_PULL': {'start_utc': '08:00', 'end_utc': '16:00', 'adx_min': 25, 'expiry_minutes': 30},
    'S5_NY_COMPRESS': {'start_utc': '13:00', 'end_utc': '22:00', 'breakout_points': 5},
    'S8_ATR_SPIKE': {'spike_atr_mult': 1.5}
}

# Strategy helper functions
def get_strategy_family(strategy: str) -> str:
    trend_family = ['S1_LONDON_BRK', 'S1F_POST_TK', 'S4_LONDON_PULL', 'S5_NY_COMPRESS']
    reversal_family = ['S1B_FAILED_BRK', 'S2_MEAN_REV', 'S3_STOP_HUNT_REV']
    independent = ['R3_CAL_MOMENTUM', 'S8_ATR_SPIKE']
    oco_pairs = ['S6_ASIAN_BRK', 'S7_DAILY_STRUCT']
    
    if strategy in trend_family:
        return 'trend'
    elif strategy in reversal_family:
        return 'reversal'
    elif strategy in independent:
        return 'independent'
    elif strategy in oco_pairs:
        return 'oco_pairs'
    else:
        return 'pyramid'

def is_phase2_strategy(strategy: str) -> bool:
    return strategy in ['R3_CAL_MOMENTUM', 'S4_LONDON_PULL', 'S5_NY_COMPRESS', 'S8_ATR_SPIKE']

def can_coexist_with_trend(strategy: str) -> bool:
    return strategy in ['R3_CAL_MOMENTUM', 'S8_ATR_SPIKE']

# Strategy display names and groups
STRATEGY_REGISTRY = {
    'S1_LONDON_BRK': 'S1 London Breakout',
    'S1B_FAILED_BRK': 'S1B Failed Breakout',
    'S1D_PYRAMID': 'S1D Pyramid',
    'S1E_PYRAMID': 'S1E Pyramid',
    'S1F_POST_TK': 'S1F Post-TK',
    'S2_MEAN_REV': 'S2 Mean Reversion',
    'S3_STOP_HUNT_REV': 'S3 Stop Hunt Reversal',
    'S6_ASIAN_BRK': 'S6 Asian Breakout',
    'S7_DAILY_STRUCT': 'S7 Daily Structure',
    'R3_CAL_MOMENTUM': 'R3 Calendar Momentum',
    'S4_LONDON_PULL': 'S4 London Pullback',
    'S5_NY_COMPRESS': 'S5 NY Compression',
    'S8_ATR_SPIKE': 'S8 ATR Spike'
}

STRATEGY_GROUPS = {
    'trend_family': ['S1_LONDON_BRK', 'S1F_POST_TK', 'S4_LONDON_PULL', 'S5_NY_COMPRESS'],
    'mean_reversion': ['S2_MEAN_REV'],
    'pattern': ['S3_STOP_HUNT_REV'],
    'pullback': ['S4_LONDON_PULL', 'S5_NY_COMPRESS'],
    'oco_pairs': ['S6_ASIAN_BRK', 'S7_DAILY_STRUCT'],
    'independent': ['R3_CAL_MOMENTUM', 'S8_ATR_SPIKE'],
    'phase1': ['S1_LONDON_BRK', 'S1B_FAILED_BRK', 'S1D_PYRAMID', 'S1E_PYRAMID', 'S1F_POST_TK', 'S2_MEAN_REV', 'S3_STOP_HUNT_REV', 'S6_ASIAN_BRK', 'S7_DAILY_STRUCT'],
    'phase2': ['R3_CAL_MOMENTUM', 'S4_LONDON_PULL', 'S5_NY_COMPRESS', 'S8_ATR_SPIKE']
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Indicators are imported from backtest.indicators

def check_volume_filter(bar: Dict[str, Any], window_bars: List[Dict[str, Any]]) -> bool:
    """
    Volume filter for S1 breakouts.
    Rejects breakouts with tick_volume < 70% of 5-bar average.
    """
    if len(window_bars) < 5:
        return True  # Pass filter if not enough data
    
    recent_volumes = [bar.get('tick_volume', 0) for bar in window_bars[-5:]]
    avg_volume = np.mean(recent_volumes)
    current_volume = bar.get('tick_volume', 0)
    
    return current_volume >= (0.7 * avg_volume)

def check_adx_trend_filter(adx_h4: float, di_plus: float, di_minus: float, direction: str) -> bool:
    """
    ADX trend filter for S6/S7.
    In strong trends (ADX>25, DI ratio>1.3×), only trending direction allowed.
    """
    if adx_h4 < 25:
        return True  # No filter in weak trends
    
    di_ratio = max(di_plus, di_minus) / max(min(di_plus, di_minus), 0.001)
    if di_ratio < 1.3:
        return True  # No filter if trend not strong enough
    
    # Check if requested direction matches trend
    if direction == "LONG":
        return di_plus > di_minus
    else:  # SHORT
        return di_minus > di_plus

# ─────────────────────────────────────────────────────────────────────────────
# S1 FAMILY STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_s1_london_brk(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S1_LONDON_BRK - Primary London Breakout.
    """
    config = STRATEGY_CONFIGS["S1_LONDON_BRK"]

    # Check regime
    regime = state.get('regime', RegimeState.NORMAL_TRENDING)
    if not is_regime_allowed_for_strategy(regime, "S1_LONDON_BRK"):
        return None

    # BUG-13 FIX: 'session' key is aliased in to_dict()
    session = state.get('session', 'OFF_HOURS')
    if session not in ['LONDON', 'LONDON_NY_OVERLAP']:
        return None

    if state.get('s1_attempts_today', 0) >= config['max_attempts']:
        return None
    if state.get('trend_family_occupied'):
        return None

    pre_london_range = state.get('pre_london_range', {})
    if not pre_london_range:
        return None

    range_size = pre_london_range.get('high', 0) - pre_london_range.get('low', 0)
    if range_size < config['min_range']:
        return None

    breakout_distance = range_size * config['breakout_pct']
    buy_level  = pre_london_range['high'] + breakout_distance
    sell_level = pre_london_range['low']  - breakout_distance

    recent_bars = context.get('recent_m5_bars', [])
    if not check_volume_filter(bar, recent_bars):
        return None

    # BUG-1 FIX: guard atr_h1 against None before multiplication
    atr_h1 = context.get('atr_h1') or 20.0

    current_price = bar['close']
    if current_price > pre_london_range.get('high', 0):
        direction   = "LONG"
        entry_price = buy_level
        stop_price  = pre_london_range['low'] - max(
            atr_h1 * config['stop_atr_mult'], config['stop_min_points']
        )
    elif current_price < pre_london_range.get('low', 0):
        direction   = "SHORT"
        entry_price = sell_level
        stop_price  = pre_london_range['high'] + max(
            atr_h1 * config['stop_atr_mult'], config['stop_min_points']
        )
    else:
        return None

    risk_points = abs(entry_price - stop_price)
    tp_price = (
        entry_price + risk_points * config['tp_r_multiple']
        if direction == "LONG"
        else entry_price - risk_points * config['tp_r_multiple']
    )

    return {
        'strategy': 'S1_LONDON_BRK',
        'direction': direction,
        'entry_type': 'BUY_STOP' if direction == 'LONG' else 'SELL_STOP',
        'entry_price': entry_price,
        'stop_price': stop_price,
        'tp_price': tp_price,
        'lot_size': 0.0,  # Will be calculated by risk engine
        'expiry': None,
        'tag': 's1_main',
        'metadata': {
            'range_size': range_size,
            'breakout_distance': breakout_distance,
            'volume_passed': True
        }
    }

def evaluate_s1b_failed_brk(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S1B_FAILED_BRK - Failed Breakout Reversal.
    """
    # Check for recent S1 that hit stop loss
    last_s1_trade = state.get('last_s1_trade', {})
    if not last_s1_trade:
        return None
    
    # Must be a failed S1 (R < 0.5)
    if last_s1_trade.get('r_multiple', 1.0) >= 0.5:
        return None
    
    # Check if reversal already triggered today
    if state.get('s1b_fired_today'):
        return None
    
    # Check if reversal family occupied
    if state.get('reversal_family_occupied'):
        return None
    
    # Determine reversal direction (opposite of failed S1)
    failed_direction = last_s1_trade.get('direction')
    reversal_direction = "SHORT" if failed_direction == "LONG" else "LONG"
    
    # Entry beyond failed extreme
    failed_extreme = last_s1_trade.get('stop_price')
    reversal_distance = 20  # Points beyond failed extreme
    
    if reversal_direction == "LONG":
        entry_price = failed_extreme + reversal_distance
        stop_price = failed_extreme - 20  # 20 points below failed extreme
    else:  # SHORT
        entry_price = failed_extreme - reversal_distance
        stop_price = failed_extreme + 20  # 20 points above failed extreme
    
    return {
        'strategy': 'S1B_FAILED_BRK',
        'direction': reversal_direction,
        'entry_type': 'STOP',
        'entry_price': entry_price,
        'stop_price': stop_price,
        'tp_price': None,  # No TP for reversal
        'lot_size': 0.0,
        'expiry': None,
        'tag': 's1b_reversal',
        'metadata': {
            'failed_s1_ticket': last_s1_trade.get('ticket'),
            'failed_direction': failed_direction
        }
    }

def evaluate_s1d_pyramid(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S1D_PYRAMID - M5 Pullback Re-entry.
    """
    config = STRATEGY_CONFIGS["S1D_PYRAMID"]

    open_s1_position = state.get('open_s1_position')
    if not open_s1_position:
        return None

    regime = state.get('regime', RegimeState.NORMAL_TRENDING)
    if regime not in [RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING]:
        return None

    reentries_today = state.get('s1d_reentries_today', 0)
    max_reentries = config['max_reentries'].get(str(regime), 5)
    if reentries_today >= max_reentries:
        return None

    # BUG-1 FIX: guard ema20_m5 and atr_m15 against None
    ema20_m5 = context.get('ema20_m5')
    if ema20_m5 is None:
        return None
    atr_m15 = context.get('atr_m15') or 15.0

    body_top    = max(bar['open'], bar['close'])
    body_bottom = min(bar['open'], bar['close'])

    s1_direction = open_s1_position.get('direction')
    if s1_direction == "LONG":
        if body_bottom > ema20_m5:
            entry_price = ema20_m5
            stop_price  = bar['low'] - max(atr_m15 * config['stop_atr_mult'],
                                           config['stop_min_points'])
        else:
            return None
    else:
        if body_top < ema20_m5:
            entry_price = ema20_m5
            stop_price  = bar['high'] + max(atr_m15 * config['stop_atr_mult'],
                                            config['stop_min_points'])
        else:
            return None

    return {
        'strategy': 'S1D_PYRAMID',
        'direction': s1_direction,
        'entry_type': 'LIMIT',
        'entry_price': entry_price,
        'stop_price': stop_price,
        'tp_price': None,
        'lot_size': 0.0,
        'expiry': bar['time'] + timedelta(minutes=config['expiry_minutes']),
        'tag': 's1d_addon',
        'metadata': {
            'parent_s1_ticket': open_s1_position.get('ticket'),
            'pullback_to_ema': True,
        },
    }

def evaluate_s1e_pyramid(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S1E_PYRAMID - Confirmed Winner Pyramid Add.
    """
    # Check for S1 position with partial exit done and BE activated
    open_s1_position = state.get('open_s1_position')
    if not open_s1_position:
        return None
    
    if not (open_s1_position.get('partial_exit_done') and open_s1_position.get('be_activated')):
        return None
    
    # Add-on size
    original_lots = open_s1_position.get('lot_size', 0.01)
    addon_lots = original_lots * STRATEGY_CONFIGS["S1E_PYRAMID"]['size_multiplier']
    
    # Market order at current price
    current_price = bar['close']
    direction = open_s1_position.get('direction')
    
    return {
        'strategy': 'S1E_PYRAMID',
        'direction': direction,
        'entry_type': 'MARKET',
        'entry_price': current_price,
        'stop_price': open_s1_position.get('stop_price'),
        'tp_price': None,
        'lot_size': addon_lots,
        'expiry': None,
        'tag': 's1e_addon',
        'metadata': {
            'parent_s1_ticket': open_s1_position.get('ticket'),
            'original_lots': original_lots
        }
    }

def evaluate_s1f_post_tk(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S1F_POST_TK - Post Time-Kill Re-entry.
    """
    config = STRATEGY_CONFIGS["S1F_POST_TK"]

    # BUG-13 FIX: 'session' alias available via to_dict()
    session = state.get('session', 'OFF_HOURS')
    if session != 'NY':
        return None
    if state.get('s1f_fired_today') or state.get('s1f_reentered_today'):
        return None

    last_s1_direction = state.get('last_s1_direction')
    # BUG-1 FIX: guard ema20_h1 and ema20_m5 against None
    ema20_h1 = context.get('ema20_h1')
    ema20_m5 = context.get('ema20_m5')
    if ema20_h1 is None or ema20_m5 is None or not last_s1_direction:
        return None

    if last_s1_direction == "LONG" and bar['close'] < ema20_h1:
        return None
    if last_s1_direction == "SHORT" and bar['close'] > ema20_h1:
        return None

    return {
        'strategy': 'S1F_POST_TK',
        'direction': last_s1_direction,
        'entry_type': 'LIMIT',
        'entry_price': ema20_m5,
        'stop_price': ema20_m5 - 15 if last_s1_direction == "LONG" else ema20_m5 + 15,
        'tp_price': None,
        'lot_size': 0.0,
        'expiry': bar['time'] + timedelta(minutes=config['expiry_minutes']),
        'tag': 's1f_reentry',
        'metadata': {
            'last_s1_direction': last_s1_direction,
            'direction_validated': True,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# S2 MEAN REVERSION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_s2_mean_rev(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S2_MEAN_REV - Range Reversion.
    """
    config = STRATEGY_CONFIGS["S2_MEAN_REV"]

    regime = state.get('regime', RegimeState.NORMAL_TRENDING)
    if regime != RegimeState.RANGING_CLEAR:
        return None
    if state.get('s2_fired_today'):
        return None

    # BUG-1 FIX: explicit None checks (not falsy — 0.0 is valid for RSI-like values)
    ema20_h1 = context.get('ema20_h1')
    rsi_h1   = context.get('rsi_h1')
    atr_h1   = context.get('atr_h1')
    if ema20_h1 is None or rsi_h1 is None or atr_h1 is None:
        return None

    distance_from_ema = abs(bar['close'] - ema20_h1)
    atr_distance = atr_h1 * config['atr_mult']

    if distance_from_ema < atr_distance:
        return None

    if bar['close'] > ema20_h1:
        if rsi_h1 < config['rsi_overbought']:
            return None
        direction = "SHORT"
        stop_price = ema20_h1 + atr_distance
    else:
        if rsi_h1 > config['rsi_oversold']:
            return None
        direction = "LONG"
        stop_price = ema20_h1 - atr_distance

    return {
        'strategy': 'S2_MEAN_REV',
        'direction': direction,
        'entry_type': 'LIMIT',
        'entry_price': ema20_h1,
        'stop_price': stop_price,
        'tp_price': None,
        'lot_size': 0.0,
        'expiry': None,
        'tag': 's2_mean_rev',
        'metadata': {
            'distance_from_ema': distance_from_ema,
            'atr_distance': atr_distance,
            'rsi_confirmation': rsi_h1,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# S3 STOP HUNT REVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_s3_stop_hunt_rev(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S3_STOP_HUNT_REV - Liquidity Sweep Reversal.
    """
    config = STRATEGY_CONFIGS["S3_STOP_HUNT_REV"]

    # BUG-13 FIX: use 'session' alias
    session = state.get('session', 'OFF_HOURS')
    if session not in config['sessions']:
        return None
    if state.get('s3_fired_today'):
        return None

    recent_m15 = context.get('recent_m15_bars', [])
    if len(recent_m15) < 12:
        return None

    range_bars = recent_m15[-12:]
    range_high = max(b['high'] for b in range_bars)
    range_low  = min(b['low']  for b in range_bars)

    # BUG-1 FIX: guard atr_h1 against None
    atr_h1 = context.get('atr_h1') or 20.0
    sweep_threshold = atr_h1 * config['sweep_atr_mult']

    current_price = bar['close']
    if current_price > range_high + sweep_threshold:
        if current_price < range_low:
            entry_price = range_high + 2
            stop_price  = range_low - (atr_h1 * 0.5)
            direction   = "LONG"
        else:
            return None
    elif current_price < range_low - sweep_threshold:
        if current_price > range_high:
            entry_price = range_low - 2
            stop_price  = range_high + (atr_h1 * 0.5)
            direction   = "SHORT"
        else:
            return None
    else:
        return None

    return {
        'strategy': 'S3_STOP_HUNT_REV',
        'direction': direction,
        'entry_type': 'BUY_STOP' if direction == 'LONG' else 'SELL_STOP',
        'entry_price': entry_price,
        'stop_price': stop_price,
        'tp_price': None,
        'lot_size': 0.0,
        'expiry': None,
        'tag': 's3_reversal',
        'metadata': {
            'range_high': range_high,
            'range_low': range_low,
            'sweep_distance': abs(current_price - (range_high if direction == 'LONG' else range_low)),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# S6/S7 OCO PAIRS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_s6_asian_brk(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """
    S6_ASIAN_BRK - Asian Range Breakout OCO pair.
    """
    config = STRATEGY_CONFIGS["S6_ASIAN_BRK"]

    if state.get('s6_placed_today'):
        return None

    current_time = bar['time']
    setup_end = current_time.replace(
        hour=int(config['expiry_utc'].split(':')[0]),
        minute=int(config['expiry_utc'].split(':')[1]),
        second=0, microsecond=0,
    )
    if current_time > setup_end:
        return None

    asian_range = state.get('asian_range', {})
    if not asian_range:
        return None
    if asian_range.get('high', 0) - asian_range.get('low', 0) < config['min_range']:
        return None

    adx_h4   = context.get('adx_h4',  20.0)
    di_plus  = context.get('di_plus',  0.0)
    di_minus = context.get('di_minus', 0.0)

    # BUG-1 FIX: guard atr_h1 against None
    atr_h1 = context.get('atr_h1') or 20.0

    breakout_distance = (asian_range['high'] - asian_range['low']) * config['breakout_pct']
    buy_level  = asian_range['high'] + breakout_distance
    sell_level = asian_range['low']  - breakout_distance

    orders = []
    if check_adx_trend_filter(adx_h4, di_plus, di_minus, "LONG"):
        orders.append({
            'strategy': 'S6_ASIAN_BRK', 'direction': 'LONG',
            'entry_type': 'BUY_STOP', 'entry_price': buy_level,
            'stop_price': asian_range['low'] - (atr_h1 * 0.5),
            'tp_price': None, 'lot_size': 0.0, 'expiry': setup_end,
            'tag': 's6_buy', 'linked_tag': 's6_sell',
            'metadata': {'trend_filtered': True},
        })
    if check_adx_trend_filter(adx_h4, di_plus, di_minus, "SHORT"):
        orders.append({
            'strategy': 'S6_ASIAN_BRK', 'direction': 'SHORT',
            'entry_type': 'SELL_STOP', 'entry_price': sell_level,
            'stop_price': asian_range['high'] + (atr_h1 * 0.5),
            'tp_price': None, 'lot_size': 0.0, 'expiry': setup_end,
            'tag': 's6_sell', 'linked_tag': 's6_buy',
            'metadata': {'trend_filtered': True},
        })

    return orders if orders else None

def evaluate_s7_daily_struct(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """
    S7_DAILY_STRUCT - Daily Structure Breakout OCO pair.
    """
    config = STRATEGY_CONFIGS["S7_DAILY_STRUCT"]

    if state.get('s7_placed_today'):
        return None

    prev_day = state.get('prev_day_ohlc', {})
    if not prev_day:
        return None

    # BUG-1 FIX: guard atr_h1 against None; use it as daily_atr proxy
    atr_h1 = context.get('atr_h1') or 20.0
    # daily_atr is roughly 5× H1 ATR for XAUUSD; use atr_h1 × 5 as proxy
    daily_atr = context.get('daily_atr', atr_h1 * 5)
    if (prev_day.get('high', 0) - prev_day.get('low', 0)) < (daily_atr * config['min_range_atr_ratio']):
        return None

    adx_h4   = context.get('adx_h4',  20.0)
    di_plus  = context.get('di_plus',  0.0)
    di_minus = context.get('di_minus', 0.0)

    buy_level  = prev_day['high'] + config['breakout_points']
    sell_level = prev_day['low']  - config['breakout_points']

    orders = []
    if check_adx_trend_filter(adx_h4, di_plus, di_minus, "LONG"):
        orders.append({
            'strategy': 'S7_DAILY_STRUCT', 'direction': 'LONG',
            'entry_type': 'BUY_STOP', 'entry_price': buy_level,
            'stop_price': prev_day['low'] - (atr_h1 * 0.5),
            'tp_price': None, 'lot_size': 0.0, 'expiry': None,
            'tag': 's7_buy', 'linked_tag': 's7_sell',
            'metadata': {'trend_filtered': True},
        })
    if check_adx_trend_filter(adx_h4, di_plus, di_minus, "SHORT"):
        orders.append({
            'strategy': 'S7_DAILY_STRUCT', 'direction': 'SHORT',
            'entry_type': 'SELL_STOP', 'entry_price': sell_level,
            'stop_price': prev_day['high'] + (atr_h1 * 0.5),
            'tp_price': None, 'lot_size': 0.0, 'expiry': None,
            'tag': 's7_sell', 'linked_tag': 's7_buy',
            'metadata': {'trend_filtered': True},
        })

    return orders if orders else None

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 STRATEGIES (INDEPENDENT LANES)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_r3_cal_momentum(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    R3_CAL_MOMENTUM - Economic Calendar Momentum.
    """
    config = STRATEGY_CONFIGS["R3_CAL_MOMENTUM"]
    
    # Check if already fired today
    if state.get('r3_fired_today'):
        return None
    
    # Check for recent high-impact event (5-35 minutes ago)
    recent_events = state.get('recent_events', [])
    qualifying_event = None
    
    for event in recent_events:
        event_time = event.get('time')
        if not event_time:
            continue
        
        time_diff = (bar['time'] - event_time).total_seconds() / 60
        if 5 <= time_diff <= 35:
            if event.get('impact_level') == 'HIGH':
                qualifying_event = event
                break
    
    if not qualifying_event:
        return None
    
    # Volatility filter - post-event move must exceed 0.3×H1 ATR
    pre_event_price = state.get('r3_pre_event_price', 0)
    post_event_move = abs(bar['close'] - pre_event_price)
    atr_h1 = context.get('atr_h1', 20)
    
    if post_event_move < (atr_h1 * config['volatility_atr_mult']):
        return None
    
    # Market order in direction of first M5 close
    direction = "LONG" if bar['close'] > pre_event_price else "SHORT"
    
    return {
        'strategy': 'R3_CAL_MOMENTUM',
        'direction': direction,
        'entry_type': 'MARKET',
        'entry_price': bar['close'],
        'stop_price': bar['close'] - (atr_h1 * 0.5) if direction == "LONG" else bar['close'] + (atr_h1 * 0.5),
        'tp_price': bar['close'] + (atr_h1 * 0.75) if direction == "LONG" else bar['close'] - (atr_h1 * 0.75),
        'lot_size': 0.0,
        'expiry': bar['time'] + timedelta(minutes=config['max_hold_minutes']),
        'tag': 'r3_momentum',
        'metadata': {
            'event_name': qualifying_event.get('name'),
            'event_time': qualifying_event.get('time'),
            'post_event_move': post_event_move,
            'independent_lane': True
        }
    }

def evaluate_s4_london_pull(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S4_LONDON_PULL - London Pullback Continuation.
    """
    config = STRATEGY_CONFIGS["S4_LONDON_PULL"]
    
    # Check session
    current_time = bar['time']
    start_utc = current_time.replace(
        hour=int(config['start_utc'].split(':')[0]),
        minute=int(config['start_utc'].split(':')[1]),
        second=0
    )
    end_utc = current_time.replace(
        hour=int(config['end_utc'].split(':')[0]),
        minute=int(config['end_utc'].split(':')[1]),
        second=0
    )
    
    if not (start_utc <= current_time <= end_utc):
        return None
    
    # Check if already fired today
    if state.get('s4_fired_today'):
        return None
    
    # Check regime (trending with increasing ADX)
    adx_h4 = context.get('adx_h4', 20)
    adx_increasing = context.get('adx_increasing', False)
    di_plus_h4 = context.get('di_plus_h4', 0)
    di_minus_h4 = context.get('di_minus_h4', 0)
    
    if adx_h4 < config['adx_min'] or not adx_increasing:
        return None
    
    # Determine direction based on ADX/DI trend
    if di_plus_h4 > di_minus_h4:
        direction = "LONG"
    else:
        direction = "SHORT"
    
    # EMA20 touch check
    ema20_m15 = context.get('ema20_m15')
    if not ema20_m15:
        return None
    
    # Check for EMA20 touch
    if not state.get('s4_ema_touched'):
        # Check if current bar touches EMA20
        if bar['low'] <= ema20_m15 <= bar['high']:
            state['s4_ema_touched'] = True
        else:
            return None
    
    # LIMIT order at EMA20
    return {
        'strategy': 'S4_LONDON_PULL',
        'direction': direction,  # Dynamic direction based on ADX/DI trend
        'entry_type': 'LIMIT',
        'entry_price': ema20_m15,
        'stop_price': ema20_m15 - 15 if direction == "LONG" else ema20_m15 + 15,  # Direction-based stop
        'tp_price': None,
        'lot_size': 0.0,
        'expiry': bar['time'] + timedelta(minutes=config['expiry_minutes']),
        'tag': 's4_pullback',
        'metadata': {
            'ema_touched': True,
            'adx_increasing': adx_increasing,
            'di_plus': di_plus_h4,
            'di_minus': di_minus_h4
        }
    }

def evaluate_s5_ny_compress(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S5_NY_COMPRESS - NY Compression Breakout.
    """
    config = STRATEGY_CONFIGS["S5_NY_COMPRESS"]
    
    # Check session
    current_time = bar['time']
    start_utc = current_time.replace(
        hour=int(config['start_utc'].split(':')[0]),
        minute=int(config['start_utc'].split(':')[1]),
        second=0
    )
    end_utc = current_time.replace(
        hour=int(config['end_utc'].split(':')[0]),
        minute=int(config['end_utc'].split(':')[1]),
        second=0
    )
    
    if not (start_utc <= current_time <= end_utc):
        return None
    
    # Check if compression confirmed - implement detection
    if not state.get('s5_compression_confirmed'):
        # Implement compression detection logic
        # Check if price has been in a tight range for the last 4 hours (London session)
        london_bars = context.get('london_bars', [])  # Need to pass this from engine
        if len(london_bars) >= 48:  # 4 hours * 12 bars per hour (M5 data)
            london_highs = [b['high'] for b in london_bars[-48:]]
            london_lows = [b['low'] for b in london_bars[-48:]]
            london_range_points = max(london_highs) - min(london_lows)
            
            # Check if range is compressed (less than ATR threshold)
            atr_current = context.get('atr_m15', 20)
            if london_range_points < atr_current * 0.5:  # Compression threshold
                state['s5_compression_confirmed'] = True
                state['london_range'] = {
                    'high': max(london_highs),
                    'low': min(london_lows)
                }
            else:
                return None
        else:
            return None
    
    # Check if already fired today
    if state.get('s5_fired_today'):
        return None
    
    # STOP order beyond London boundary
    london_range = state.get('london_range', {})
    if not london_range:
        return None
    
    buy_level = london_range['high'] + config['breakout_points']
    sell_level = london_range['low'] - config['breakout_points']
    
    # Determine direction based on current price
    current_price = bar['close']
    if current_price > london_range.get('high', 0):
        direction = "LONG"
        entry_price = buy_level
        stop_price = london_range['low'] - 20
    elif current_price < london_range.get('low', 0):
        direction = "SHORT"
        entry_price = sell_level
        stop_price = london_range['high'] + 20
    else:
        return None
    
    return {
        'strategy': 'S5_NY_COMPRESS',
        'direction': direction,
        'entry_type': 'BUY_STOP' if direction == 'LONG' else 'SELL_STOP',
        'entry_price': entry_price,
        'stop_price': stop_price,
        'tp_price': None,
        'lot_size': 0.0,
        'expiry': None,
        'tag': 's5_breakout',
        'metadata': {
            'london_range': london_range,
            'compression_confirmed': True
        }
    }

def evaluate_s8_atr_spike(
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    S8_ATR_SPIKE - Flash Spike Continuation (independent lane).

    Phase 1: spike bar detected  → store midpoint/direction, return None.
    Phase 2: next qualifying bar closes past midpoint → fire MARKET entry.
    """
    config = STRATEGY_CONFIGS["S8_ATR_SPIKE"]

    if state.get('s8_fired_today'):
        return None

    # BUG-1 FIX: guard atr_h1 against None before any multiplication
    atr_h1 = context.get('atr_h1')
    if atr_h1 is None:
        return None
    atr_h1 = float(atr_h1)
    if atr_h1 <= 0:
        return None

    spike_threshold = atr_h1 * config['spike_atr_mult']
    bar_range = bar['high'] - bar['low']

    spike_midpoint  = state.get('s8_spike_midpoint')
    spike_direction = state.get('s8_spike_direction')

    # Phase 2: confirmation bar
    if spike_midpoint is not None and spike_direction is not None:
        if bar_range >= spike_threshold:   # must still be a significant bar
            if spike_direction == "LONG" and bar['close'] > spike_midpoint:
                return {
                    'strategy': 'S8_ATR_SPIKE',
                    'direction': 'LONG',
                    'entry_type': 'MARKET',
                    'entry_price': bar['close'],
                    'stop_price': bar['close'] - (atr_h1 * 0.5),
                    'tp_price':   bar['close'] + (atr_h1 * 1.0),
                    'lot_size': 0.0,
                    'expiry': None,
                    'tag': 's8_spike_confirmed',
                    'metadata': {'spike_confirmed': True, 'independent_lane': True},
                }
            elif spike_direction == "SHORT" and bar['close'] < spike_midpoint:
                return {
                    'strategy': 'S8_ATR_SPIKE',
                    'direction': 'SHORT',
                    'entry_type': 'MARKET',
                    'entry_price': bar['close'],
                    'stop_price': bar['close'] + (atr_h1 * 0.5),
                    'tp_price':   bar['close'] - (atr_h1 * 1.0),
                    'lot_size': 0.0,
                    'expiry': None,
                    'tag': 's8_spike_confirmed',
                    'metadata': {'spike_confirmed': True, 'independent_lane': True},
                }
        # Confirmation failed or not a spike bar — reset armed state
        state['s8_spike_midpoint'] = None
        state['s8_spike_direction'] = None

    # Phase 1: detect spike bar — arm for next bar confirmation
    if bar_range >= spike_threshold:
        state['s8_spike_midpoint']  = (bar['high'] + bar['low']) / 2
        state['s8_spike_direction'] = "LONG" if bar['close'] > bar['open'] else "SHORT"

    return None

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EVALUATION DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(
    strategy: str,
    state: Dict[str, Any],
    bar: Dict[str, Any],
    context: Dict[str, Any]
) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
    """
    Main strategy evaluation dispatcher.
    """
    strategy_map = {
        'S1_LONDON_BRK': evaluate_s1_london_brk,
        'S1B_FAILED_BRK': evaluate_s1b_failed_brk,
        'S1D_PYRAMID': evaluate_s1d_pyramid,
        'S1E_PYRAMID': evaluate_s1e_pyramid,
        'S1F_POST_TK': evaluate_s1f_post_tk,
        'S2_MEAN_REV': evaluate_s2_mean_rev,
        'S3_STOP_HUNT_REV': evaluate_s3_stop_hunt_rev,
        'S6_ASIAN_BRK': evaluate_s6_asian_brk,
        'S7_DAILY_STRUCT': evaluate_s7_daily_struct,
        'R3_CAL_MOMENTUM': evaluate_r3_cal_momentum,
        'S4_LONDON_PULL': evaluate_s4_london_pull,
        'S5_NY_COMPRESS': evaluate_s5_ny_compress,
        'S8_ATR_SPIKE': evaluate_s8_atr_spike,
    }
    
    evaluator = strategy_map.get(strategy)
    if not evaluator:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return evaluator(state, bar, context)
