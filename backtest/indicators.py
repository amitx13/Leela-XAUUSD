import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using Wilder's RMA (pinned to TradingView)."""
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift(1))
    lc = np.abs(df['low'] - df['close'].shift(1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    # Wilder's RMA
    atr = pd.Series(tr, index=df.index)
    return atr.ewm(alpha=1/period, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate EMA."""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_adx_h4(df: pd.DataFrame, period: int = 14) -> Optional[Dict[str, float]]:
    """Calculate ADX with DI+/DI-."""
    if len(df) < period + 1:
        return None
    
    adx = ta.adx(df['high'], df['low'], df['close'], length=period)
    
    if adx is None or len(adx) < 2:
        return None
    
    latest = adx.iloc[-1]
    
    # Check if ADX is increasing
    is_increasing = False
    if len(adx) >= 2:
        is_increasing = bool(latest['ADX_14'] > adx.iloc[-2]['ADX_14'])
        
    return {
        'adx': float(latest['ADX_14']),
        'di_plus': float(latest['DMP_14']),
        'di_minus': float(latest['DMN_14']),
        'increasing': is_increasing
    }
