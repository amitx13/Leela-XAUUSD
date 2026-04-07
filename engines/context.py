"""
engines/context.py — Shared Market Context.

This engine unifies all MT5 data fetching (OHLCV) into a single object
instantiated at the start of each job cycle (M15, M5, Regime).
Reduces RPyC/MT5 API calls by ~60% and ensures all signals evaluate
on identical price/indicator snapshots.
"""
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import pytz
from engines.data_engine import fetch_ohlcv, get_tf_constant
from utils.logger import log_event, log_warning
import config

class MarketContext:
    def __init__(self, state: dict):
        self.state = state
        self.now_utc = datetime.now(pytz.utc)
        
        # Raw DataFrames
        self.data_m5 = None
        self.data_m15 = None
        self.data_h1 = None
        self.data_h4   = None
        
        # Performance Cache
        self.regime_val = None
        self.regime_multiplier = 1.0
        
        # Indicators
        self.atr_h1 = None
        self.ema20_h1 = None
        self.ema20_m5 = None
        
        # H4 ADX items
        self.adx_h4 = None
        self.di_plus_h4 = None
        self.di_minus_h4 = None
        self.adx_h4_is_increasing = False

    def refresh(self, tfs: list[str] = ["M5", "M15", "H1", "H4"]):
        """
        Fetches fresh data for specified timeframes and computes indicators.
        Call this at the beginning of any job that needs multiple data points.
        """
        if "H1" in tfs:
            self.data_h1 = fetch_ohlcv("H1", count=100)
            if self.data_h1 is not None and not self.data_h1.empty:
                # Use RMA for ATR as per spec (§B2)
                self.data_h1["atr"] = ta.atr(self.data_h1["high"], self.data_h1["low"], self.data_h1["close"], 
                                         length=14, mamode=config.ATR_MAMODE)
                self.data_h1["ema20"] = ta.ema(self.data_h1["close"], length=20)
                
                self.atr_h1 = float(self.data_h1["atr"].iloc[-1])
                self.ema20_h1 = float(self.data_h1["ema20"].iloc[-1])
                
                # Update shared state cache for Truth Engine / Candidates
                self.state["last_atr_h1_raw"] = self.atr_h1
                close_h1 = float(self.data_h1["close"].iloc[-1])
                self.state["last_atr_pct_h1"] = (self.atr_h1 / close_h1 * 100) if close_h1 > 0 else 0.0

        if "H4" in tfs:
            self.data_h4 = fetch_ohlcv("H4", count=100)
            if self.data_h4 is not None and not self.data_h4.empty:
                adx_df = ta.adx(self.data_h4["high"], self.data_h4["low"], self.data_h4["close"], length=14)
                if adx_df is not None and not adx_df.empty:
                    self.adx_h4 = float(adx_df["ADX_14"].iloc[-1])
                    self.di_plus_h4 = float(adx_df["DMP_14"].iloc[-1])
                    self.di_minus_h4 = float(adx_df["DMN_14"].iloc[-1])
                    
                    # Slope check (iloc[-2] is last closed bar)
                    if len(adx_df) >= 3:
                        self.adx_h4_is_increasing = adx_df["ADX_14"].iloc[-2] > adx_df["ADX_14"].iloc[-3]
                    
                    # Update state cache
                    self.state["last_adx_h4"] = self.adx_h4
                    self.state["last_di_plus_h4"] = self.di_plus_h4
                    self.state["last_di_minus_h4"] = self.di_minus_h4

        if "M15" in tfs:
            self.data_m15 = fetch_ohlcv("M15", count=100)

        if "M5" in tfs:
            self.data_m5 = fetch_ohlcv("M5", count=100)
            if self.data_m5 is not None and not self.data_m5.empty:
                self.data_m5["ema20"] = ta.ema(self.data_m5["close"], length=20)
                self.ema20_m5 = float(self.data_m5["ema20"].iloc[-1])

    def get_last_closed_bar(self, tf: str) -> dict | None:
        """Helper to get iloc[-2] bar for any timeframe."""
        df = getattr(self, f"data_{tf.lower()}")
        if df is None or len(df) < 2:
            return None
        row = df.iloc[-2]
        return {
            "time":  row["time"],
            "open":  float(row["open"]),
            "high":  float(row["high"]),
            "low":   float(row["low"]),
            "close": float(row["close"]),
            "tick_volume": float(row.get("tick_volume", 0)),
        }

    def get_forming_bar(self, tf: str) -> dict | None:
        """Helper to get iloc[-1] bar (current forming bar)."""
        df = getattr(self, f"data_{tf.lower()}")
        if df is None or len(df) < 1:
            return None
        row = df.iloc[-1]
        return {
            "time":  row["time"],
            "open":  float(row["open"]),
            "high":  float(row["high"]),
            "low":   float(row["low"]),
            "close": float(row["close"]),
        }
