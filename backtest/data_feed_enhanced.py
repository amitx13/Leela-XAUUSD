"""
Enhanced Data Feed

Multi-timeframe support with event simulation and spread modeling.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger("backtest.enhanced_data_feed")


class EnhancedHistoricalDataFeed:
    """
    Enhanced historical data feed with multi-timeframe support.
    
    Features:
    - Multiple timeframes (M5, M15, H1, H4, D1)
    - Economic event simulation
    - Spread modeling
    - DXY correlation data
    - Efficient caching
    """
    
    def __init__(self, cache_dir: str = "backtest_data"):
        self.cache_dir = cache_dir
        self.data = {}  # timeframe -> DataFrame
        self.events = []  # Economic events
        self.spreads = {}  # timeframe -> spread series
        self.dxy_data = {}  # timeframe -> DXY series
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Enhanced data feed initialized with cache dir: {cache_dir}")
    
    def load_data(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str] = None
    ) -> None:
        """Load historical data for specified timeframes."""
        if timeframes is None:
            timeframes = ["M5", "M15", "H1", "H4", "D1"]
        
        logger.info(f"Loading data for {timeframes} from {start_date} to {end_date}")
        
        for timeframe in timeframes:
            self._load_timeframe_data(timeframe, start_date, end_date)
        
        # Load economic events
        self._load_economic_events(start_date, end_date)
        
        # Load DXY correlation data
        self._load_dxy_data(start_date, end_date)
        
        # Generate spread data
        self._generate_spread_data(timeframes, start_date, end_date)
        
        logger.info("Enhanced data loading completed")
    
    def _load_timeframe_data(self, timeframe: str, start_date: datetime, end_date: datetime):
        """Load data for specific timeframe."""
        cache_file = os.path.join(self.cache_dir, f"xauusd_{timeframe.lower()}.csv")
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, parse_dates=["time"])
                df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
                self.data[timeframe] = df
                logger.info(f"Loaded {len(df)} {timeframe} bars from cache")
                return
            except Exception as e:
                logger.error(f"Error loading {cache_file}: {e}")
        
        # Generate synthetic data if no cache file
        logger.warning(f"No cache file for {timeframe}, generating synthetic data")
        self.data[timeframe] = self._generate_synthetic_data(timeframe, start_date, end_date)
    
    def _generate_synthetic_data(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        # Calculate number of bars based on timeframe
        if timeframe == "M5":
            freq = "5min"
        elif timeframe == "M15":
            freq = "15min"
        elif timeframe == "H1":
            freq = "1H"
        elif timeframe == "H4":
            freq = "4H"
        elif timeframe == "D1":
            freq = "1D"
        else:
            freq = "1H"
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducible results
        
        n_bars = len(date_range)
        base_price = 2000.0  # Base XAUUSD price
        
        # Generate price series first
        returns = np.random.normal(0, 0.0002, n_bars)  # Small random movements
        prices = [base_price]
        
        for i in range(1, n_bars):
            change = returns[i] * base_price
            new_price = prices[-1] + change
            
            # Add some volatility patterns
            if i % 100 == 0:  # Occasional bigger moves
                change *= 5
            
            prices.append(new_price)
        
        # Create OHLC data from prices
        opens = prices[:-1]
        closes = prices[1:]
        
        # Generate highs and lows
        highs = []
        lows = []
        for i in range(len(opens)):
            high = max(opens[i], closes[i])
            low = min(opens[i], closes[i])
            highs.append(high)
            lows.append(low)
        
        # Generate volumes
        volumes = np.random.randint(100, 1000, n_bars-1)
        
        # Create DataFrame with matching lengths
        df = pd.DataFrame({
            "time": date_range[1:],  # Skip first to match OHLC
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        logger.info(f"Generated {len(df)} synthetic {timeframe} bars")
        return df
    
    def _load_economic_events(self, start_date: datetime, end_date: datetime):
        """Load economic events for the period."""
        # In a real system, this would load from a calendar API
        # For backtest, we'll generate synthetic high-impact events
        
        events = []
        current = start_date
        
        # Generate major news events (FOMC, NFP, CPI, etc.)
        while current <= end_date:
            # Randomly decide if this is a major event day
            if np.random.random() < 0.1:  # 10% chance of major event
                # Generate event around 8:30 AM NY time (13:30 UTC)
                event_time = current.replace(hour=13, minute=30, second=0)
                
                events.append({
                    "time": event_time,
                    "name": np.random.choice(["FOMC", "NFP", "CPI", "GDP"]),
                    "impact": np.random.choice(["HIGH", "MEDIUM", "LOW"]),
                    "currency": "USD",
                    "forecast": np.random.choice(["Better", "Worse", "Same"]),
                })
                
                logger.info(f"Generated economic event: {events[-1]['name']} at {event_time}")
            
            current += timedelta(days=1)
        
        self.events = events
        logger.info(f"Generated {len(events)} economic events")
    
    def _load_dxy_data(self, start_date: datetime, end_date: datetime):
        """Load DXY correlation data."""
        # In a real system, this would load from market data API
        # For backtest, we'll generate synthetic DXY data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq="1h")
        
        # Generate synthetic DXY with some correlation to XAUUSD
        dxy_values = []
        base_dxy = 100.0
        
        for i in range(len(date_range)):
            # DXY tends to be inversely correlated with XAUUSD
            change = np.random.normal(0, 0.5, 1)  # DXY volatility
            new_dxy = base_dxy + change
            
            # Add some trend patterns
            if i % 50 == 0:  # Occasional trend changes
                change *= 2
            
            dxy_values.append(new_dxy)
            base_dxy = new_dxy
        
        # Create DataFrame with proper 1D arrays
        dxy_changes = np.diff(dxy_values).tolist()
        if len(dxy_changes) < len(dxy_values):
            dxy_changes.insert(0, 0.0)  # Insert 0 for first element to match length
        
        dxy_df = pd.DataFrame({
            "time": date_range,
            "dxy": dxy_values,
            "change": dxy_changes
        })
        
        self.dxy_data["H1"] = dxy_df
        logger.info(f"Generated {len(dxy_df)} DXY data points")
    
    def _generate_spread_data(self, timeframes: List[str], start_date: datetime, end_date: datetime):
        """Generate realistic spread data."""
        for timeframe in timeframes:
            if timeframe not in self.data:
                continue
            
            df = self.data[timeframe]
            if df.empty:
                continue
            
            # Generate spread based on time and volatility
            import numpy as np
            np.random.seed(42)
            
            n_bars = len(df)
            spreads = []
            
            for i in range(n_bars):
                hour = df.iloc[i]["time"].hour
                
                # Base spread varies by session
                if 13 <= hour <= 17:  # London/NY overlap - tight spreads
                    base_spread = np.random.normal(1.5, 0.3, 1)
                elif 8 <= hour <= 12:  # Asian session - wider spreads
                    base_spread = np.random.normal(2.5, 0.5, 1)
                else:  # Other times - moderate spreads
                    base_spread = np.random.normal(2.0, 0.4, 1)
                
                # Add volatility-based spread widening
                if i > 0:
                    prev_range = df.iloc[i]["high"] - df.iloc[i-1]["low"]
                    curr_range = df.iloc[i]["high"] - df.iloc[i]["low"]
                    if curr_range > prev_range * 1.5:  # Volatility spike
                        base_spread *= 2.0
                
                spreads.append(max(0.5, base_spread))  # Minimum 0.5 spread
            
            spread_df = pd.DataFrame({
                "time": df["time"],
                "spread": spreads
            })
            
            self.spreads[timeframe] = spread_df
            logger.info(f"Generated spread data for {timeframe}")
    
    def get_bars(self, timeframe: str, current_time: datetime, count: int = 1000) -> List[Dict[str, Any]]:
        """Get recent bars for specified timeframe."""
        if timeframe not in self.data:
            return []
        
        df = self.data[timeframe]
        if df.empty:
            return []
        
        # Filter bars up to current time
        df_filtered = df[df['time'] <= current_time].tail(count)
        
        # Convert to list of dictionaries
        bars = []
        for _, row in df_filtered.iterrows():
            bars.append({
                "time": row["time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            })
        
        return bars
    
    def get_upcoming_events(self, current_time: datetime, lookahead_hours: int = 2) -> List[Dict[str, Any]]:
        """Get upcoming economic events within lookahead window."""
        upcoming = []
        cutoff = current_time + timedelta(hours=lookahead_hours)
        
        for event in self.events:
            if current_time < event["time"] <= cutoff:
                upcoming.append(event)
        
        return upcoming
    
    def get_current_spread(self, current_time: datetime) -> float:
        """Get current spread for specified time."""
        # Find appropriate timeframe (use M5 for current spread)
        timeframe = "M5"
        
        if timeframe in self.spreads:
            spread_df = self.spreads[timeframe]
            if not spread_df.empty:
                # Find spread for current time
                mask = spread_df["time"] <= current_time
                if mask.any():
                    current_spread = spread_df[mask].iloc[-1]["spread"]
                    return current_spread
        
        # Default spread
        return 2.0
    
    def get_dxy_correlation(self, current_time: datetime) -> Optional[float]:
        """Get DXY value for correlation analysis."""
        if "H1" in self.dxy_data and not self.dxy_data["H1"].empty:
            dxy_df = self.dxy_data["H1"]
            mask = dxy_df["time"] <= current_time
            if mask.any():
                return dxy_df[mask].iloc[-1]["dxy"]
        
        return None
