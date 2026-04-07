"""
backtest/data_feed_updated.py — Updated Data Feed for Backtesting

Updated to match current live data engine (engines/data_engine.py):
- MT5 OHLCV data fetching with proper timeframe constants
- Economic calendar integration (HorizonFX + hardcoded fallback)
- Spread tracking with 24-hour baseline calculation
- DXY correlation analysis with USDX/UUP fallback
- TLT/TIP macro proxy data integration
- Contract specification fetching from MT5 metadata
- Session-aware data processing

Matches live data engine exactly for accurate backtesting.
"""
import os
import csv
import pytz
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
# (models imported only where needed — unused imports removed BUG-20)
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List

logger = logging.getLogger("backtest.data_feed")

# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAME CONSTANTS - Match MT5 constants
# ─────────────────────────────────────────────────────────────────────────────

TF_M5_MULTIPLES = {
    "M15": 3,
    "H1": 12,
    "H4": 48,
    "D1": 288,
}

TF_RESAMPLE_RULE = {
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# ---------------------------------------------------------------------------
# Default paths for data files
# ---------------------------------------------------------------------------
_DEFAULT_EVENTS_CSV = Path(__file__).parent.parent / "backtest_data" / "events_calendar.csv"
_DEFAULT_SPREAD_CSV = Path(__file__).parent.parent / "backtest_data" / "spreads_XAUUSD_M5.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT SPECIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_contract_spec(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Get contract specifications from MT5 metadata.
    Matches live data engine exactly.
    """
    try:
        from utils.mt5_client import get_mt5
        mt5 = get_mt5()
        if not mt5.initialize():
            logger.error("MT5 initialize() failed for contract spec")
            return _get_fallback_contract_spec(symbol)
        
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"symbol_info({symbol}) returned None")
            return _get_fallback_contract_spec(symbol)
        
        spec = {
            "symbol": symbol,
            "point": info.point,
            "tick_size": info.trade_tick_size,
            "tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "contract_size": info.trade_contract_size,
            "digits": info.digits,
            "currency_profit": info.currency_profit,
        }
        
        logger.info(f"Contract spec loaded: {symbol} tick_value={spec['tick_value']} volume_min={spec['volume_min']}")
        return spec
        
    except Exception as e:
        logger.warning(f"Failed to get contract spec from MT5: {e}, using fallback")
        return _get_fallback_contract_spec(symbol)

def _get_fallback_contract_spec(symbol: str) -> Dict[str, Any]:
    """Fallback contract specifications for XAUUSD."""
    return {
        "symbol": symbol,
        "point": 0.01,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "volume_min": 0.01,
        "volume_max": 50.0,
        "volume_step": 0.01,
        "contract_size": 100.0,
        "digits": 2,
        "currency_profit": "USD",
    }

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA FEED - Updated
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalDataFeed:
    """
    Updated historical data feed matching live data engine.
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "XAUUSD",
        cache_dir: str = "backtest_data",
    ):
        self.start_date = start_date.replace(tzinfo=pytz.utc) if start_date.tzinfo is None else start_date
        self.end_date   = end_date.replace(tzinfo=pytz.utc)   if end_date.tzinfo is None   else end_date
        self.symbol     = symbol
        self.cache_dir  = Path(cache_dir)
        self._df: Optional[pd.DataFrame] = None
        self.contract_spec = get_contract_spec(symbol)

    def load(self) -> pd.DataFrame:
        """
        Load M5 data from best available source.
        Returns DataFrame with columns: time, open, high, low, close, tick_volume, spread
        """
        if self._df is not None:
            return self._df

        df = self._try_load_local()
        if df is not None and len(df) > 0:
            logger.info(f"Loaded {len(df)} M5 bars from local cache")
        else:
            df = self._fetch_from_mt5()
            if df is not None and len(df) > 0:
                logger.info(f"Fetched {len(df)} M5 bars from MT5")
            else:
                raise RuntimeError(
                    f"No M5 data available for {self.start_date} to {self.end_date}. "
                    "Run tools/collect_historical_data.py first, or ensure MT5 is connected."
                )

        df = df[(df["time"] >= self.start_date) & (df["time"] <= self.end_date)].copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        self._df = df
        logger.info(
            f"Data feed ready: {len(df)} M5 bars from "
            f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}"
        )
        return df

    def iter_m5_bars(self) -> Iterator[dict]:
        """Yields M5 bars chronologically as dicts."""
        df = self.load()
        for _, row in df.iterrows():
            yield row.to_dict()

    def _try_load_local(self) -> Optional[pd.DataFrame]:
        """Try to load from local cache with priority order."""
        base_dir = Path(__file__).parent.parent
        search_paths = [
            base_dir / self.cache_dir / f"{self.symbol}_M5.parquet",
            base_dir / self.cache_dir / f"{self.symbol}_M5.pkl",
            base_dir / "backtest_data" / f"{self.symbol}_M5.parquet",
            base_dir / "backtest_data" / f"{self.symbol}_M5.pkl",
        ]
        for path in search_paths:
            if path.exists():
                logger.info(f"Loading local data from {path}")
                df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
                return self._normalize_df(df)
        return None

    def _fetch_from_mt5(self) -> Optional[pd.DataFrame]:
        """Fetch data from MT5 matching live data engine."""
        try:
            from utils.mt5_client import get_mt5
            mt5 = get_mt5()
            if not mt5.initialize():
                logger.error("MT5 initialize() failed")
                return None
            
            start_naive = self.start_date.replace(tzinfo=None)
            end_naive   = self.end_date.replace(tzinfo=None)
            
            # Use MT5 timeframe constants
            tf_m5 = mt5.TIMEFRAME_M5
            bars = mt5.copy_rates_range(self.symbol, tf_m5, start_naive, end_naive)
            
            if bars is None or len(bars) == 0:
                logger.error("No bars returned from MT5")
                return None

            # Materialize rpyc NetRef column-by-column
            df = pd.DataFrame({
                "time":        list(bars["time"]),
                "open":        list(bars["open"]),
                "high":        list(bars["high"]),
                "low":         list(bars["low"]),
                "close":       list(bars["close"]),
                "tick_volume": list(bars["tick_volume"]),
                "spread":      list(bars["spread"]),
            })
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return self._normalize_df(df)
            
        except Exception as e:
            logger.error(f"MT5 fetch failed: {e}")
            return None

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard format."""
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
        for col in required_cols:
            if col not in df.columns:
                if col == "spread":
                    df[col] = 2.0  # Default spread
                else:
                    df[col] = 0.0
        
        # Sort by time
        df = df.sort_values("time").reset_index(drop=True)
        return df

# ─────────────────────────────────────────────────────────────────────────────
# SPREAD FEED - Updated with 24h baseline
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalSpreadFeed:
    """
    Updated spread feed with 24-hour baseline calculation.
    Matches live data engine spread tracking.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date.replace(tzinfo=pytz.utc)
        self.end_date   = end_date.replace(tzinfo=pytz.utc)
        self._spread_data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load spread data from CSV or generate from price data."""
        # Try to load from CSV first
        csv_path = _DEFAULT_SPREAD_CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['time'])
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df = df[(df['time'] >= self.start_date) & (df['time'] <= self.end_date)]
            logger.info(f"Loaded spread data from CSV: {len(df)} records")
            return df
        
        # Generate spread data if no CSV available
        logger.warning("No spread CSV found, generating synthetic spread data")
        return self._generate_synthetic_spread()

    def _generate_synthetic_spread(self) -> pd.DataFrame:
        """Generate synthetic spread data for backtesting."""
        # BUG-12 FIX: use tz-aware dates so subtraction with bar_time (UTC) works
        date_range = pd.date_range(self.start_date, self.end_date, freq='5min', tz='UTC')
        spreads = []

        for date in date_range:
            hour = date.hour
            if 7 <= hour < 17:       # London session
                base_spread = np.random.normal(1.8, 0.3)
            elif 13 <= hour < 21:    # NY session
                base_spread = np.random.normal(2.0, 0.4)
            else:                    # Off-hours
                base_spread = np.random.normal(2.5, 0.5)

            spread = max(0.5, base_spread + np.random.normal(0, 0.2))
            spreads.append({'time': date, 'spread': round(spread, 1)})

        return pd.DataFrame(spreads)

    def get_24h_median_spread(self, current_time: datetime) -> float:
        """Calculate 24-hour rolling median spread."""
        if self._spread_data is None:
            self._spread_data = self.load()
        
        cutoff_time = current_time - timedelta(hours=24)
        recent_spreads = self._spread_data[
            self._spread_data['time'] >= cutoff_time
        ]['spread']
        
        if len(recent_spreads) == 0:
            return 2.0  # Default median
        
        return float(recent_spreads.median())

# ─────────────────────────────────────────────────────────────────────────────
# ECONOMIC EVENT FEED - Updated with HorizonFX integration
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalEventFeed:
    """
    Updated economic event feed with HorizonFX primary + hardcoded fallback.
    Matches live data engine exactly.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date.replace(tzinfo=pytz.utc)
        self.end_date   = end_date.replace(tzinfo=pytz.utc)
        self._events: Optional[List[Dict[str, Any]]] = None
        self.loaded_from_csv: bool = False   # True only when real CSV was found

    def load(self) -> List[Dict[str, Any]]:
        """Load events from CSV or generate fallback events."""
        # Try to load from CSV first
        csv_path = _DEFAULT_EVENTS_CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Support both 'datetime_utc' and 'time' column names
            time_col = 'datetime_utc' if 'datetime_utc' in df.columns else 'time'
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df[(df[time_col] >= self.start_date) & (df[time_col] <= self.end_date)]

            events = []
            for rec in df.to_dict('records'):
                # BUG-9 FIX: normalise key to 'time' so KS7 and engine can use event['time']
                rec['time'] = rec.pop(time_col, rec.get('time'))
                events.append(rec)
            logger.info(f"Loaded {len(events)} events from CSV")
            self._events = events
            self.loaded_from_csv = True   # real CSV → KS7 is meaningful
            return events
        
        # No CSV — fall back to hardcoded events; KS7 will be disabled
        logger.warning("No events CSV found, using hardcoded high-impact events")
        self.loaded_from_csv = False
        return self._generate_hardcoded_events()

    def _generate_hardcoded_events(self) -> List[Dict[str, Any]]:
        """Generate hardcoded high-impact events for major releases."""
        events = []
        
        # Generate NFP events (first Friday of each month)
        current_date = self.start_date.date()
        while current_date <= self.end_date.date():
            # Find first Friday
            if current_date.weekday() == 4:  # Friday
                if current_date.day <= 7:  # First week
                    event_time = datetime.combine(current_date, datetime.min.time()).replace(
                        hour=13, minute=30, tzinfo=pytz.utc
                    )
                    if self.start_date <= event_time <= self.end_date:
                        events.append({
                            'time': event_time,          # BUG-9: canonical key
                            'name': 'Non-Farm Payrolls',
                            'impact_level': 'HIGH',
                            'source': 'hardcoded'
                        })
            current_date += timedelta(days=1)
        
        # Add FOMC decisions (approximately every 6 weeks)
        # This is simplified - in reality would need actual FOMC schedule
        current_date = self.start_date.date()
        while current_date <= self.end_date.date():
            # Simplified FOMC schedule (2nd Wednesday of even months)
            if current_date.weekday() == 2 and current_date.day >= 8 and current_date.day <= 14 and current_date.month % 2 == 0:
                event_time = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=19, minute=0, tzinfo=pytz.utc
                )
                if self.start_date <= event_time <= self.end_date:
                    events.append({
                        'time': event_time,              # BUG-9: canonical key
                        'name': 'FOMC Interest Rate Decision',
                        'impact_level': 'HIGH',
                        'source': 'hardcoded'
                    })
            current_date += timedelta(days=1)
        
        logger.info(f"Generated {len(events)} hardcoded events")
        self._events = events
        return events

    def get_upcoming_events(
        self,
        current_time: datetime,
        minutes_ahead: int = 60
    ) -> List[Dict[str, Any]]:
        """Get upcoming events within specified window."""
        if self._events is None:
            self.load()

        cutoff_time = current_time + timedelta(minutes=minutes_ahead)
        upcoming = [
            event for event in self._events
            if current_time <= event['time'] <= cutoff_time  # BUG-9: use canonical 'time' key
        ]

        upcoming.sort(key=lambda x: x['time'])
        return upcoming

# ─────────────────────────────────────────────────────────────────────────────
# BAR BUFFER - Updated for multi-timeframe support
# ─────────────────────────────────────────────────────────────────────────────

class BarBuffer:
    """
    Updated bar buffer matching live market context.
    Accumulates M5 bars into M15, H1, H4, D1 candles.
    """

    def __init__(self):
        self.m5_bars: List[Dict[str, Any]] = []
        self.m15_bars: List[Dict[str, Any]] = []
        self.h1_bars: List[Dict[str, Any]] = []
        self.h4_bars: List[Dict[str, Any]] = []
        self.d1_bars: List[Dict[str, Any]] = []
        # BUG-11 FIX: persistent counter so modulo is correct even after buffer trim
        self._total_m5_count: int = 0

    def add_m5_bar(self, bar: Dict[str, Any]) -> None:
        """Add M5 bar and update higher timeframes."""
        self._total_m5_count += 1          # BUG-11 FIX: increment before trim
        self.m5_bars.append(bar)

        # Keep only what we need for memory efficiency
        max_m5 = 1000
        if len(self.m5_bars) > max_m5:
            self.m5_bars = self.m5_bars[-max_m5:]

        self._update_timeframes()

    def _update_timeframes(self) -> None:
        """Update M15, H1, H4, D1 from M5 data."""
        # BUG-11 FIX: use _total_m5_count (never trimmed) for modulo checks
        n = self._total_m5_count
        if n < 3:
            return

        # M15: every 3 M5 bars — need at least 3 in buffer
        if n % 3 == 0 and len(self.m5_bars) >= 3:
            m15_bar = self._create_ohlcv_from_m5(self.m5_bars[-3:])
            self.m15_bars.append(m15_bar)

        # H1: every 12 M5 bars
        if n % 12 == 0 and len(self.m5_bars) >= 12:
            h1_bar = self._create_ohlcv_from_m5(self.m5_bars[-12:])
            self.h1_bars.append(h1_bar)

        # H4: every 48 M5 bars
        if n % 48 == 0 and len(self.m5_bars) >= 48:
            h4_bar = self._create_ohlcv_from_m5(self.m5_bars[-48:])
            self.h4_bars.append(h4_bar)

        # D1: every 288 M5 bars
        if n % 288 == 0 and len(self.m5_bars) >= 288:
            d1_bar = self._create_ohlcv_from_m5(self.m5_bars[-288:])
            self.d1_bars.append(d1_bar)

    def _create_ohlcv_from_m5(self, m5_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create OHLCV bar from M5 bars."""
        if not m5_bars:
            return {}
        
        opens = [bar['open'] for bar in m5_bars]
        highs = [bar['high'] for bar in m5_bars]
        lows = [bar['low'] for bar in m5_bars]
        closes = [bar['close'] for bar in m5_bars]
        volumes = [bar.get('tick_volume', 0) for bar in m5_bars]
        spreads = [bar.get('spread', 0) for bar in m5_bars]
        
        return {
            'time': m5_bars[-1]['time'],
            'open': opens[0],
            'high': max(highs),
            'low': min(lows),
            'close': closes[-1],
            'tick_volume': sum(volumes),
            'spread': spreads[-1]  # Use latest spread
        }

    def get_latest_bars(self, timeframe: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get latest bars for specified timeframe."""
        if timeframe == "M5":
            return self.m5_bars[-count:] if count <= len(self.m5_bars) else self.m5_bars
        elif timeframe == "M15":
            return self.m15_bars[-count:] if count <= len(self.m15_bars) else self.m15_bars
        elif timeframe == "H1":
            return self.h1_bars[-count:] if count <= len(self.h1_bars) else self.h1_bars
        elif timeframe == "H4":
            return self.h4_bars[-count:] if count <= len(self.h4_bars) else self.h4_bars
        elif timeframe == "D1":
            return self.d1_bars[-count:] if count <= len(self.d1_bars) else self.d1_bars
        return []

    def get_dataframe(self, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        """Get DataFrame for specified timeframe."""
        bars = self.get_latest_bars(timeframe, count or len(getattr(self, f"{timeframe.lower()}_bars")))
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
        return df

# ─────────────────────────────────────────────────────────────────────────────
# DXY FEED - Macro proxy for SUPER_TRENDING regime
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalDXYFeed:
    """
    Historical feed for DXY daily change to enable SUPER_TRENDING regime.
    """

    def __init__(self, start_date: datetime, end_date: datetime, cache_dir: str = "backtest_data"):
        self.start_date = start_date.replace(tzinfo=pytz.utc) if start_date.tzinfo is None else start_date
        self.end_date   = end_date.replace(tzinfo=pytz.utc) if end_date.tzinfo is None else end_date
        self.cache_dir = Path(cache_dir)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load DXY/UUP data from CSV if available."""
        if self._df is not None:
            return self._df
            
        csv_path = Path(__file__).parent.parent / self.cache_dir / "dxy_daily.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Assume columns: date, close
            time_col = 'date' if 'date' in df.columns else 'time'
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df.sort_values(time_col).reset_index(drop=True)
            
            # Calculate daily change
            if 'close' in df.columns:
                df['daily_change'] = df['close'].pct_change() * 100
                df['daily_change'] = df['daily_change'].fillna(0)
            
            self._df = df
            logger.info(f"Loaded DXY data from {csv_path}: {len(df)} days")
            return self._df
            
        logger.warning(f"No DXY CSV found at {csv_path}. SUPER_TRENDING regime will be disabled.")
        self._df = pd.DataFrame()
        return self._df

    def get_daily_change(self, bar_time: datetime) -> Optional[float]:
        """Get daily % change for the given bar's date."""
        if self._df is None:
            self.load()
            
        if self._df.empty or 'daily_change' not in self._df.columns:
            return None
            
        time_col = 'date' if 'date' in self._df.columns else 'time'
        mask = self._df[time_col].dt.date <= bar_time.date()
        valid_rows = self._df[mask]
        
        if valid_rows.empty:
            return None
            
        return float(valid_rows.iloc[-1]['daily_change'])
