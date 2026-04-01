"""
utils/session.py — DST-safe session detection.

Uses pytz timezone objects for London and New York — handles DST automatically.
Returns one of four canonical session strings used throughout the system.

Session strings (exact — used as DB values and regime gate keys):
  LONDON_NY_OVERLAP  — both open simultaneously (highest liquidity)
  LONDON             — London only
  NY                 — New York only
  OFF_HOURS          — neither open (Asian session, weekend, daily close)

London hours : 08:00–17:00 local (adjusts for BST/GMT automatically)
NY hours     : 08:00–17:00 local (adjusts for EDT/EST automatically)
"""
import pytz
from datetime import datetime


LONDON_TZ = pytz.timezone("Europe/London")
NY_TZ      = pytz.timezone("America/New_York")
IST_TZ     = pytz.timezone("Asia/Kolkata")


def get_current_session() -> str:
    """
    Returns the current trading session based on real-time DST-adjusted hours.
    Called by regime engine on every regime calculation.
    """
    now_utc    = datetime.now(pytz.utc).hour
    now_london = datetime.now(LONDON_TZ).hour
    now_ny     = datetime.now(NY_TZ).hour

    if 8 <= now_london < 17 and 8 <= now_ny < 17:
        return "LONDON_NY_OVERLAP"
    elif 8 <= now_london < 17:
        return "LONDON"
    elif 8 <= now_ny < 17:
        return "NY"
    elif 0 <= now_utc < 8:
        return "ASIAN"
    return "OFF_HOURS"


def get_session_for_datetime(dt: datetime) -> str:
    """
    Returns session for a given UTC datetime.
    Used in backtesting, Truth Engine replay, and unit tests.
    dt must be timezone-aware.
    """
    assert dt.tzinfo is not None, \
        "get_session_for_datetime: dt must be timezone-aware. Got naive datetime."

    london_hour = dt.astimezone(LONDON_TZ).hour
    ny_hour     = dt.astimezone(NY_TZ).hour
    utc_hour = dt.astimezone(pytz.utc).hour

    if 8 <= london_hour < 17 and 8 <= ny_hour < 17:
        return "LONDON_NY_OVERLAP"
    elif 8 <= london_hour < 17:
        return "LONDON"
    elif 8 <= ny_hour < 17:
        return "NY"
    elif 0 <= utc_hour < 8:
        return "ASIAN"
    return "OFF_HOURS"


def is_london_session_active() -> bool:
    return datetime.now(LONDON_TZ).hour in range(8, 17)


def is_ny_session_active() -> bool:
    return datetime.now(NY_TZ).hour in range(8, 17)


def is_trading_hours() -> bool:
    """True if any major session is open."""
    return get_current_session() != "OFF_HOURS"


def get_london_local_time() -> datetime:
    """Current time in London timezone. Used for time-kill checks."""
    return datetime.now(LONDON_TZ)


def get_ny_local_time() -> datetime:
    """Current time in NY timezone. Used for S1f time-kill check."""
    return datetime.now(NY_TZ)


def get_ist_time() -> datetime:
    """Current IST time. Used for daily resets and calendar pulls."""
    return datetime.now(IST_TZ)
