"""
Backtest Strategies Package

Modular strategy implementations for XAUUSD backtest system.
Each strategy mirrors the live system implementation exactly.
"""

from .base_strategy import BaseStrategy
from .s1_family import S1LondonBrk, S1bFailedBrk, S1cStopHunt, S1dPyramid, S1ePyramid, S1fPostTk
from .s2_mean_rev import S2MeanRev
from .s3_stop_hunt import S3StopHuntRev
from .s4_london_pull import S4LondonPull
from .s5_ny_compress import S5NyCompress
from .s6_asian_brk import S6AsianBrk
from .s7_daily_struct import S7DailyStruct
from .s8_atr_spike import S8AtrSpike
from .r3_calendar_momentum import R3CalendarMomentum

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    "S1_LONDON_BRK": S1LondonBrk,
    "S1B_FAILED_BRK": S1bFailedBrk,
    "S1C_STOP_HUNT": S1cStopHunt,
    "S1D_PYRAMID": S1dPyramid,
    "S1E_PYRAMID": S1ePyramid,
    "S1F_POST_TK": S1fPostTk,
    "S2_MEAN_REV": S2MeanRev,
    "S3_STOP_HUNT_REV": S3StopHuntRev,
    "S4_LONDON_PULL": S4LondonPull,
    "S5_NY_COMPRESS": S5NyCompress,
    "S6_ASIAN_BRK": S6AsianBrk,
    "S7_DAILY_STRUCT": S7DailyStruct,
    "S8_ATR_SPIKE": S8AtrSpike,
    "R3_CAL_MOMENTUM": R3CalendarMomentum,
}

ALL_STRATEGIES = list(STRATEGY_REGISTRY.keys())

# Strategy families for position management
TREND_FAMILY_STRATEGIES = [
    "S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK", "S1E_PYRAMID",
    "S4_LONDON_PULL", "S5_NY_COMPRESS"
]

REVERSAL_FAMILY_STRATEGIES = [
    "S3_STOP_HUNT_REV"
]

INDEPENDENT_LANE_STRATEGIES = [
    "S8_ATR_SPIKE", "R3_CAL_MOMENTUM"
]

PENDING_ORDER_STRATEGIES = [
    "S1_LONDON_BRK", "S6_ASIAN_BRK", "S7_DAILY_STRUCT"
]

__all__ = [
    "BaseStrategy",
    "STRATEGY_REGISTRY",
    "ALL_STRATEGIES",
    "TREND_FAMILY_STRATEGIES", 
    "REVERSAL_FAMILY_STRATEGIES",
    "INDEPENDENT_LANE_STRATEGIES",
    "PENDING_ORDER_STRATEGIES",
    S1LondonBrk, S1bFailedBrk, S1cStopHunt, S1dPyramid, S1ePyramid, S1fPostTk,
    S2MeanRev, S3StopHuntRev, S4LondonPull, S5NyCompress,
    S6AsianBrk, S7DailyStruct, S8AtrSpike, R3CalendarMomentum
]

TREND_FAMILY_STRATEGIES = [
    "S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK", "S1E_PYRAMID",
    "S4_LONDON_PULL", "S5_NY_COMPRESS"
    "S3_STOP_HUNT_REV"
]

INDEPENDENT_LANE_STRATEGIES = [
    "S8_ATR_SPIKE", "R3_CAL_MOMENTUM"
]

PENDING_ORDER_STRATEGIES = [
    "S1_LONDON_BRK", "S6_ASIAN_BRK", "S7_DAILY_STRUCT"
]

__all__ = [
    "BaseStrategy",
    "STRATEGY_REGISTRY",
    "ALL_STRATEGIES",
    "TREND_FAMILY_STRATEGIES", 
    "REVERSAL_FAMILY_STRATEGIES",
    "INDEPENDENT_LANE_STRATEGIES",
    "PENDING_ORDER_STRATEGIES",
] + list(STRATEGY_REGISTRY.values())
