"""
config.py — All constants, env vars, and calibration defaults.
All values that require Phase 0 calibration are clearly flagged.
All thresholds read from system_config DB table at startup (ADD-3).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Environment ──────────────────────────────────────────────────────────────
ENV = os.getenv("ENV", "dev")   # 'dev' or 'prod'

# ── MT5 rpyc bridge (mt5linux) ───────────────────────────────────────────────
MT5_HOST   = os.getenv("MT5_HOST", "localhost")
MT5_PORT   = int(os.getenv("MT5_PORT", "18812"))
SYMBOL     = "XAUUSD"
MAGIC      = 20260320  # date-based magic — consistent across ALL MT5 operations

# ── Database ─────────────────────────────────────────────────────────────────
# explicit 127.0.0.1 — forces TCP, never Unix socket (arch decision)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://xauusd_user:changeme@127.0.0.1:5432/xauusd"
)

# ── Broker — IC Markets Raw Spread ───────────────────────────────────────────
# RULE: CONTRACT_SPEC is populated from MT5 metadata on every startup.
# Never hardcode tick_value, point_size, or lot_limits here.
# This dict is the runtime container only — starts empty, filled in init.
CONTRACT_SPEC: dict = {}

# Commission: $3.50/lot/side = $7.00 round trip (verify against live account)
COMMISSION_PER_LOT_PER_SIDE = 3.50

# ── Spread fallbacks (C1 Fix) ─────────────────────────────────────────────────
# Used during startup until 6 real readings exist (~30 min)
SPREAD_FALLBACK_POINTS = {
    "ASIAN":          35,
    "LONDON":         20,
    "LONDON_NY":      20,
    "NY":             22,
    "OFF_HOURS":      50,  # conservative — should not trade anyway
}
SPREAD_INIT_MIN_READINGS = 6
SPREAD_INIT_TIMEOUT_MIN  = 30

# ── Kill Switch thresholds ────────────────────────────────────────────────────
KS2_SPREAD_MULTIPLIER      = 2.5    # reject if spread > 2.5× 24h avg (v1.1)
PREPLACEMENT_SPREAD_MULTIPLIER = 1.2  # block S1 pending if spread > 1.2× session-window baseline
# S1 pre-London spread baseline: same wall-clock window (UTC=GMT), Mon–Fri, last N days from spread_log
S1_PREPLACEMENT_SPREAD_WINDOW_START_MIN_UTC = 7 * 60 + 45   # 07:45
S1_PREPLACEMENT_SPREAD_WINDOW_END_MIN_UTC = 8 * 60 + 5      # 08:05
S1_PREPLACEMENT_SPREAD_LOOKBACK_TRADING_DAYS = 5
KS3_DAILY_LOSS_LIMIT_PCT   = -0.040  # KS-1 FIX: was -0.030 — too tight, fires after 2-3 trades
KS4_LOSS_STREAK_COUNT      = 6       # KS-3 FIX: was 4 — 4 consecutive losses is normal for breakout systems
KS4_REDUCED_TRADES         = 3       # KS-3 FIX: was 5 — shorter penalty duration
KS5_WEEKLY_LOSS_LIMIT_PCT  = -0.120  # v3.0 FIX: was -0.100 — widened to match weekly volatility profile
KS6_DRAWDOWN_LIMIT_PCT     = 0.20    # v3.0 FIX: was 0.12 — 12% DD too tight for gold trend-following
KS7_PRE_EVENT_MINUTES      = 45      # blackout window before HIGH impact event
KS7_POST_EVENT_MINUTES     = 20      # minimum wait after event release
KS7_ATR_RESUME_MULTIPLIER  = 1.30    # resume only if ATR < 130% pre-event ATR

# ── SMTP alerts (COMP-1) ──────────────────────────────────────────────────────
SMTP_HOST          = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT          = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER          = os.getenv("SMTP_USER", "")
SMTP_PASS          = os.getenv("SMTP_PASS", "")
ALERT_RECIPIENT    = os.getenv("ALERT_RECIPIENT", "")
SMTP_TIMEOUT_SEC   = 10  # COMP-1: never use blocking default

# ── HorizonFX Economic Calendar (ADD-1) ───────────────────────────────────────
HORIZONFX_BASE_URL    = os.getenv(
    "HORIZONFX_BASE_URL", "https://economic-calendar.horizonfx.id/events"
)
HORIZONFX_TIMEOUT_SEC = 10
CALENDAR_PULL_RETRIES = 3
CALENDAR_PULL_HOUR_IST   = 0   # midnight IST
CALENDAR_PULL_MINUTE_IST = 0

# Hardcoded fallback event windows (ADD-1)
# IMPORTANT: Covers SCHEDULED releases only.
# Unscheduled Fed speeches are NOT catchable — logged as limitation.
# Values are approximate IST times (UTC+5:30). Verify against real calendar.
HARDCODED_EVENT_PATTERNS = [
    # NFP: first Friday of every month, 18:30 IST (08:00 ET)
    {"name": "NFP",           "weekday": 4,  "week_of_month": 1,
     "hour_ist": 18, "minute_ist": 30, "impact": "HIGH"},
    # FOMC: 8 meetings/year, Wed ~01:30 IST next-day (19:00 ET prior day)
    # Approximate only — exact dates must come from live calendar
    {"name": "FOMC_APPROX",   "weekday": 2,  "week_of_month": None,
     "hour_ist": 1, "minute_ist": 30, "impact": "HIGH",
     "note": "APPROXIMATE — live calendar must confirm"},
    # CPI: ~10th of each month, 18:30 IST
    {"name": "CPI",           "day_of_month": 10,
     "hour_ist": 18, "minute_ist": 30, "impact": "HIGH"},
    # PPI: ~11th of each month, 18:30 IST
    {"name": "PPI",           "day_of_month": 11,
     "hour_ist": 18, "minute_ist": 30, "impact": "HIGH"},
    # Retail Sales: ~15th of each month, 18:30 IST
    {"name": "RETAIL_SALES",  "day_of_month": 15,
     "hour_ist": 18, "minute_ist": 30, "impact": "HIGH"},
]

# ── Regime Engine ─────────────────────────────────────────────────────────────
REGIME_STALENESS_SEC         = 1200   # 20 minutes → NO_TRADE if exceeded
REGIME_HYSTERESIS_COUNT      = 3      # same state for 3 consecutive readings to flip
REGIME_JOB_INTERVAL_MIN      = 15     # APScheduler interval
REGIME_JOB_COALESCE          = True   # skip missed runs
REGIME_JOB_MAX_INSTANCES     = 1      # never run two simultaneously

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD        = 14
ATR_MAMODE        = "RMA"   # Wilder's smoothing — pinned. Matches TV and MT5.
ATR_LOOKBACK_DAYS = 30

# ATR percentile reference calibration
# ── ATR Thresholds (authoritative — calibrate_atr.py reads these) ──────────
ATR_PCT_QUIET_REF            = 30    # S2 gate floor
ATR_PCT_SUPER_THRESHOLD      = 55    # SUPER_TRENDING starts here
ATR_PCT_UNSTABLE_THRESHOLD   = 85    # UNSTABLE above here
ATR_PCT_NO_TRADE_THRESHOLD   = 95    # NO_TRADE above here

# ── S3 Stop Hunt ────────────────────────────────────────────────────────────
S3_SWEEP_THRESHOLD_ATR  = 0.3   # price must sweep 0.3×ATR below range_low
S3_STOP_ATR_MULT        = 0.5   # stop = sweep_low - 0.5×ATR
S3_RECLAIM_OFFSET_PTS   = 2.0   # BUY STOP 2pts above reclaim candle high
S3_WINDOW_CANDLES       = 3     # reclaim must happen within 3 M15 bars (45 min)

# ── S6 Asian Breakout ───────────────────────────────────────────────────────
S6_MIN_RANGE_PTS        = 8.0   # skip if Asian range < 8pts
S6_STOP_ATR_MULT        = 0.5   # stop = 0.5×ATR14 from entry
S6_BREAKOUT_DIST_PTS    = 2.0   # pts beyond range boundary before entry

# ── S7 Daily Structure ──────────────────────────────────────────────────────
S7_MIN_RANGE_ATR_RATIO  = 0.75  # skip if prev_day_range < 0.75×daily_ATR14
S7_ENTRY_OFFSET_PTS     = 5.0   # BUY STOP at prev_day_high + 5pts + spread
S7_STOP_OFFSET_PTS      = 10.0  # stop at prev_day_low - 10pts (LONG)
S7_SIZE_MULTIPLIER      = 0.5   # 0.5x base lot — wider stops, smaller size

# ── Portfolio Correlation ───────────────────────────────────────────────────
PORTFOLIO_CORR_THRESHOLD      = 0.65  # alert if any strategy pair exceeds this
PORTFOLIO_CORR_CHECK_EVERY_N  = 10    # run after every N new closed trades

# ── KS4 countdown ────────────────────────────────────────────────────────────
KS4_REDUCED_TRADE_COUNT      = KS4_REDUCED_TRADES  # CONSOLIDATED: was duplicate of KS4_REDUCED_TRADES

# ── Portfolio Risk Brain ──────────────────────────────────────────────────────
MAX_DAILY_VAR_PCT            = 0.02  # 2% of account all strategies combined
MAX_SESSION_LOTS             = 0.15  # total open lots per session
PARTIAL_FILL_THRESHOLD       = 0.80  # >= 80% fill = treat as full fill (v1.1)

# ── Signal Engine ─────────────────────────────────────────────────────────────
MAX_S1_FAMILY_ATTEMPTS = 4   # EXP-8 FIX: was 3 — 3rd attempt often IS the real breakout
MAX_S1F_ATTEMPTS       = 1   # S1f independent daily limit (G4 Fix)

BREAKOUT_DIST_PCT   = 0.12   # 12% of range_size for S1 confirmation
HUNT_THRESHOLD_PCT  = 0.08   # 8% of range_size for stop-hunt detection
MIN_RANGE_SIZE_PTS  = 10     # minimum viable pre-London range
CHASE_MAX_PCT       = 0.08   # max 8% of range_size chase budget

S1_TIME_KILL_HOUR_LONDON   = 16
S1_TIME_KILL_MIN_LONDON    = 30
S1F_TIME_KILL_HOUR_NY      = 13
S1F_TIME_KILL_MIN_NY       = 0

S1B_RESET_CANDLES      = 6    # B3 Fix: reset failed_breakout_flag after 6 M15 candles
S1C_RESET_CANDLES      = 3    # reset stop_hunt flags after 3 M15 candles
S1B_AUTO_RESET_CANDLES = S1B_RESET_CANDLES   # alias used by signal_engine
S1C_AUTO_RESET_CANDLES = S1C_RESET_CANDLES   # alias used by signal_engine

M5_REENTRY_MAX_SUPER  = 8
M5_REENTRY_MAX_NORMAL = 5
M5_LOSS_PAUSE_COUNT   = 5   # EXP-7 FIX: was 3 — too conservative for high-freq addon with small lots
M5_LIMIT_EXPIRY_MIN   = 5   # B4 Fix: M5 limit orders expire after 1 candle

# ── Position Management ───────────────────────────────────────────────────────
PARTIAL_EXIT_R         = 2.0   # EXP-3 FIX: was 1.0 — let winners run further before partial
BE_ACTIVATION_R        = 1.5   # EXP-4 FIX: was 0.75 — too aggressive, normal retracements stop out at entry
ATR_TRAIL_MULTIPLIER   = 2.5   # EXP-5 FIX: was 1.5 — too tight for gold, normal M15 pullbacks clip trail

S1D_STOP_POINTS_MIN    = 15    # LOOP-3 FIX: was 10 — too tight for XAUUSD M5
S1D_STOP_POINTS_MAX    = 20    # LOOP-3 FIX: was 12
S1D_TARGET_POINTS_MIN  = 35
S1D_TARGET_POINTS_MAX  = 50
S1D_SIZE_PCT           = 0.50  # 0.5× normal lot
S1E_SIZE_PCT           = 0.50  # pyramid add: 50% of original S1 lots

S1F_STOP_POINTS        = 15    # tighter — late session
S1F_TARGET_POINTS_MIN  = 35

# ── Risk Engine ───────────────────────────────────────────────────────────────
BASE_RISK_PHASE_1      = 0.010   # 1.0% per trade
BASE_RISK_PHASE_2      = 0.020   # 2.0% per trade (unlocked after 50 proven trades)
BASE_RISK_PHASE_3_MAX  = 0.040   # absolute maximum ever — 4%
V1_LOT_HARD_CAP        = 0.50    # V1 cap — raise only after Phase 2 gate confirmed
FAMILY_EXPOSURE_MAX    = 0.015   # 1.5% combined trend family max

PHASE_1_TRADE_GATE     = 50
PHASE_2_WIN_RATE_MIN   = 0.45
PHASE_2_EXPECTANCY_MIN = 0.15    # +0.15R minimum
PHASE_2_MAX_DD         = 0.15    # max drawdown 15%
PHASE_3_MIN_SHARPE     = 1.0

# ── Macro Bias (ADD-3: TLT threshold is Phase 0 calibration placeholder) ─────
# Actual runtime value is read from system_config table on startup.
# This default is ONLY used if system_config row is missing (should never happen).
TLT_SLOPE_THRESHOLD_DEFAULT = 0.15  # PHASE 0 CALIBRATION REQUIRED
MACRO_PULL_HOUR_IST         = 9
MACRO_PULL_MINUTE_IST       = 0
MACRO_PROMOTE_DELTA_PP      = 8.0   # promote after >8pp A+/standard delta
MACRO_PROMOTE_TRADE_MIN     = 50    # minimum trades before promotion decision

# ── DXY correlation ───────────────────────────────────────────────────────────
DXY_CORR_LOOKBACK         = 50
DXY_CORR_SUPER_THRESHOLD  = -0.70  # macro_boost gate for SUPER_TRENDING
DXY_CORR_NEUTRAL_FALLBACK = -0.50  # used if calculation fails
# F4: DXY EWMA variance threshold for stability check
# When DXY returns have EWMA variance > this, macro_boost is disabled
# Typical DXY hourly return variance: 0.0001-0.0004 (0.01-0.02% hourly moves)
# During whipsaw: can spike to 0.0009+ (0.03%+ hourly moves)
DXY_VARIANCE_SPIKE_THRESHOLD = 0.0008  # squared return units

# ── Spread guard S2 ───────────────────────────────────────────────────────────
S2_MAX_SPREAD_RATIO = 1.5   # S2 only enters if spread <= 1.5× avg

# ── Execution ─────────────────────────────────────────────────────────────────
ORDER_DEVIATION_POINTS     = 10
EMERGENCY_DEVIATION_POINTS = 20
HEARTBEAT_INTERVAL_SEC     = 60

# ── Conviction level ─────────────────────────────────────────────────────────
CONVICTION_CLEAR_HORIZON_MIN = 90   # no events within 90 min for A+
CONVICTION_SPREAD_RATIO_MAX  = 1.2  # spread <= 1.2× avg for A+

# This threshold PREVENTS trades when conditions are genuinely poor.
MIN_CONDITION_MULTIPLIER    = 0.35   # Tune after 50+ live trades

# ── R3 — Calendar Momentum ────────────────────────────────────────────────────
R3_MIN_DELAY_MIN            = 5      # Min minutes post-event before R3 arms
R3_ARMED_WINDOW_MIN         = 35     # R3 expires 35 min post-event
R3_MAX_HOLD_MIN             = 30     # Force-close after 30 min from entry
R3_STOP_ATR_MULT            = 0.50   # Stop = 0.5 × H1 ATR from entry
R3_TP_ATR_MULT              = 0.75   # TP   = 0.75 × H1 ATR → 1.5:1 RR
R3_SEVERITY_THRESHOLD       = 35     # Severity >= 35 → R3 takes it; < 35 → S8
R3_DIRECTION_MIN_MOVE_RATIO = 0.05   # Min move = 0.05 × H1 ATR to confirm direction

# ── S4 — London Pullback ──────────────────────────────────────────────────────
S4_SESSION_START_HOUR_UTC   = 7      # London session start (UTC)
S4_SESSION_END_HOUR_UTC     = 12     # London session end / S4 entry window end
S4_TOUCH_ATR_FACTOR         = 0.10   # EMA20 touch zone = 0.1 × H1 ATR proximity
S4_STOP_ATR_BUFFER          = 0.30   # Stop beyond touch bar low/high = 0.3 × H1 ATR
S4_TP_RR_RATIO              = 1.50   # TP = 1.5 × stop distance
S4_ADX_MIN_THRESHOLD        = 20     # ADX H4 must be > 20 for S4 to fire
S4_HARD_EXIT_HOUR_UTC       = 16     # Force-close any S4 position at 16:00 UTC

# ── S5 — NY Compression Breakout ─────────────────────────────────────────────
S5_COMPRESSION_RATIO        = 0.70   # London range < 0.70 × D1 ATR14 = compressed
S5_ENTRY_START_HOUR_UTC     = 12     # Entry window opens at London close
S5_ENTRY_END_HOUR_UTC       = 15     # Entry window closes at 15:00 UTC
S5_STOP_ATR_BUFFER          = 0.30   # Stop beyond London extreme = 0.3 × H1 ATR
S5_TP_RANGE_MULT            = 1.00   # TP = 1.0 × London range from entry
S5_HARD_EXIT_HOUR_UTC       = 22     # Force-close any S5 position at 22:00 UTC

# ── Edge Decay Detection (1A-3) ───────────────────────────────────────────────
EDGE_WARNING_EXPECTANCY  = 0.10   # below system minimum of 0.15R
EDGE_CRITICAL_EXPECTANCY = 0.05   # near zero edge
EDGE_WARNING_WR          = 0.40   # below system minimum of 45%
EDGE_CRITICAL_WR         = 0.35   # severely degraded
EDGE_MIN_TRADES          = 30     # minimum trades for detection
