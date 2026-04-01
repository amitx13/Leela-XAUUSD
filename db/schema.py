"""
db/schema.py — All CREATE TABLE and CREATE SCHEMA statements.

Single source of truth for all database objects.
Run create_all_schemas() once during Phase 0 setup.
Never alter tables manually — add a migration function below any schema change.

Schemas:
  market_data    — price data, macro signals, economic events
  system_state   — regime log, spread log, trades, performance, persistent state
  public         — system_config (ADD-3: calibration values)

All timestamps stored as TIMESTAMPTZ (UTC). Convert to IST only at display time.
"""
from sqlalchemy import text
from db.connection import engine
from utils.logger import log_event


# ─────────────────────────────────────────────────────────────────────────────
# Schema creation SQL
# ─────────────────────────────────────────────────────────────────────────────

SCHEMAS = ["market_data", "system_state"]

MARKET_DATA_OHLCV = """
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
    id          BIGSERIAL    PRIMARY KEY,
    timestamp   TIMESTAMPTZ  NOT NULL,
    timeframe   VARCHAR(5)   NOT NULL,    -- 'M5', 'M15', 'H1', 'H4'
    open        DECIMAL(10,2) NOT NULL,
    high        DECIMAL(10,2) NOT NULL,
    low         DECIMAL(10,2) NOT NULL,
    close       DECIMAL(10,2) NOT NULL,
    volume      BIGINT,                   -- tick volume — never used as signal gate
    spread      DECIMAL(8,2),
    created_at  TIMESTAMPTZ  DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uix_ohlcv_ts_tf
    ON market_data.ohlcv (timestamp, timeframe);
"""

# ADD-5: macro_proxy_instrument column on every row
MARKET_DATA_MACRO_SIGNALS = """
CREATE TABLE IF NOT EXISTS market_data.macro_signals (
    id                      BIGSERIAL    PRIMARY KEY,
    date                    DATE         NOT NULL UNIQUE,
    proxy_close             DECIMAL(10,4),              -- TLT or TIP close
    proxy_3d_slope          DECIMAL(12,6),
    proxy_vs_10d_ma         DECIMAL(12,6),
    macro_bias_label        VARCHAR(20),                -- LONG/SHORT/BOTH/NONE_PERMITTED
    macro_proxy_instrument  VARCHAR(5)   NOT NULL DEFAULT 'TLT',  -- ADD-5
    created_at              TIMESTAMPTZ  DEFAULT now()
);
COMMENT ON COLUMN market_data.macro_signals.macro_proxy_instrument IS
    'ADD-5: instrument used for macro bias calc. TLT until >50 trades, then TIP if promoted.';
"""

MARKET_DATA_ECONOMIC_EVENTS = """
CREATE TABLE IF NOT EXISTS market_data.economic_events (
    event_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    event_name      VARCHAR(100) NOT NULL,
    scheduled_utc   TIMESTAMPTZ  NOT NULL,
    impact_level    VARCHAR(10)  NOT NULL DEFAULT 'HIGH',
    released_flag   BOOLEAN,             -- NULL = unknown (fallback mode)
    source          VARCHAR(20)  NOT NULL DEFAULT 'HORIZONFX',  -- ADD-1
    fallback_used   BOOLEAN      NOT NULL DEFAULT FALSE,          -- ADD-1
    created_at      TIMESTAMPTZ  DEFAULT now()
);
COMMENT ON COLUMN market_data.economic_events.released_flag IS
    'NULL when fallback calendar is active — unknown release status (ADD-1).';
COMMENT ON COLUMN market_data.economic_events.fallback_used IS
    'TRUE if row was generated from hardcoded pattern list, not live API (ADD-1).';
"""

SYSTEM_STATE_REGIME_LOG = """
CREATE TABLE IF NOT EXISTS system_state.regime_log (
    id                  BIGSERIAL    PRIMARY KEY,
    timestamp           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    adx_h4              DECIMAL(8,4),
    atr_pct_h1          DECIMAL(8,4),
    session             VARCHAR(20),
    regime_state        VARCHAR(20)  NOT NULL,
    size_multiplier     DECIMAL(5,3),
    dxy_corr            DECIMAL(8,4),
    macro_boost         BOOLEAN,
    regime_age_seconds  INTEGER,
    combined_exposure_pct DECIMAL(8,5)  -- family exposure snapshot (S1+S1d+S1e)
);
COMMENT ON COLUMN system_state.regime_log.combined_exposure_pct IS
    'Combined S1+S1d+S1e live exposure as fraction of equity. Max 1.5%.';
"""

SYSTEM_STATE_SPREAD_LOG = """
CREATE TABLE IF NOT EXISTS system_state.spread_log (
    id          BIGSERIAL    PRIMARY KEY,
    timestamp   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    spread_pts  DECIMAL(8,2) NOT NULL,
    session     VARCHAR(20)
);
"""

SYSTEM_STATE_PERSISTENT = """
CREATE TABLE IF NOT EXISTS system_state.system_state_persistent (
    id          SERIAL      PRIMARY KEY,
    saved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    state_date  DATE        NOT NULL UNIQUE,

    -- Daily counters (reset to 0 when date changes — C4 Fix)
    consecutive_m5_losses    INTEGER     DEFAULT 0,
    s1_family_attempts_today INTEGER     DEFAULT 0,
    s1f_attempts_today       INTEGER     DEFAULT 0,
    daily_net_pnl_pct        DECIMAL(8,5) DEFAULT 0,
    daily_commission_paid    DECIMAL(8,4) DEFAULT 0,

    -- Rolling state (persist across days)
    consecutive_losses        INTEGER     DEFAULT 0,
    peak_equity               DECIMAL(12,2),
    failed_breakout_flag      BOOLEAN     DEFAULT FALSE,
    failed_breakout_direction VARCHAR(5),
    last_s1_direction         VARCHAR(5),
    stop_hunt_detected        BOOLEAN     DEFAULT FALSE,
    trading_enabled           BOOLEAN     DEFAULT TRUE,
    shutdown_reason           VARCHAR(100),

    -- Warm-start regime (v1.1 DB migration)
    current_regime        VARCHAR(20)  DEFAULT 'NO_TRADE',
    regime_calculated_at  TIMESTAMPTZ  DEFAULT now(),
    size_multiplier       DECIMAL(5,3) DEFAULT 0.0,

    -- v1.1 state additions
    ks4_reduced_trades_remaining INTEGER     DEFAULT 0,
    reversal_family_occupied     BOOLEAN     DEFAULT FALSE,
    s1b_pending_ticket           BIGINT      DEFAULT NULL,
    s1d_ema_touched_today        BOOLEAN     DEFAULT FALSE,
    s1d_fired_today              BOOLEAN     DEFAULT FALSE,
    s3_sweep_candle_time         TIMESTAMPTZ DEFAULT NULL
);
COMMENT ON COLUMN system_state.system_state_persistent.consecutive_losses IS
'ROLLING: a WIN resets to 0. Midnight IST does NOT reset. Streak spans overnight.';
"""

SYSTEM_STATE_TRADES = """
CREATE TABLE IF NOT EXISTS system_state.trades (
    -- Identity
    trade_id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_type           VARCHAR(25) NOT NULL,
    strategy_version      VARCHAR(10),
    mt5_ticket            BIGINT,
    campaign_id           UUID,
    s1b_parent_campaign_id UUID,

    -- Direction + timing
    direction             VARCHAR(5)  NOT NULL,
    entry_time            TIMESTAMPTZ,
    exit_time             TIMESTAMPTZ,
    entry_price           DECIMAL(10,3),
    exit_price            DECIMAL(10,3),

    -- Stops (B5 Fix: two fields, original NEVER changes)
    stop_price_original   DECIMAL(10,3),
    stop_price_current    DECIMAL(10,3),

    -- Sizing
    lot_size              DECIMAL(8,4),

    -- P&L
    pnl_gross_dollars     DECIMAL(10,2),
    commission_entry      DECIMAL(8,4),
    commission_exit       DECIMAL(8,4),
    total_commission      DECIMAL(8,4),
    pnl_net_dollars       DECIMAL(10,2),
    pnl_points            DECIMAL(10,2),
    r_multiple            DECIMAL(8,4),
    outcome               VARCHAR(10),       -- WIN / LOSS / BREAKEVEN
    exit_reason           VARCHAR(30),
    time_in_trade_min     DECIMAL(8,2),

    -- Context at entry
    regime_at_entry       VARCHAR(20),
    size_multiplier_used  DECIMAL(5,3),
    adx_h4_at_entry       DECIMAL(8,4),
    atr_h1_percentile     DECIMAL(8,4),
    session               VARCHAR(20),
    regime_age_seconds    INTEGER,
    macro_bias            VARCHAR(20),
    tlt_3d_slope          DECIMAL(10,6),
    dxy_corr_at_entry     DECIMAL(8,4),
    macro_boost_at_entry  BOOLEAN,
    macro_proxy_at_entry  VARCHAR(5),        -- ADD-5: TLT or TIP

    -- Range context
    asian_range_high      DECIMAL(10,3),
    asian_range_low       DECIMAL(10,3),
    asian_range_size_pts  DECIMAL(8,2),

    -- Signal context
    conviction_level      VARCHAR(15),       -- A_PLUS / STANDARD / AFTER_STREAK
    stop_hunt_detected    BOOLEAN DEFAULT FALSE,
    failed_breakout_trade BOOLEAN DEFAULT FALSE,
    post_time_kill_reentry BOOLEAN DEFAULT FALSE,
    s1_family_attempt_num INTEGER,
    london_hour_at_entry  INTEGER,
    event_proximity_min   INTEGER,
    equity_at_entry       DECIMAL(12,2),
    risk_pct_used         DECIMAL(8,5),

    -- Execution quality
    spread_at_entry       DECIMAL(8,2),
    spread_vs_avg_ratio   DECIMAL(6,3),
    slippage_points       DECIMAL(8,2),
    order_type_used       VARCHAR(10),       -- STOP / LIMIT / MARKET

    -- Position management flags
    partial_exit_done     BOOLEAN DEFAULT FALSE,
    be_activated          BOOLEAN DEFAULT FALSE,
    pyramid_add_done      BOOLEAN DEFAULT FALSE,
    m5_reentry_count      INTEGER DEFAULT 0,

    created_at            TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_trades_mt5_ticket  ON system_state.trades (mt5_ticket);
CREATE INDEX IF NOT EXISTS ix_trades_campaign_id ON system_state.trades (campaign_id);
CREATE INDEX IF NOT EXISTS ix_trades_exit_time   ON system_state.trades (exit_time);
"""

SYSTEM_STATE_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS system_state.performance (
    id              BIGSERIAL    PRIMARY KEY,
    date            DATE         NOT NULL UNIQUE,
    gross_pnl       DECIMAL(10,2) DEFAULT 0,
    total_commission DECIMAL(10,2) DEFAULT 0,
    net_pnl         DECIMAL(10,2) DEFAULT 0,
    trade_count     INTEGER       DEFAULT 0,
    win_count       INTEGER       DEFAULT 0,
    peak_equity     DECIMAL(12,2),
    drawdown_pct    DECIMAL(8,5),
    updated_at      TIMESTAMPTZ   DEFAULT now()
);
"""


# system_config table (ADD-3) — calibration values and operational parameters
# Read on startup via initialize_system(). Never hardcode calibration values in code.
SYSTEM_CONFIG = """
CREATE TABLE IF NOT EXISTS public.system_config (
    key         VARCHAR(50)  PRIMARY KEY,
    value       TEXT         NOT NULL,
    set_at      TIMESTAMPTZ  DEFAULT now(),
    set_by      VARCHAR(50),   -- 'PHASE_0_CALIBRATION', 'ANALYST_REVIEW', 'DEFAULT'
    notes       TEXT
);

-- Seed default calibration values (Phase 0 will overwrite these)
INSERT INTO public.system_config (key, value, set_by, notes) VALUES
    ('TLT_SLOPE_THRESHOLD',     '0.15',  'DEFAULT',
     'PHASE 0 CALIBRATION REQUIRED. Verify against TLT/XAU 2022-2024.'),
    ('MACRO_PROXY_INSTRUMENT',  'TLT',   'DEFAULT',
     'Switch to TIP if macro_bias delta never exceeds 8pp after 50 trades.'),
    ('ATR_PCT_QUIET_REF',        '30',    'DEFAULT',
     'PHASE 0 CALIBRATION REQUIRED. ~quiet ATR percentile (8-12 pts).'),
    ('ATR_PCT_SUPER_THRESHOLD',  '55',    'DEFAULT',
     'PHASE 0 CALIBRATION REQUIRED. SUPER_TRENDING lower bound.'),
    ('ATR_PCT_UNSTABLE_THRESHOLD','85',   'DEFAULT',
     'PHASE 0 CALIBRATION REQUIRED. UNSTABLE lower bound.'),
    ('ATR_PCT_NO_TRADE_THRESHOLD','94',   'DEFAULT',
     'PHASE 0 CALIBRATION REQUIRED. NO_TRADE lower bound.'),
    ('BASE_RISK_PHASE',          '1',     'DEFAULT',
     'Current phase: 1=1%, 2=2%. Change only after Phase 2 gate confirmed.'),
    ('STRATEGY_VERSION',         'V1',    'DEFAULT',
     'Incremented on any strategy parameter change.')
ON CONFLICT (key) DO NOTHING;
"""


def create_all_schemas() -> None:
    """
    Run once during Phase 0 setup.
    Creates all schemas, tables, indexes, and seeds system_config defaults.
    Safe to re-run — all statements use IF NOT EXISTS.
    """
    with engine.begin() as conn:
        # Create schemas
        for schema in SCHEMAS:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            log_event("SCHEMA_CREATED", schema=schema)

        # Create all tables
        tables = [
            ("market_data.ohlcv",                       MARKET_DATA_OHLCV),
            ("market_data.macro_signals",               MARKET_DATA_MACRO_SIGNALS),
            ("market_data.economic_events",             MARKET_DATA_ECONOMIC_EVENTS),
            ("system_state.regime_log",                 SYSTEM_STATE_REGIME_LOG),
            ("system_state.spread_log",                 SYSTEM_STATE_SPREAD_LOG),
            ("system_state.trades",                     SYSTEM_STATE_TRADES),
            ("system_state.performance",                SYSTEM_STATE_PERFORMANCE),
            ("system_state.system_state_persistent",    SYSTEM_STATE_PERSISTENT),
            ("public.system_config",                    SYSTEM_CONFIG),
        ]

        for table_name, ddl in tables:
            conn.execute(text(ddl))
            log_event("TABLE_CREATED", table=table_name)

    log_event("ALL_SCHEMAS_CREATED")


def get_config_value(key: str, default: str | None = None) -> str | None:
    """
    Read a calibration value from system_config.
    Used by initialize_system() to load Phase 0 calibrated thresholds.
    """
    from db.connection import execute_query
    rows = execute_query(
        "SELECT value FROM public.system_config WHERE key = :key",
        {"key": key}
    )
    if rows:
        return rows[0]["value"]
    return default


def verify_schema() -> bool:
    """Check all expected tables exist. Returns True if all present."""
    expected = [
        ("market_data",  "ohlcv"),
        ("market_data",  "macro_signals"),
        ("market_data",  "economic_events"),
        ("system_state", "regime_log"),
        ("system_state", "spread_log"),
        ("system_state", "trades"),
        ("system_state", "performance"),
        ("system_state", "system_state_persistent"),
        ("public",       "system_config"),
    ]
    missing = []
    with engine.connect() as conn:
        for schema, table in expected:
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :s AND table_name = :t
            """), {"s": schema, "t": table}).fetchone()
            if result:
                print(f"  ✅ {schema}.{table}")
            else:
                print(f"  ❌ {schema}.{table} — MISSING")
                missing.append(f"{schema}.{table}")
    if missing:
        log_event("SCHEMA_VERIFY_FAILED", missing=missing)
        return False
    log_event("SCHEMA_VERIFY_OK", tables=len(expected))
    return True