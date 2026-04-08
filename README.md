# Leela XAUUSD Algorithmic Trading System

## Overview

Leela is a sophisticated algorithmic trading system designed specifically for XAUUSD (Gold) trading on MetaTrader 5. The system implements multiple trading strategies across different market sessions with advanced risk management, real-time analytics, and comprehensive monitoring capabilities.

**Key Features:**
- **13 Trading Strategies** across different market conditions and sessions
- **6-State Regime Engine** for dynamic market classification
- **Advanced Risk Management** with 7 kill switches and portfolio-level controls
- **Multi-Session Coverage** (Asian, London, New York, and overlaps)
- **Economic Calendar Integration** for event-driven trading
- **Real-time Analytics** and performance monitoring
- **Complete Audit Trail** for every trade decision
- **Phase-based Progression** from conservative to aggressive sizing

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Quick Start Guide](#quick-start-guide)
3. [Trading Strategies](#trading-strategies)
4. [Risk Management](#risk-management)
5. [Market Regimes](#market-regimes)
6. [Installation & Setup](#installation--setup)
7. [Configuration](#configuration)
8. [Daily Operations](#daily-operations)
9. [Monitoring & Analytics](#monitoring--analytics)
10. [Troubleshooting](#troubleshooting)
11. [System Components](#system-components)

---

## System Architecture

The system follows a layered architecture where each layer has specific responsibilities and safety checks:

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET DATA LAYER                         │
│  MT5 (real-time) + External APIs (economic calendar)        │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                    DATA ENGINE                              │
│  OHLCV processing, spread tracking, indicator calculations  │
│  Economic events, DXY correlation, ATR computations         │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                   REGIME ENGINE                             │
│  6-state classification: NO_TRADE → SUPER_TRENDING          │
│  ADX H4, ATR percentile H1, DXY correlation, hysteresis    │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                   SIGNAL ENGINE                            │
│  10 strategies with ATR-based stops, volume filters,       │
│  ADX trend bias, TP targets on all strategies              │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                    RISK ENGINE                              │
│  Position sizing with 50% reduction floor, conviction boost │
│  7 kill switches, compound condition gate                   │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                PORTFOLIO RISK BRAIN                         │
│  Cross-strategy exposure, VAR, same-family correlation kill │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│               EXECUTION ENGINE                              │
│  Order placement with TP for all strategies, chase logic    │
│  Spread-adjusted BUY STOPs, R3 independent family          │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                 TRUTH ENGINE                                │
│  Performance analytics, EWMA conviction, edge decay monitor │
│  Starvation tracking, weekly reviews                        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Data Engine**: Handles all market data feeds from MT5 and external APIs
- **Regime Engine**: Classifies market conditions into 6 states with hysteresis
- **Signal Engine**: Generates trade signals from 10 different strategies
- **Risk Engine**: Manages position sizing and implements safety mechanisms
- **Execution Engine**: Handles order placement and position management
- **Truth Engine**: Monitors performance and system health

---

## Quick Start Guide

### For Beginners

If you're new to algorithmic trading or this system, follow these steps:

1. **Prerequisites Check**
   - Ensure you have MetaTrader 5 installed
   - Have a funded trading account (demo or live)
   - Python 3.11+ installed on your system

2. **System Setup**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd xauusd_algo
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your MT5 and database details
   nano .env
   ```

4. **Database Setup**
   ```bash
   # Start PostgreSQL (using Docker)
   docker-compose up -d
   
   # Initialize database schema
   python -c "from db.init_db import init_database; init_database()"
   ```

5. **First Run (Paper Trading)**
   ```bash
   # Run pre-session checklist
   python main.py --checklist
   
   # Start in paper trading mode
   python main.py --paper
   ```

### Daily Startup Routine

1. **Pre-Market (12:00 IST)**
   ```bash
   python main.py --checklist
   ```
   - Verifies MT5 connection
   - Checks database connectivity
   - Validates contract specifications
   - Reviews recent performance

2. **System Start (Before 12:25 IST)**
   ```bash
   python main.py --live  # For live trading
   # or
   python main.py --paper  # For paper trading
   ```

3. **Monitor**
   - Watch for WARM_START completion
   - Monitor order placement

---

## Trading Strategies

The system implements 13 distinct strategies across different market conditions:

### Phase 1 Strategies (Core)

#### S1 Family - London Momentum Strategies

**S1_LONDON_BRK** - Primary London Breakout
- **Session**: London (08:00-17:00 London local time, DST-adjusted)
- **Setup**: Pre-London range (00:00-07:55 UTC)
- **Entry**: BUY/SELL STOP at range boundaries + 12% breakout distance
- **Volume Filter**: Rejects breakouts with low volume confirmation
- **Stop Loss**: ATR-based (max(0.3×H1 ATR, 5.0 points))
- **Take Profit**: 2.5R from entry
- **Max per Day**: 4 attempts

**S1C_STOP_HUNT** - Stop Hunt Pre-Signal Detection
- **Purpose**: Detects potential liquidity sweeps before London open
- **Trigger**: Price touches EMA20 on M15 during pre-London session
- **Action**: Sets state flag to reduce S1 confirmation threshold from 3 to 2 touches
- **Reset**: Clears at London open if no sweep confirmed

**S1B_FAILED_BRK** - Failed Breakout Reversal
- **Trigger**: S1 fills and hits stop loss
- **Entry**: STOP order beyond false breakout extreme
- **Auto-Reset**: Clears after 6 M15 candles if no reversal

**S1D_PYRAMID** - M5 Pullback Re-entry
- **Trigger**: S1 position open + M5 body close above/below EMA20
- **Entry**: LIMIT at EMA20 with 5-minute expiry
- **Size**: 0.5× base lot
- **Max Re-entries**: SUPER=8, NORMAL=5

**S1E_PYRAMID** - Trend Continuation Add
- **Trigger**: Partial exit done + BE activated
- **Entry**: Market order
- **Size**: 0.5× original S1 lots

**S1F_POST_TK** - Post Time-Kill Re-entry
- **Session**: NY (after London 16:30 time kill)
- **Entry**: LIMIT at M5 EMA20 with direction validation

#### S2_MEAN_REV - Range Reversion
- **Regime Gate**: RANGING_CLEAR only
- **Signal**: H1 close > 1.5×ATR from 20 EMA + RSI confirmation
- **Entry**: LIMIT at EMA20
- **Stop**: 1.5×ATR beyond extreme

#### S3_STOP_HUNT_REV - Liquidity Sweep Reversal
- **Sessions**: London + early NY (08:00-16:30 UTC)
- **Range Source**: Rolling 3-hour M15 range
- **Trigger**: Price sweeps range by >0.3×ATR then reclaims within 3 bars
- **Entry**: BUY STOP 2pts above reclaim candle high

#### S6_ASIAN_BRK - Asian Session Breakout
- **Session**: Asian (00:00-05:30 UTC setup)
- **Range**: Asian high/low (00:00-05:30 UTC)
- **ADX Trend Filter**: Only trending direction in strong trends
- **Expiry**: 08:00 UTC

#### S7_DAILY_STRUCT - Daily Structure Breakout
- **Setup**: Previous day OHLC (midnight reset)
- **Filter**: Previous day range >0.75×daily ATR
- **ADX Trend Filter**: Same as S6
- **Size**: 0.5× base lot

### Phase 2 Strategies (Advanced)

#### R3_CAL_MOMENTUM - Economic Calendar Momentum
- **Trigger**: High-impact economic event release
- **Wait**: 5 minutes post-release
- **Volatility Filter**: Post-event move must exceed 0.3×H1 ATR
- **Entry**: Market order in direction of first M5 close
- **Hold Limit**: 30 minutes
- **Family**: Independent (coexists with trend positions)

#### S4_LONDON_PULL - London Pullback Continuation
- **Session**: London (07:00-12:00 UTC)
- **Regime Gate**: Trending (ADX > 20 AND increasing)
- **Entry**: LIMIT at M15 EMA20 with 15-minute expiry
- **Hard Exit**: 16:00 UTC

#### S5_NY_COMPRESS - NY Compression Breakout
- **Session**: NY (12:00-15:00 UTC)
- **Trigger**: London range < 0.70×D1 ATR14 (compressed)
- **Entry**: BUY/SELL STOP 2pts beyond London boundary
- **Hard Exit**: 22:00 UTC

#### S8_ATR_SPIKE - Flash Spike Continuation
- **Trigger**: M15 candle range > 1.5×ATR(14,H1)
- **Confirmation**: Next M15 close past spike midpoint
- **Entry**: Market order at current bid/ask
- **Size**: 0.5× base lot

---

## Risk Management

### Kill Switches (KS1-KS7)

| Switch | Trigger | Action | Recovery |
|--------|---------|--------|-----------|
| KS1 | Stop modification against trade | Reject modification | Manual review |
| KS2 | Spread >2.5× 24h median | Reject order | Wait for spread normalization |
| KS3 | Daily loss > -7% | Block new entries | Next day reset |
| KS4 | 4 consecutive losses | Reduce size 50% for 3 trades | Auto-recovery |
| KS5 | Weekly loss > -15% | Block entries this week | Next week reset |
| KS6 | Drawdown > 20% from peak | Emergency shutdown | Manual review |
| KS7 | High-impact event proximity | Block entries 45min pre/20min post | Auto-resume |

### Position Sizing Algorithm

**Step-by-Step Calculation:**
1. **Base Risk**: 1.0% account equity (Phase 1) or 2.0% (Phase 2)
2. **Conviction Adjustment**: 
   - A_PLUS: ×1.25 (clear horizon + regime alignment)
   - OBSERVATION: ×0.75 (macro misalignment)
   - STANDARD: ×1.0 (default)
3. **KS4 Recovery**: ×0.5 for 3 trades after 4-loss streak
4. **Event Severity**: 0.5-1.5× based on economic event risk score
5. **Spread Penalty**: 0.7-1.0× based on current vs median spread ratio
6. **Volatility Scalar**: 0.7-1.3× from EWMA ATR percentile
7. **Reduction Floor**: Combined multipliers minimum 0.50
8. **Compound Gate**: Block if combined < 0.35
9. **Final Sizing**: Apply to account risk → convert to lots using contract specs

**Formula**: `Lots = (Equity × BaseRisk × Multipliers) / (StopLossPoints × TickValue × ContractSize)`

### Portfolio Risk Controls

- **Max Daily VAR**: 2.0% of account equity
- **Max Session Lots**: 0.15 lots total
- **Correlation Kill**: Same TREND_FAMILY + same direction → 0.65×
- **TREND_FAMILY**: {S1_LONDON_BRK, S1F_POST_TK, S4_LONDON_PULL, S5_NY_COMPRESS}

---

## Market Regimes

The system classifies market conditions into 6 states with hysteresis:

**Regime Calculation Overview:**
- **ADX H4**: 14-period ADX on 4-hour timeframe (trend strength)
- **ATR Percentile**: Current H1 ATR vs 90-day historical distribution
- **DXY Correlation**: 20-period correlation with Dollar Index
- **Hysteresis**: Requires 3 consecutive readings to confirm state change

**State Transitions:**
- NO_TRADE: Triggered by extreme ATR (>95th percentile) or system errors
- UNSTABLE: High volatility (85-95% ATR) with low trend strength
- RANGING_CLEAR: Low ADX (<18) regardless of volatility
- WEAK_TRENDING: Moderate ADX (18-26) with any volatility
- NORMAL_TRENDING: Strong ADX (26-35) with normal volatility
- SUPER_TRENDING: Very strong ADX (>35) + high volatility + bearish DXY

| State | ADX H4 | ATR Percentile H1 | DXY Macro | Size Multiplier | Strategies Allowed |
|-------|--------|-------------------|-----------|-----------------|-------------------|
| NO_TRADE | Any | >95% | Any | 0.0× | S7 only (pending) |
| UNSTABLE | Any | 85-95% | Any | 0.4× | S3, S7 |
| RANGING_CLEAR | <18 | Any | Any | 0.7× | S2, S3, S6, S7 |
| WEAK_TRENDING | 18-26 | Any | Any | 0.8× | All strategies |
| NORMAL_TRENDING | 26-35 | Any | No boost | 1.0× | All strategies |
| SUPER_TRENDING | >35 | >55% | DXY < -0.70 | **1.5×** | All strategies |

### Session Definitions (DST-Safe Local Time)

| Session | Local Time Zone | Start | End | Characteristics |
|---------|-----------------|-------|-----|----------------|
| ASIAN | UTC | 22:00 | 07:00 | Lower volatility, tight ranges |
| LONDON | London Time (BST/GMT) | 08:00 | 17:00 | High volatility, trend formation |
| NY | New York Time (EDT/EST) | 08:00 | 17:00 | Second highest liquidity |
| OVERLAP | Both | 13:00-16:00 London | 08:00-11:00 NY | Peak liquidity, strongest moves |

**Note**: Session detection uses pytz for automatic DST adjustments. London and NY sessions follow local business hours, not fixed UTC times.

---

## Installation & Setup

### Prerequisites

- **Python 3.11+**
- **MetaTrader 5** with rpyc bridge (mt5linux)
- **PostgreSQL** (Docker recommended)
- **Linux environment** (Ubuntu 20.04+ recommended)

### Step-by-Step Installation

1. **System Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv postgresql-client docker.io
   ```

2. **Python Environment**
   ```bash
   cd /path/to/xauusd_algo
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Database Setup**
   ```bash
   # Using Docker (recommended)
   docker-compose up -d
   
   # Manual PostgreSQL setup
   # Create database and user as specified in .env
   ```

4. **MT5 Configuration**
   - Install MT5 and configure rpyc bridge
   - Add XAUUSD to Market Watch
   - Ensure trading is enabled
   - Note your MT5 terminal path for rpyc connection

5. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your specific settings:
   # - MT5 connection details
   # - Database credentials
   # - SMTP settings for alerts
   # - API keys for economic calendar
   ```

6. **Database Initialization**
   ```bash
   python db/init_db.py
   ```

7. **System Verification**
   ```bash
   python main.py --checklist
   ```

---

## Configuration

### Core Parameters (config.py)

#### Risk Parameters
```python
BASE_RISK_PHASE_1      = 0.010   # 1.0% per trade
BASE_RISK_PHASE_2      = 0.020   # 2.0% (after 50 proven trades)
V1_LOT_HARD_CAP        = 0.50    # Maximum lot size
MIN_CONDITION_MULTIPLIER = 0.35    # Compound gate threshold
```

#### Kill Switch Thresholds
```python
KS3_DAILY_LOSS_LIMIT_PCT   = -0.070  # -7%
KS4_LOSS_STREAK_COUNT      = 4       # 4 consecutive losses
KS5_WEEKLY_LOSS_LIMIT_PCT  = -0.150  # -15%
KS6_DRAWDOWN_LIMIT_PCT     = 0.20    # 20% drawdown
```

#### Strategy Parameters
```python
MIN_RANGE_SIZE_PTS      = 10     # Minimum viable pre-London range
S6_MIN_RANGE_PTS       = 8.0    # Minimum Asian range
S7_MIN_RANGE_ATR_RATIO = 0.75   # Minimum daily range ratio
PARTIAL_EXIT_R         = 2.0    # Take 50% at 2R
BE_ACTIVATION_R        = 1.5    # BE after 1.5R
```

#### ATR and Volatility
```python
ATR_PERIOD             = 14      # ATR calculation period
ATR_MAMODE             = "RMA"   # Wilder's smoothing
ATR_EWMA_DECAY         = 0.95    # EWMA decay for percentile calc
```

### Environment Variables (.env)

```bash
# Environment
ENV=prod

# MT5 Connection
MT5_HOST=localhost
MT5_PORT=18812

# Database
DATABASE_URL=postgresql://xauusd_user:password@127.0.0.1:5432/xauusd

# SMTP Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
ALERT_RECIPIENT=your-email@gmail.com

# Economic Calendar
HORIZONFX_BASE_URL=https://economic-calendar.horizonfx.id/events
```

---

## Daily Operations

### Session Schedule (IST)

| Time | Event | Strategy |
|------|-------|----------|
| 00:00 | Midnight reset + S7 orders | Daily |
| 05:30 | Asian close + S6 orders | Asian |
| 11:00 | S6 orders expire | Asian |
| 12:25 | S1 pre-London range | London |
| 13:30 | London open | London |
| 17:30 | S5 compression check at noon UTC | S5 |
| 19:00 | NY open | NY |
| 21:30 | S4 hard exit (16:00 UTC) | S4 |
| 22:00 | London time kill | London |
| 03:30 | S5 hard exit (22:00 UTC) | S5 |

### Daily Startup Sequence

1. **Pre-Market Check (12:00 IST)**
   ```bash
   python main.py --checklist
   ```
   - Verify MT5 connection
   - Check database connectivity
   - Validate contract specifications
   - Review recent performance metrics

2. **System Start (Before 12:25 IST)**
   ```bash
   # For live trading
   python main.py --live
   
   # For paper trading (recommended for testing)
   python main.py --paper
   ```

3. **Monitor Key Events**
   - WARM_START completion (regime engine initialization)
   - S6/S7 pending order placement
   - S1 pre-London range calculation
   - Economic calendar updates

### Shutdown Procedure

- **Graceful Shutdown**: Press Ctrl+C
- **Automatic Actions**:
  - Cancels all pending orders
  - Persists critical state
  - Closes database connections
  - Logs shutdown reason

---

## Monitoring & Analytics

### Truth Engine Analytics

The Truth Engine provides comprehensive performance monitoring:

#### Daily Analytics
- Trades per strategy
- Win/loss ratios
- P&L percentages
- Commission tracking
- Spread analysis

#### Weekly Reviews
```bash
python main.py --weekly
```
- Full performance review
- Strategy effectiveness analysis
- Risk metric evaluation
- Action items for optimization

### Conviction Levels

**Conviction Calculation:**
- **Macro Bias**: DXY trend + TLT/TIP yield curve analysis
- **Regime Alignment**: Current regime vs optimal regime for strategy
- **Event Context**: Proximity to high-impact economic events
- **Recent Performance**: Last 10 trades for same strategy

| Level | Criteria | Size Effect |
|-------|----------|-------------|
| STANDARD | Default conditions | 1.0× |
| A_PLUS | Clear horizon + regime alignment | **1.25×** |
| OBSERVATION | Macro misalignment or elevated risk | **0.75×** |

**Activation Gate**: Requires 50+ trades AND >8pp win-rate delta between A+ and OBSERVATION.

### Performance Benchmarks

- **Minimum Win Rate**: 45%
- **Minimum Expectancy**: 0.15R
- **Maximum Drawdown**: 15% (Phase 2 gate)
- **Sharpe Ratio Target**: >1.0 (Phase 3 gate)

### Edge Decay Detection

The system monitors for strategy degradation:

**Performance Metrics Calculation:**
- **Expectancy**: (WinRate × AvgWin) - (LossRate × AvgLoss) in R-multiples
- **Win Rate**: Winning trades / Total trades
- **Minimum Trades**: 50 trades required for reliable statistics
- **Rolling Window**: Last 100 trades for current performance

**Thresholds:**
- **Warning Level**: Expectancy < 0.10R or Win Rate < 40%
- **Critical Level**: Expectancy < 0.05R or Win Rate < 35%
- **Action**: Auto-revert to Phase 1 if critical levels detected

---

## Troubleshooting

### Common Issues

#### No Signals Generated
```
Checklist:
1. Verify regime is not NO_TRADE
2. Check KS7 active status (economic events)
3. Confirm spread is not elevated (KS2)
4. Validate session times
5. Check volume filter (S1 may reject low-volume breakouts)
6. Check ADX trend filter (S6/S7 may filter counter-trend)
7. Check R3 volatility filter (may reject minor event moves)
```

#### Lot Size Too Small
```
1. Check if KS4 countdown is active (halves base risk)
2. Verify regime multiplier (SUPER=1.5×, not 0.5×)
3. Confirm NY session gets 1.0× (not 0.8× penalty)
4. Check conviction level (OBSERVATION reduces size)
```

#### MT5 Connection Issues
```
1. Verify rpyc bridge is running (mt5linux)
2. Check firewall settings for port 18812
3. Ensure MT5 terminal is open and logged in
4. Validate magic number configuration
```

### Diagnostic Commands

```bash
# Pre-session validation
python main.py --checklist

# Weekly performance review
python main.py --weekly

# ATR calibration
python tools/calibrate_atr.py

# Collect historical data
python tools/collect_historical_data.py

# Fetch events history
python tools/fetch_events_history.py
```

### Log Analysis

All logs are structured with KEY=VALUE format for easy parsing:

```bash
# View recent logs
tail -f logs/xauusd.log

# Filter for specific events
grep "KS3_FIRED" logs/xauusd.log

# Analyze trade decisions
grep "S1_ORDER_PLACED" logs/xauusd.log
```

---

## System Components

### Database Schema

#### Core Tables

**system_state.system_state_persistent**
- Stores daily state for warm-start recovery
- Includes all Phase 2 fields (R3/S4/S5 flags)
- DXY variance, spread multiplier

**system_state.trades**
- Complete trade audit trail
- Entry/exit prices, times, lot sizes
- P&L calculation with R-multiple tracking
- Phase 2 audit columns

**market_data.economic_events**
- Event tracking with HorizonFX source flag
- Fallback indicator for hardcoded events

#### Logging Tables
- **system_state.spread_log**: Spread tracking every 5 minutes
- **system_state.regime_log**: Regime changes with context
- **market_data.macro_signals**: Daily bias calculations

### Engine Modules

#### Data Engine
- MT5 OHLCV data fetching
- Economic calendar integration (HorizonFX + fallback)
- Spread tracking and baseline calculation
- DXY correlation analysis
- TLT/TIP macro proxy data

#### Regime Engine
- 6-state market classification
- Hysteresis implementation (3 consecutive readings)
- ATR percentile calculation with EWMA weighting
- ADX trend strength analysis
- Session-aware volatility normalization

#### Signal Engine
- 10 strategy implementations
- Volume and trend filters
- Time-based restrictions and expirations
- Pending order management
- Add-on signal generation

#### Risk Engine
- Position sizing with phase-based progression
- 7 kill switch implementations
- Conviction level calculations
- Portfolio-level risk controls
- Correlation monitoring

#### Execution Engine
- Order placement with spread adjustment
- Fill detection and reconciliation
- Position management (partial exits, BE activation)
- Emergency shutdown procedures
- Ghost position detection

#### Truth Engine
- Performance analytics and reporting
- Edge decay detection
- Conviction level optimization
- Weekly review generation
- Starvation tracking

### Utility Modules

#### Session Management
- DST-safe session detection
- Time zone handling (London, NY, IST)
- Session boundary notifications

#### Logging System
- Structured KEY=VALUE logging
- Rotating file handlers
- Multiple severity levels
- Parseable format for analysis

#### MT5 Client
- Connection management with auto-reconnect
- Error handling and retry logic
- Contract specification fetching
- Position reconciliation

---

## Best Practices

### For New Users

1. **Start with Paper Trading**
   - Use `--paper` flag for at least 2 weeks
   - Monitor all strategy behaviors
   - Understand risk management actions

2. **Monitor Daily**
   - Run `--checklist` before market open
   - Review logs for any warnings or errors
   - Check economic calendar for high-impact events

3. **Understand Your Risk**
   - Start with minimum position sizing
   - Monitor drawdown levels carefully
   - Respect kill switch activations

### For Advanced Users

1. **Customization**
   - Modify strategy parameters in config.py
   - Adjust risk thresholds based on account size
   - Fine-tune regime thresholds for your broker

2. **Optimization**
   - Use weekly reviews to identify weak strategies
   - Monitor conviction level effectiveness
   - Adjust correlation thresholds as needed

3. **Automation**
   - Set up automated alerts for kill switches
   - Implement log monitoring for early issue detection
   - Schedule regular database maintenance

---

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Run `python main.py --weekly`
   - Review performance metrics
   - Check for edge decay warnings

2. **Monthly**
   - Archive old log files
   - Update economic calendar patterns
   - Review and update risk parameters

3. **Quarterly**
   - Full system health check
   - Strategy performance review
   - Parameter re-calibration if needed

### Getting Help

1. **Check Logs First**
   - All issues are logged with structured format
   - Search logs for error codes and warnings
   - Review recent trade decisions

2. **Diagnostic Tools**
   - Use `--checklist` for system validation
   - Run individual tool scripts for specific issues
   - Monitor database connection status

3. **Performance Issues**
   - Check MT5 connection stability
   - Verify database performance
   - Review system resource usage

---

## Disclaimer

This is an advanced algorithmic trading system that involves significant risk. Past performance is not indicative of future results. Always:

- Start with paper trading
- Understand all strategies before risking real capital
- Monitor positions actively
- Respect risk management rules
- Never risk more than you can afford to lose

The system is provided as-is for educational and research purposes. Users are responsible for their own trading decisions and associated risks.

---

*"The biggest edge in this system isn't any single strategy — it's fixing bugs that prevent existing edge from being realized."*