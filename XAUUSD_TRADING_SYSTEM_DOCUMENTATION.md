# XAUUSD Trading System - Complete Documentation
### Version 3.0 | Date: 2026-04-01 | Author: Amit Prasad
### Classification: System Architecture & Operations Manual
### Post-Critical Review Edition — All BUG/LOOP/SIZE/KS/EXP fixes applied

---

## TABLE OF CONTENTS
1. [System Overview](#section-1--system-overview)
2. [Architecture](#section-2--architecture)
3. [Trading Strategies](#section-3--trading-strategies)
4. [Risk Management](#section-4--risk-management)
5. [Market Data & Indicators](#section-5--market-data--indicators)
6. [Database Schema](#section-6--database-schema)
7. [Operational Procedures](#section-7--operational-procedures)
8. [Monitoring & Analytics](#section-8--monitoring--analytics)
9. [Configuration](#section-9--configuration)
10. [Troubleshooting](#section-10--troubleshooting)
11. [Recommendations for Future Improvements](#section-11--recommendations)

---

## SECTION 1 - SYSTEM OVERVIEW

### 1.1 System Identity
- **Name**: Leela XAUUSD Algorithmic Trading System
- **Instrument**: XAUUSD (Spot Gold)
- **Platform**: MetaTrader 5 (MT5) via mt5linux rpyc bridge
- **Magic Number**: 20260320
- **Database**: PostgreSQL (Docker container, TCP on 127.0.0.1:5432)
- **Language**: Python 3.x with APScheduler
- **Trading Style**: Multi-strategy volatility harvesting across all market sessions

### 1.2 Core Philosophy
The system operates on the principle that XAUUSD exhibits predictable volatility patterns across different market sessions and economic events. Rather than predicting direction, the system harvests volatility transitions through multiple complementary strategies with strict risk controls.

### 1.3 Key Features
- **10 Trading Strategies**: S1 family (5), S2, S3, S4, S5, S6, S7, S8, R3
- **Multi-Session Coverage**: Asian, London, New York, and overlap sessions
- **Economic Calendar Integration**: Automated high-impact event handling (HorizonFX + hardcoded fallback)
- **6-State Regime Engine**: Dynamic market state classification with hysteresis
- **Risk-First Approach**: 7 kill switches + portfolio-level risk controls
- **ATR-Based Stops**: All strategies use ATR-scaled stops (v3.0 — replaced fixed-point stops)
- **Volume Filtering**: S1 breakouts require minimum volume confirmation (v3.0)
- **ADX Trend Filtering**: S6/S7 dual orders biased by trend direction (v3.0)
- **Active Conviction Sizing**: A+ conditions boost size after 50+ trades (v3.0)
- **Complete Audit Trail**: Every trade decision logged with full context

---

## SECTION 2 - ARCHITECTURE

### 2.1 System Layers
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
│  Performance analytics, conviction delta, weekly reviews    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Regime States (v3.0 — updated multipliers)
| State | ADX H4 | ATR Percentile H1 | DXY Macro | Size Multiplier | Strategies Allowed |
|-------|--------|-------------------|-----------|-----------------|-------------------|
| NO_TRADE | Any | >95% | Any | 0.0× | S7 only (pending) |
| UNSTABLE | Any | 85-95% | Any | 0.4× | S3, S7 |
| RANGING_CLEAR | <18 | Any | Any | 0.7× | S2, S3, S6, S7 |
| WEAK_TRENDING | 18-26 | Any | Any | 0.8× | All strategies |
| NORMAL_TRENDING | 26-35 | Any | No boost | 1.0× | All strategies |
| SUPER_TRENDING | >35 | >55% | DXY < -0.70 | **1.5×** | All strategies |

**Session Multiplier** (v3.0): London, London-NY Overlap, and NY = 1.0×. Asian/Off-hours = 0.7×.

### 2.3 Session Definitions (UTC)
| Session | Start | End | Characteristics |
|---------|-------|-----|----------------|
| ASIAN | 22:00 | 07:00 | Lower volatility, tight ranges |
| LONDON | 07:00 | 16:00 | High volatility, trend formation |
| NY | 13:00 | 21:00 | Second highest liquidity |
| OVERLAP | 13:00 | 16:00 | Peak liquidity, strongest moves |

---

## SECTION 3 - TRADING STRATEGIES

### 3.1 S1 Family - London Momentum Strategies

#### S1_LONDON_BRK - Primary London Breakout
- **Session**: London (08:00-16:30 London local)
- **Setup**: Pre-London range (00:00-07:55 UTC)
- **Entry**: BUY/SELL STOP at range boundaries + breakout distance (12% of range)
- **Volume Filter** (v3.0): Rejects breakouts with tick_volume < 70% of 5-bar average
- **Minimum Range**: 10 points
- **Stop Loss**: ATR-based — `max(0.3 × H1 ATR, 5.0)` beyond opposite range boundary (v3.0 — was fixed 10% of range)
- **Take Profit**: 2.5R from entry (v3.0 — previously no TP)
- **Expiry**: 16:30 London local time
- **Family**: Trend family (blocks other trend strategies)
- **Max per Day**: 4 (v3.0 — was 3)

#### S1B_FAILED_BRK - Failed Breakout Reversal
- **Trigger**: S1 fills and hits stop loss with specific conditions
- **Entry**: STOP order beyond false breakout extreme
- **Auto-Reset**: Flag clears after 6 M15 candles if no reversal
- **Max per Day**: 1 per S1 campaign

#### S1D_PYRAMID - M5 Pullback Re-entry
- **Trigger**: S1 position open + M5 body close above/below EMA20
- **Entry**: LIMIT at EMA20 with 5-minute expiry
- **Size**: 0.5× base lot
- **Stop**: ATR-based — `max(0.75 × M15 ATR, 15pts)` (v3.0 — was fixed 10-12pts)
- **Max Re-entries**: SUPER=8, NORMAL=5
- **Loss Pause**: After 5 consecutive M5 losses (v3.0 — was 3)

#### S1E_PYRAMID - Trend Continuation Add
- **Trigger**: Partial exit done + BE activated + SUPER/NORMAL regime
- **Entry**: Market order
- **Size**: 0.5× original S1 lots
- **Max per Day**: 1 per campaign

#### S1F_POST_TK - Post Time-Kill Re-entry
- **Session**: NY (after London 16:30 TK)
- **Direction Validation** (v3.0): Checks `last_s1_direction` against H1 EMA20 — rejects if market reversed
- **Entry**: LIMIT at M5 EMA20 with 5-minute expiry
- **Stop**: 15 points
- **Max per Day**: 1

### 3.2 S2_MEAN_REV - Range Reversion (v3.0 — updated)
- **Regime Gate**: RANGING_CLEAR only
- **Signal**: H1 close > **1.5× ATR** from 20 EMA + **RSI confirmation** (v3.0 — was 2.5× ATR, no RSI)
  - SHORT: close > EMA20 + 1.5×ATR AND RSI > 70
  - LONG: close < EMA20 - 1.5×ATR AND RSI < 30
- **ATR Percentile**: 30th-85th (not too quiet, not chaotic)
- **Entry**: LIMIT at EMA20
- **Stop**: 1.5× ATR beyond extreme
- **Regime Exit**: Immediate close if regime transitions to any trending state

### 3.3 S3_STOP_HUNT_REV - Liquidity Sweep Reversal (v3.0 — dynamic range)
- **Sessions**: London + early NY (08:00-16:30 UTC)
- **Range Source**: Rolling 3-hour M15 range (v3.0 — was stale pre-London range)
- **Trigger**: Price sweeps range by >0.3×ATR then M15 close reclaims within 3 bars
- **Entry**: BUY STOP 2pts above reclaim candle high
- **Stop**: sweep_low - 0.5×ATR
- **Max per Session**: 1

### 3.4 S6_ASIAN_BRK - Asian Session Breakout (v3.0 — trend filtered)
- **Session**: Asian (00:00-05:30 UTC setup)
- **Range**: Asian high/low (00:00-05:30 UTC)
- **Minimum Range**: 8 points
- **Entry**: Dual BUY/SELL STOP at range boundaries + 5% of range
- **ADX Trend Filter** (v3.0): In strong trends (ADX>25, DI ratio>1.3×), only trending direction placed
- **Expiry**: 08:00 UTC
- **Stop**: 0.5×ATR from entry
- **Max per Day**: 1

### 3.5 S7_DAILY_STRUCT - Daily Structure Breakout (v3.0 — ATR stop + trend filtered)
- **Setup**: Previous day OHLC (midnight reset)
- **Filter**: Previous day range >0.75×daily ATR
- **Entry**: BUY/SELL STOP at prev day extremes + 5pts
- **Stop**: **0.5 × daily ATR** from entry (v3.0 — was prev_day opposite extreme ± 10pts, 50-90pts wide)
- **ADX Trend Filter** (v3.0): Same as S6 — only trending direction in strong trends
- **Size**: 0.5× base lot
- **Max per Day**: 1

### 3.6 S8_ATR_SPIKE - Flash Spike Continuation (v3.0 — market entry)
- **Trigger**: M15 candle range > 1.5× ATR(14,H1)
- **Confirmation**: Next M15 close past spike midpoint (within 3 bars)
- **Entry**: **Market order at current bid/ask** (v3.0 — was limit at spike midpoint, frequently unfillable)
- **Stop**: 0.5× ATR from entry
- **Size**: 0.5× base lot
- **Max per Day**: 1

### 3.7 Phase 2 Strategies

#### R3_CAL_MOMENTUM - Economic Calendar Momentum (v3.0 — volatility filter)
- **Trigger**: High-impact economic event release
- **Wait**: 5 minutes post-release
- **Volatility Filter** (v3.0): Post-event move must exceed 0.3× H1 ATR — rejects minor releases
- **Entry**: Market order in direction of first M5 close
- **Stop**: 0.5 × H1 ATR
- **TP**: 0.75 × H1 ATR (1.5:1 RR)
- **Hold Limit**: 30 minutes
- **Family**: Independent (coexists with trend positions)

#### S4_LONDON_PULL - London Pullback Continuation
- **Session**: London (07:00-12:00 UTC)
- **Regime Gate**: Trending (ADX > 20 AND increasing)
- **Entry**: LIMIT at M15 EMA20 with 15-minute expiry
- **Stop**: touch_bar extreme - 0.3 × H1 ATR
- **TP**: 1.5 × stop distance
- **Hard Exit**: 16:00 UTC
- **Position Management** (v3.0): Momentum cycling in SUPER/NORMAL, BE activation, partial exit

#### S5_NY_COMPRESS - NY Compression Breakout (v3.0 — STOP order entry)
- **Session**: NY (12:00-15:00 UTC)
- **Trigger**: London range < 0.70 × D1 ATR14 (compressed)
- **Entry**: **BUY/SELL STOP 2pts beyond London boundary** (v3.0 — was M15 close, guaranteed adverse slippage)
- **Stop**: Opposite London extreme + 0.3 × H1 ATR
- **TP**: 1.0 × London range from entry
- **Hard Exit**: 22:00 UTC
- **Position Management** (v3.0): Momentum cycling, BE activation, partial exit

---

## SECTION 4 - RISK MANAGEMENT

### 4.1 Kill Switches (KS1-KS7) — v3.0 updated thresholds

| Switch | Trigger | Action | Recovery |
|--------|---------|--------|-----------|
| KS1 | Stop modification against trade | Reject modification | Manual review |
| KS2 | Spread >2.5× 24h median at placement | Reject order | Wait for spread normalization |
| KS3 | Daily loss > **-4.0%** (v3.0 — was -3%) | Block new entries today | Next day reset |
| KS4 | **6** consecutive losses (v3.0 — was 4) | Reduce size 50% for **3** trades (v3.0 — was 5) | Auto-recovery |
| KS5 | Weekly loss > **-10.0%** (v3.0 — was -8%) | Block entries this week | Next week reset |
| KS6 | Drawdown > **12%** from peak (v3.0 — was 8%) | Emergency shutdown | Manual review |
| KS7 | High-impact event proximity | Block entries 45min pre/20min post | Auto-resume after ATR check |

### 4.2 Position Sizing Algorithm (v3.0 — with reduction floor + conviction)
```
1. Base risk = 1.0% (Phase 1) or 2.0% (Phase 2: 50+ trades, WR>45%, exp>+0.15R)
2. Conviction boost (v3.0): A+ = ×1.25, OBSERVATION = ×0.75 (after 50+ trades with >8pp delta)
3. KS4 countdown: ×0.5 for 3 trades after 6-loss streak
4. Severity multiplier: from economic event risk score
5. Spread multiplier: from current vs median spread ratio
6. Vol scalar: from EWMA ATR percentile
7. REDUCTION FLOOR (v3.0): severity × spread × vol_scalar clamped to minimum 0.50
8. Compound gate: if severity × spread × vol_scalar < 0.35 → block trade entirely
9. Final: max(volume_min, min(calculated_lots, V1_LOT_HARD_CAP))
```

### 4.3 Portfolio Risk Controls (v3.0 — updated)
- **Max Daily VAR**: 2.0% of account equity
- **Max Session Lots**: 0.15 lots total
- **Correlation Kill** (v3.0): Only same TREND_FAMILY + same direction → 0.65× (was: any same direction → 0.5×)
  - TREND_FAMILY = {S1_LONDON_BRK, S1F_POST_TK, S4_LONDON_PULL, S5_NY_COMPRESS}
- **SUPER_TRENDING Scaling** (v3.0): Removed double-halving — regime multiplier (1.5×) handles sizing
- **Correlation Monitoring**: Every 10 closed trades, Pearson correlation of daily P&L

### 4.4 Position Management (v3.0 — all strategies managed)
| Regime | S1 Family | S4/S5 | S6/S7 |
|--------|-----------|-------|-------|
| SUPER/NORMAL | Momentum cycle exit → S1d re-entry | Momentum cycle exit + BE activation | BE activation + partial exit |
| WEAK | Partial exit at **2.0R** (v3.0 — was 1.0R) + BE at **1.5R** (v3.0 — was 0.75R) | Same | Same |
| ATR Trail | **2.5×** M15 ATR (v3.0 — was 1.5×) | Same | Same |

---

## SECTION 5 - MARKET DATA & INDICATORS

### 5.1 Data Sources
- **Primary**: MT5 real-time tick data via rpyc bridge
- **Economic Calendar**: HorizonFX API with hardcoded fallback (NFP, FOMC, CPI, PPI, Retail Sales)
- **Macro Proxy**: TLT/TIP ETF via yfinance (daily 09:00 IST)
- **DXY Correlation**: USDX via MT5 or UUP ETF fallback

### 5.2 Core Indicators

#### ADX (Average Directional Index)
- **Period**: 14, **Timeframe**: H4, **Smoothing**: Wilder's RMA
- **Usage**: Trend strength + direction (DI+/DI- for S6/S7 trend filter)
- **Slope Detection**: Required for S4 (increasing ADX)

#### ATR (Average True Range)
- **Period**: 14, **Smoothing**: Wilder's RMA (pinned to TradingView)
- **H1**: Regime percentile, stop calculations, R3/S8 sizing
- **M15**: Position management trailing, S1d stop calculation
- **D1**: S7 filter, S5 compression check

#### RSI (Relative Strength Index) — v3.0
- **Period**: 14, **Timeframe**: H1
- **Usage**: S2 mean reversion confirmation (>70 SHORT, <30 LONG)

### 5.3 Spread Management
- **Tracking**: Real-time spread logging every 5 minutes
- **Baseline**: 24-hour rolling **median** (v3.0 — SQL syntax fixed)
- **Gate**: KS2 blocks orders at >2.5× median spread
- **Adjustment**: All BUY STOP orders include current spread

---

## SECTION 6 - DATABASE SCHEMA

### 6.1 Core Tables

#### system_state.system_state_persistent
Stores daily state for warm-start recovery. Includes all Phase 2 fields (r3/s4/s5 flags), DXY variance, spread multiplier.

#### system_state.trades
Complete trade audit trail with:
- Entry/exit prices, times, lot sizes
- P&L calculation: `(price_diff / tick_size) × tick_value × lot_size` (v3.0 — BUG-9 fix)
- R-multiple always vs `stop_price_original`
- Phase 2 audit columns (r3_pre_event_price, s4_adx, s5_london_range, severity/spread/compound multipliers)

#### market_data.economic_events
Event tracking with HorizonFX source flag and fallback indicator.

### 6.2 Logging Tables
- **system_state.spread_log**: Spread tracking every 5 minutes
- **system_state.regime_log**: Regime changes with context
- **market_data.macro_signals**: Daily bias calculations

---

## SECTION 7 - OPERATIONAL PROCEDURES

### 7.1 Daily Startup Sequence
1. **Pre-Market Check** (12:00 IST): `python main.py --checklist`
2. **System Start** (Before 12:25 IST): `python main.py --live`
3. **Monitor For**: WARM_START, S6/S7 orders, S1 pending placement

### 7.2 Session Schedule (IST)
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

### 7.3 Shutdown Procedure
- Ctrl+C for graceful shutdown
- Cancels all pending orders, persists critical state

---

## SECTION 8 - MONITORING & ANALYTICS

### 8.1 Truth Engine Analytics
- **Daily**: Trades per strategy, win/loss ratios, P&L%, commission, spread analysis
- **Weekly**: `python main.py --weekly` — full performance review with action items

### 8.2 Conviction Levels (v3.0 — ACTIVE sizing)
| Level | Criteria | Size Effect |
|-------|----------|-------------|
| STANDARD | Default conditions | 1.0× |
| A_PLUS | Clear horizon (90 min no events) + regime alignment | **1.25×** (v3.0 — was observation only) |
| OBSERVATION | Macro misalignment or elevated risk | **0.75×** (v3.0 — was observation only) |

**Activation Gate**: Requires 50+ trades AND >8pp win-rate delta between A+ and OBSERVATION.

### 8.3 Performance Benchmarks
- **Minimum Win Rate**: 45%
- **Minimum Expectancy**: 0.15R
- **Maximum Drawdown**: 15% (Phase 2 gate)
- **Sharpe Ratio Target**: >1.0 (Phase 3 gate)

---

## SECTION 9 - CONFIGURATION

### 9.1 Core Parameters (config.py) — v3.0 values

#### Risk Parameters
```python
BASE_RISK_PHASE_1      = 0.010   # 1.0% per trade
BASE_RISK_PHASE_2      = 0.020   # 2.0% (after 50 proven trades)
V1_LOT_HARD_CAP        = 0.50    # Maximum lot size
MIN_CONDITION_MULTIPLIER = 0.35  # Compound gate threshold
```

#### Kill Switch Thresholds (v3.0)
```python
KS3_DAILY_LOSS_LIMIT_PCT   = -0.040  # -4% (was -3%)
KS4_LOSS_STREAK_COUNT      = 6       # (was 4)
KS4_REDUCED_TRADES         = 3       # (was 5)
KS5_WEEKLY_LOSS_LIMIT_PCT  = -0.100  # -10% (was -8%)
KS6_DRAWDOWN_LIMIT_PCT     = 0.12    # 12% (was 8%)
```

#### Position Management (v3.0)
```python
PARTIAL_EXIT_R         = 2.0   # Take 50% at 2R (was 1R)
BE_ACTIVATION_R        = 1.5   # BE after 1.5R + swing (was 0.75R)
ATR_TRAIL_MULTIPLIER   = 2.5   # 2.5× M15 ATR trail (was 1.5×)
S1D_STOP_POINTS_MIN    = 15    # M5 re-entry min stop (was 10)
S1D_STOP_POINTS_MAX    = 20    # M5 re-entry max stop (was 12)
M5_LOSS_PAUSE_COUNT    = 5     # Pause after 5 M5 losses (was 3)
MAX_S1_FAMILY_ATTEMPTS = 4     # S1+S1b daily limit (was 3)
```

---

## SECTION 10 - TROUBLESHOOTING

### 10.1 Common Issues

#### No Signals Generated
```
Checklist:
1. Verify regime is not NO_TRADE
2. Check KS7 active status (economic events)
3. Confirm spread is not elevated (KS2)
4. Validate session times
5. Check volume filter (S1 — may reject low-volume breakouts)
6. Check ADX trend filter (S6/S7 — may filter counter-trend leg)
7. Check R3 volatility filter (may reject minor event moves)
```

#### Lot Size Too Small
```
v3.0 Fix: SIZE-1 reduction floor ensures severity × spread × vol_scalar
never reduces below 50%. If still seeing min lots:
1. Check if KS4 countdown is active (halves base risk)
2. Verify regime multiplier (SUPER=1.5×, not 0.5×)
3. Confirm NY session gets 1.0× (not 0.8× penalty)
```

### 10.2 Diagnostic Commands
```bash
python main.py --checklist     # Pre-session validation
python main.py --weekly        # Weekly performance review
python tools/calibrate_atr.py  # ATR calibration
```

---

## SECTION 11 - RECOMMENDATIONS FOR DRASTIC IMPROVEMENTS

### HIGH PRIORITY — Expected to significantly improve expectancy

#### REC-1: Multi-Timeframe Confluence Scoring
Instead of binary signal/no-signal, score each setup on a 0-100 scale based on:
- H4 trend alignment (ADX direction + strength)
- H1 structure (higher highs/lows)
- M15 momentum (EMA slope)
- Volume confirmation
- DXY correlation strength

Only take trades scoring >65. This single change could improve win rate by 10-15pp by filtering marginal setups.

#### REC-2: Adaptive ATR Percentile Thresholds
Current ATR thresholds are static (30/55/85/95). XAUUSD volatility regime has shifted significantly since 2024. Implement rolling 90-day recalibration:
```python
# Every Sunday midnight:
atr_distribution = get_90_day_atr_distribution()
ATR_PCT_QUIET_REF = np.percentile(atr_distribution, 30)
ATR_PCT_SUPER_THRESHOLD = np.percentile(atr_distribution, 55)
```
This prevents the system from being permanently stuck in one regime during volatility regime shifts.

#### REC-3: Session-Specific Strategy Performance Tracking
Track win rate and expectancy per strategy PER SESSION. If S1 has 60% WR in London but 30% in London-NY overlap, automatically disable S1 in overlap after 30+ trades confirm the pattern. This is a data-driven strategy filter that improves over time.

#### REC-4: Implement Trailing TP (Partial Close at 2R, Trail Remainder)
Current system takes 50% at 2R and trails the rest. Consider a 3-tier exit:
- 33% at 1.5R (lock in some profit early)
- 33% at 3R (capture the meat of the move)
- 34% trails with 2.5× ATR (ride the tail)

This captures more of the distribution tail on big moves while still banking profits.

#### REC-5: Order Flow / Tick Volume Divergence
Add a pre-entry check: if price is making new highs but tick volume is declining (bearish divergence), skip the LONG entry. This is a powerful filter for false breakouts that volume alone doesn't catch.

### MEDIUM PRIORITY — Structural improvements

#### REC-6: Implement Proper Backtesting Framework
The system has no backtesting capability. Without it, every parameter change is a live experiment. Build a replay engine that:
- Reads historical M5/M15/H1 OHLCV from database
- Simulates all signal engines with historical state
- Produces strategy-level P&L curves
- Enables parameter optimization before live deployment

#### REC-7: Add Intraday Equity Curve Monitoring
Track equity curve slope intraday. If equity drops >1.5% in 2 hours, reduce all new position sizes by 50% for the rest of the session (not a full halt like KS3, just a throttle). This catches "bad days" earlier than the daily limit.

#### REC-8: Implement Strategy Rotation Based on Regime History
If the last 5 regime readings were all RANGING_CLEAR, boost S2/S3 sizing and reduce S1/S4 sizing. If trending for 5+ readings, do the opposite. This dynamically allocates capital to strategies that match current conditions.

#### REC-9: Add Spread Prediction Model
Instead of reacting to current spread, predict spread 15 minutes ahead using:
- Time of day (spread follows predictable intraday patterns)
- Upcoming events (spread widens before NFP)
- Recent spread trajectory

This allows pre-emptive order timing — place orders when spread is predicted to be lowest.

#### REC-10: Implement Monte Carlo Position Sizing
Replace fixed 1%/2% risk with Kelly Criterion-derived sizing:
```python
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
safe_kelly = kelly_fraction * 0.25  # Quarter-Kelly for safety
```
This mathematically optimizes growth rate while controlling drawdown.

---

## APPENDICES

### Appendix A - Change Log
- **v3.0** (2026-04-01): Critical Review fixes applied
  - BUG-7/8/9: Fixed is_dxy_stable import, SQL syntax, P&L formula
  - LOOP-1-9: ATR-based stops, dynamic ranges, market entries, direction validation
  - SIZE-1-5: Reduction floor, SUPER 1.5×, NY penalty removed, correlation kill fixed
  - KS-1-4: Widened all kill switch thresholds
  - EXP-1-10: TP targets, volume filter, partial at 2R, BE at 1.5R, trail at 2.5×, trend filters, conviction sizing
- **v2.0** (2026-03-31): Phase 2 deployment
  - Added R3, S4, S5, S8 strategies
  - Implemented portfolio risk brain
  - Added correlation monitoring
- **v1.0** (2026-03-20): Initial deployment
  - S1 family, S2, S3, S6, S7

### Appendix B - Glossary
- **ATR**: Average True Range — volatility measure
- **ADX**: Average Directional Index — trend strength
- **DI+/DI-**: Directional Indicators — trend direction
- **RSI**: Relative Strength Index — momentum oscillator
- **VAR**: Value at Risk — portfolio risk measure
- **KS**: Kill Switch — risk control mechanism
- **R-Multiple**: Risk-adjusted return measure (profit / initial risk)
- **Kelly Criterion**: Optimal bet sizing formula

---

*"The biggest edge in this system isn't any single strategy — it's fixing the bugs that prevent existing edge from being realized."*

*Documentation Version 3.0 — Post-Critical Review Edition*
