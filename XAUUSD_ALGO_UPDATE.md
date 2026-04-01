

# LEELA v3.1 — Implementation Roadmap
### Prepared for: Cross-Analyst Implementation Reference
### Author: System Architecture Review
### Date: To be set by lead developer

---

## PHASING VALIDATION

Your phasing is correct. One small adjustment:

Move **Risk-of-Ruin Simulator** from Phase 2 into **late Phase 1** — it shares 70% of its infrastructure with the backtesting framework. Building them separately means duplicate work. The Monte Carlo layer is a thin wrapper on top of the replay engine.

Revised structure:

```
PHASE 1A: Quick Wins (Day 1-2)
  → Conviction Decay
  → Starvation Tracking  
  → Edge Decay Detection

PHASE 1B: Foundation (Week 1-3)
  → Backtesting Framework
  → Risk-of-Ruin Simulator (built on top of backtest engine)

PHASE 1C: Backtest-Validated Deployments (Week 3-6)
  → Adaptive ATR Thresholds
  → Micro State Engine
  → Intraday Equity Throttle
  → 3-Tier Exit

PHASE 2: Architectural Evolution (Week 7+)
  → Opportunity Budget Engine (after multiplier audit)
```

---

## PHASE 1A — QUICK WINS

---

### 1A-1: Conviction Decay Weighting (EWMA)

**What changes**: The conviction system currently computes A_PLUS and OBSERVATION win rates using equal-weighted last 50 trades. This replaces equal weighting with exponential decay so recent trades influence conviction more than older trades.

**Where it lives**: Inside your conviction calculation function — wherever you currently compute `a_plus_wr` and `observation_wr`.

**Current logic (what to find in your code)**:
```python
# You likely have something like:
a_plus_trades = get_trades(conviction='A_PLUS', last_n=50)
a_plus_wr = sum(1 for t in a_plus_trades if t.is_win) / len(a_plus_trades)

observation_trades = get_trades(conviction='OBSERVATION', last_n=50)  
observation_wr = sum(1 for t in observation_trades if t.is_win) / len(observation_trades)

delta = a_plus_wr - observation_wr
if delta > 0.08:  # 8pp
    conviction_sizing_active = True
```

**Replace with**:
```python
def ewma_win_rate(trades, alpha=0.05):
    """
    Exponentially weighted win rate.
    alpha=0.05 means trade from 20 trades ago has ~36% of 
    the weight of the most recent trade.
    Half-life ≈ 14 trades at alpha=0.05.
    
    Higher alpha = more reactive to recent results (noisier)
    Lower alpha = more stable (slower to detect shifts)
    
    alpha=0.05 is conservative — appropriate for 50-trade windows.
    """
    if len(trades) < 10:
        return None  # insufficient data
    
    # Order: oldest first, newest last
    trades_ordered = sorted(trades, key=lambda t: t.close_time)
    
    n = len(trades_ordered)
    weights = []
    for i in range(n):
        # i=0 is oldest, i=n-1 is newest
        weight = (1 - alpha) ** (n - 1 - i)
        weights.append(weight)
    
    total_weight = sum(weights)
    weighted_wins = sum(
        w * (1.0 if t.is_win else 0.0) 
        for w, t in zip(weights, trades_ordered)
    )
    
    return weighted_wins / total_weight


def compute_conviction_delta():
    a_plus_trades = get_trades(conviction='A_PLUS', last_n=50)
    obs_trades = get_trades(conviction='OBSERVATION', last_n=50)
    
    a_plus_wr = ewma_win_rate(a_plus_trades)
    obs_wr = ewma_win_rate(obs_trades)
    
    if a_plus_wr is None or obs_wr is None:
        return None, False  # not enough data
    
    delta = a_plus_wr - obs_wr
    active = delta > 0.08  # 8pp threshold unchanged
    
    return delta, active
```

**What NOT to change**: The activation gate (50+ trades, >8pp delta), the sizing multipliers (A+ = 1.25×, OBS = 0.75×), the conviction classification logic itself. Only the win rate calculation method changes.

**Validation before deploying**: Run both old and new calculation on your existing trade history. Print both values side by side for 5-10 cycles. They should be similar but the EWMA version should react faster to recent streaks.

```python
# One-time validation script
trades = get_all_trades(conviction='A_PLUS')
old_wr = simple_win_rate(trades[-50:])
new_wr = ewma_win_rate(trades[-50:])
print(f"Equal weight WR: {old_wr:.3f}")
print(f"EWMA WR:         {new_wr:.3f}")
print(f"Difference:      {new_wr - old_wr:+.3f}")
# If difference is >5pp, investigate which recent trades are driving it
```

**Database changes**: None.

**Risk**: Near zero. If EWMA produces unexpected values, the activation gate (50+ trades, >8pp) prevents it from affecting sizing.

---

### 1A-2: Silent Trade Starvation Tracking

**What changes**: Adds automated counters at every point in the signal pipeline where a trade can be blocked. Surfaces warnings when the system is generating signals but executing none.

**Where it lives**: Inside your main signal evaluation loop — the function that runs every cycle and checks each strategy.

**Architecture**:

```python
# New class — add to your system state module
class StarvationTracker:
    """
    Tracks signal flow through the gate stack.
    Resets every session. Persists daily totals to DB.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.counters = {
            'signals_evaluated':     0,  # strategy checked conditions
            'signals_generated':     0,  # conditions met, signal created
            'blocked_regime':        0,  # killed by regime gate
            'blocked_ks':            0,  # killed by any kill switch (log which)
            'blocked_spread':        0,  # KS2 spread gate
            'blocked_volume':        0,  # S1 volume filter
            'blocked_adx_filter':    0,  # S6/S7 ADX trend filter
            'blocked_compound_gate': 0,  # severity×spread×vol < 0.35
            'blocked_family':        0,  # family correlation kill
            'blocked_portfolio':     0,  # session lot cap / VAR limit
            'blocked_event':         0,  # KS7 event proximity
            'blocked_other':         0,  # any other gate
            'orders_placed':         0,  # made it through all gates
            'orders_filled':         0,  # actually executed
        }
        self.blocked_details = []  # list of (timestamp, strategy, gate, context)
    
    def record_block(self, strategy, gate, context=""):
        self.counters[f'blocked_{gate}'] += 1
        self.blocked_details.append({
            'time': datetime.utcnow(),
            'strategy': strategy,
            'gate': gate,
            'context': context
        })
    
    def record_signal(self):
        self.counters['signals_generated'] += 1
    
    def record_order(self):
        self.counters['orders_placed'] += 1
    
    def record_fill(self):
        self.counters['orders_filled'] += 1
    
    def check_starvation(self, session_name):
        """Call at end of each session or every 2 hours."""
        gen = self.counters['signals_generated']
        exe = self.counters['orders_placed']
        
        alerts = []
        
        # Case 1: signals generated but none executed
        if gen >= 3 and exe == 0:
            # Find which gate is killing everything
            top_blocker = max(
                [(k, v) for k, v in self.counters.items() 
                 if k.startswith('blocked_') and v > 0],
                key=lambda x: x[1],
                default=('none', 0)
            )
            alerts.append(
                f"STARVATION [{session_name}]: {gen} signals generated, "
                f"0 executed. Top blocker: {top_blocker[0]} ({top_blocker[1]}×)"
            )
        
        # Case 2: no signals generated during active session
        if self.counters['signals_evaluated'] > 0 and gen == 0:
            alerts.append(
                f"SILENT [{session_name}]: {self.counters['signals_evaluated']} "
                f"strategies evaluated, 0 signals generated. "
                f"Check if conditions are genuinely absent or data feed issue."
            )
        
        # Case 3: high block rate (>80% of signals blocked)
        if gen > 5 and exe < gen * 0.2:
            block_rate = 1 - (exe / gen)
            alerts.append(
                f"HIGH BLOCK RATE [{session_name}]: {block_rate:.0%} of "
                f"signals blocked. Gates may be over-constrained."
            )
        
        return alerts
    
    def daily_summary(self):
        """Call at end of trading day. Persist to DB."""
        summary = {
            'date': date.today(),
            'total_evaluated': self.counters['signals_evaluated'],
            'total_generated': self.counters['signals_generated'],
            'total_executed': self.counters['orders_placed'],
            'total_filled': self.counters['orders_filled'],
            'pass_through_rate': (
                self.counters['orders_placed'] / self.counters['signals_generated']
                if self.counters['signals_generated'] > 0 else 0
            ),
            'top_blockers': sorted(
                [(k, v) for k, v in self.counters.items() 
                 if k.startswith('blocked_')],
                key=lambda x: x[1], reverse=True
            )[:3],
            'blocked_details': self.blocked_details
        }
        return summary
```

**Integration points** — you need to add counter calls at each gate in your existing code:

```python
# In your signal evaluation loop (pseudocode of YOUR existing flow)
def evaluate_strategies():
    for strategy in active_strategies:
        tracker.counters['signals_evaluated'] += 1
        
        # Gate 1: Regime
        if not regime_allows(strategy):
            tracker.record_block(strategy, 'regime', 
                                f'current={current_regime}')
            continue
        
        # Gate 2: Kill switches
        ks_result = check_kill_switches(strategy)
        if ks_result:
            tracker.record_block(strategy, 'ks', 
                                f'triggered={ks_result}')
            continue
        
        # Gate 3: Strategy-specific conditions
        signal = strategy.evaluate()
        if signal is None:
            # No signal — conditions not met (this is normal, don't count)
            continue
        
        tracker.record_signal()  # signal generated
        
        # Gate 4: Volume filter (S1)
        if strategy == 'S1' and not volume_check(signal):
            tracker.record_block(strategy, 'volume',
                                f'vol={current_vol}, threshold={threshold}')
            continue
        
        # Gate 5: Compound gate
        compound = severity * spread_mult * vol_scalar
        if compound < 0.35:
            tracker.record_block(strategy, 'compound_gate',
                                f'compound={compound:.2f}')
            continue
        
        # Gate 6: Family correlation
        if family_blocked(strategy, signal.direction):
            tracker.record_block(strategy, 'family',
                                f'dir={signal.direction}')
            continue
        
        # Gate 7: Portfolio limits
        if portfolio_full(strategy):
            tracker.record_block(strategy, 'portfolio',
                                f'session_lots={current_session_lots}')
            continue
        
        # All gates passed
        place_order(strategy, signal)
        tracker.record_order()


# Call at session boundaries (add to your scheduler)
# End of London session:
alerts = tracker.check_starvation('LONDON')
for alert in alerts:
    logger.warning(alert)

# End of trading day:
summary = tracker.daily_summary()
persist_starvation_summary(summary)  # write to DB
tracker.reset()
```

**Database addition** — one new table:

```sql
CREATE TABLE system_state.starvation_log (
    id SERIAL PRIMARY KEY,
    log_date DATE NOT NULL,
    session VARCHAR(20),
    signals_evaluated INT,
    signals_generated INT,
    orders_placed INT,
    orders_filled INT,
    pass_through_rate DECIMAL(5,4),
    top_blocker_1 VARCHAR(50),
    top_blocker_1_count INT,
    top_blocker_2 VARCHAR(50),
    top_blocker_2_count INT,
    top_blocker_3 VARCHAR(50),
    top_blocker_3_count INT,
    details JSONB,  -- full blocked_details list
    created_at TIMESTAMP DEFAULT NOW()
);
```

**What to watch for after deployment**: If pass_through_rate drops below 15% consistently across multiple sessions, your gates are over-constrained. Look at the top blockers to identify which gate to investigate.

---

### 1A-3: Edge Decay Detection

**What changes**: Adds a rolling expectancy and win rate monitor to your Truth Engine weekly output. Alerts when edge is declining toward minimum thresholds. Auto-reverts to Phase 1 risk if edge drops critically.

**Where it lives**: Inside your Truth Engine / weekly review function.

```python
class EdgeDecayMonitor:
    """
    Monitors rolling expectancy and win rate for slow edge erosion.
    
    WHY THIS EXISTS:
    Kill switches catch acute drawdowns (-4% daily, -10% weekly, 12% DD).
    Nothing catches slow edge decay from +0.25R to +0.05R over 3 months.
    This fills that gap.
    """
    
    # Thresholds (add to config.py)
    EDGE_WARNING_EXPECTANCY = 0.10   # below system minimum of 0.15R
    EDGE_CRITICAL_EXPECTANCY = 0.05  # near zero edge
    EDGE_WARNING_WR = 0.40           # below system minimum of 45%
    EDGE_CRITICAL_WR = 0.35          # severely degraded
    MIN_TRADES_FOR_DETECTION = 30    # need enough data
    
    def check_edge_health(self):
        """
        Call weekly (in Truth Engine) and daily (lightweight check).
        Returns: status, alerts, recommended_actions
        """
        # Pull last 100 closed trades (all strategies combined)
        recent_trades = get_closed_trades(last_n=100)
        
        if len(recent_trades) < self.MIN_TRADES_FOR_DETECTION:
            return 'INSUFFICIENT_DATA', [], []
        
        # Compute rolling metrics
        wins = [t for t in recent_trades if t.r_multiple > 0]
        losses = [t for t in recent_trades if t.r_multiple <= 0]
        
        win_rate = len(wins) / len(recent_trades)
        avg_win_r = (
            sum(t.r_multiple for t in wins) / len(wins) 
            if wins else 0
        )
        avg_loss_r = (
            abs(sum(t.r_multiple for t in losses)) / len(losses) 
            if losses else 0
        )
        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
        
        # Also compute for last 30 trades (faster signal)
        recent_30 = recent_trades[-30:]
        wins_30 = [t for t in recent_30 if t.r_multiple > 0]
        losses_30 = [t for t in recent_30 if t.r_multiple <= 0]
        wr_30 = len(wins_30) / len(recent_30) if recent_30 else 0
        
        alerts = []
        actions = []
        status = 'HEALTHY'
        
        # Check expectancy
        if expectancy < self.EDGE_CRITICAL_EXPECTANCY:
            status = 'CRITICAL'
            alerts.append(
                f"EDGE CRITICAL: Rolling 100-trade expectancy = "
                f"{expectancy:+.3f}R (threshold: {self.EDGE_CRITICAL_EXPECTANCY}R)"
            )
            actions.append('AUTO_REVERT_PHASE_1')
            actions.append('MANUAL_REVIEW_REQUIRED')
            
        elif expectancy < self.EDGE_WARNING_EXPECTANCY:
            status = 'WARNING'
            alerts.append(
                f"EDGE WARNING: Rolling 100-trade expectancy = "
                f"{expectancy:+.3f}R (approaching minimum {self.EDGE_WARNING_EXPECTANCY}R)"
            )
            actions.append('REDUCE_TO_PHASE_1_RISK')
        
        # Check win rate
        if win_rate < self.EDGE_CRITICAL_WR:
            status = 'CRITICAL'
            alerts.append(
                f"WIN RATE CRITICAL: {win_rate:.1%} "
                f"(threshold: {self.EDGE_CRITICAL_WR:.0%})"
            )
        elif win_rate < self.EDGE_WARNING_WR:
            if status != 'CRITICAL':
                status = 'WARNING'
            alerts.append(
                f"WIN RATE WARNING: {win_rate:.1%} "
                f"(approaching minimum {self.EDGE_WARNING_WR:.0%})"
            )
        
        # Trend detection: compare last 30 vs last 100
        if len(recent_30) >= 20:
            wr_100 = win_rate
            if wr_30 < wr_100 - 0.10:  # 10pp decline in recent trades
                alerts.append(
                    f"TREND WARNING: Recent 30-trade WR ({wr_30:.1%}) is "
                    f"{(wr_100 - wr_30):.0%} below 100-trade WR ({wr_100:.1%}). "
                    f"Edge may be actively deteriorating."
                )
        
        # Execute actions
        if 'AUTO_REVERT_PHASE_1' in actions:
            self.revert_to_phase_1()
        
        return status, alerts, actions
    
    def revert_to_phase_1(self):
        """
        Automatically reduce risk to Phase 1 levels.
        Does NOT shut down — just reduces exposure while 
        you investigate.
        """
        # Update config
        set_config('CURRENT_BASE_RISK', 0.010)  # Force 1% regardless of phase
        
        logger.critical(
            "EDGE DECAY AUTO-REVERT: Risk reduced to Phase 1 (1%). "
            "Manual review required before restoring Phase 2."
        )
        
        # Persist the event
        log_to_db('edge_decay_revert', {
            'timestamp': datetime.utcnow(),
            'action': 'auto_revert_phase_1',
            'requires_manual_review': True
        })


# Integration into Truth Engine
def weekly_review():
    # ... existing weekly review code ...
    
    # Add edge decay check
    monitor = EdgeDecayMonitor()
    status, alerts, actions = monitor.check_edge_health()
    
    print(f"\n{'='*50}")
    print(f"EDGE HEALTH: {status}")
    for alert in alerts:
        print(f"  ⚠ {alert}")
    for action in actions:
        print(f"  → ACTION: {action}")
    print(f"{'='*50}\n")


# Also run a lightweight daily check
def daily_edge_check():
    """Add to your daily scheduler, runs once at end of trading day."""
    monitor = EdgeDecayMonitor()
    status, alerts, _ = monitor.check_edge_health()
    if status != 'HEALTHY':
        for alert in alerts:
            logger.warning(alert)
```

**Database addition**: One entry in your existing regime_log or a simple new table:

```sql
CREATE TABLE system_state.edge_health_log (
    id SERIAL PRIMARY KEY,
    check_date DATE NOT NULL,
    check_type VARCHAR(10),  -- 'DAILY' or 'WEEKLY'
    status VARCHAR(20),      -- 'HEALTHY', 'WARNING', 'CRITICAL'
    expectancy_100 DECIMAL(6,4),
    win_rate_100 DECIMAL(5,4),
    expectancy_30 DECIMAL(6,4),
    win_rate_30 DECIMAL(5,4),
    alerts JSONB,
    actions_taken JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## PHASE 1B — FOUNDATION

---

### 1B-1: Backtesting Framework

**What it is**: A replay engine that simulates your entire trading system on historical data. Reads stored OHLCV, computes indicators, runs regime engine, generates signals, simulates execution, and produces per-strategy P&L curves.

**Architecture**:

```
┌─────────────────────────────────────┐
│         HISTORICAL DATA STORE       │
│  M1/M5/M15/H1/H4/D1 OHLCV         │
│  Historical spreads (spread_log)    │
│  Historical events (economic_events)│
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         DATA REPLAY ENGINE          │
│  Feeds candles chronologically      │
│  Simulates tick-level within bars   │
│  Provides spread at each timestamp  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│      SYSTEM SIMULATION ENGINE       │
│  Exact same logic as live system:   │
│  - Indicator computation            │
│  - Regime classification            │
│  - Kill switch evaluation           │
│  - Signal generation                │
│  - Risk/sizing engine               │
│  - Portfolio brain                  │
│  - Position management              │
│  But with simulated execution       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│       EXECUTION SIMULATOR           │
│  - STOP orders: fill at price +     │
│    simulated slippage (0.5-1pt)     │
│  - LIMIT orders: fill only if       │
│    price crosses level              │
│  - Market orders: fill at close +   │
│    half spread + slippage           │
│  - Spread from historical data      │
│  - Order expiry simulation          │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│       RESULTS ENGINE                │
│  Per-strategy P&L curves            │
│  Aggregate equity curve             │
│  Walk-forward validation            │
│  Parameter sensitivity analysis     │
└─────────────────────────────────────┘
```

**Key design principle**: The simulation engine must use the EXACT SAME code as your live system wherever possible. Don't rewrite strategy logic for backtesting — import it.

```python
# Project structure
backtest/
├── engine.py              # Main replay loop
├── data_feed.py           # Historical data provider
├── execution_simulator.py # Order fill simulation
├── results.py             # Analytics and reporting
├── walk_forward.py        # Walk-forward validation
└── monte_carlo.py         # Risk-of-ruin (Phase 1B-2)

# engine.py references your LIVE code:
from strategies.s1_london_brk import S1LondonBreakout
from strategies.s2_mean_rev import S2MeanReversion
# ... etc
from risk.sizing import compute_position_size
from risk.kill_switches import evaluate_kill_switches
from regime.engine import classify_regime
```

**Implementation — Core Replay Engine**:

```python
class BacktestEngine:
    def __init__(self, config):
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.initial_balance = config.get('initial_balance', 10000)
        self.slippage_points = config.get('slippage_points', 0.7)
        
        # Data feeds
        self.data_feed = HistoricalDataFeed(
            self.start_date, self.end_date
        )
        self.spread_feed = HistoricalSpreadFeed(
            self.start_date, self.end_date
        )
        self.event_feed = HistoricalEventFeed(
            self.start_date, self.end_date
        )
        
        # System state — mirrors live system
        self.state = SimulatedSystemState(self.initial_balance)
        self.execution_sim = ExecutionSimulator(self.slippage_points)
        
        # Results collection
        self.trade_log = []
        self.equity_curve = []
        self.strategy_pnl = {s: [] for s in ALL_STRATEGIES}
        self.starvation_tracker = StarvationTracker()  # reuse live code
    
    def run(self):
        """
        Main replay loop.
        Iterates through time at M5 resolution (primary tick).
        Computes higher timeframe indicators from accumulated bars.
        """
        bar_buffer = BarBuffer()  # accumulates M5 into M15, H1, H4, D1
        
        for m5_bar in self.data_feed.iter_m5_bars():
            timestamp = m5_bar.time
            
            # 1. Update bar buffer (builds higher TF candles)
            bar_buffer.add_m5(m5_bar)
            
            # 2. Get current spread from historical data
            current_spread = self.spread_feed.get_spread_at(timestamp)
            
            # 3. Get upcoming events
            upcoming_events = self.event_feed.get_events_near(
                timestamp, window_minutes=60
            )
            
            # 4. Check if higher TF bars completed
            new_m15 = bar_buffer.pop_completed('M15')
            new_h1 = bar_buffer.pop_completed('H1')
            new_h4 = bar_buffer.pop_completed('H4')
            new_d1 = bar_buffer.pop_completed('D1')
            
            # 5. Update indicators (only when new bars complete)
            if new_h4:
                self.state.update_adx_h4(bar_buffer.get_series('H4'))
            if new_h1:
                self.state.update_atr_h1(bar_buffer.get_series('H1'))
                self.state.update_rsi_h1(bar_buffer.get_series('H1'))
                self.state.update_ema20_h1(bar_buffer.get_series('H1'))
            if new_m15:
                self.state.update_atr_m15(bar_buffer.get_series('M15'))
                self.state.update_ema20_m15(bar_buffer.get_series('M15'))
            
            # 6. Classify regime (same logic as live)
            self.state.regime = classify_regime(
                adx_h4=self.state.adx_h4,
                atr_pct_h1=self.state.atr_percentile_h1,
                dxy_corr=self.state.dxy_correlation
            )
            
            # 7. Process pending orders (check fills)
            self.execution_sim.process_pending_orders(
                self.state.pending_orders,
                m5_bar,  # use bar OHLC for fill simulation
                current_spread,
                self.state
            )
            
            # 8. Manage open positions (BE, partials, trails)
            self.manage_positions(m5_bar, timestamp)
            
            # 9. Check session boundaries (order expiry, time kills)
            self.check_session_events(timestamp)
            
            # 10. Evaluate strategies (same logic as live)
            session = get_session(timestamp)
            for strategy in get_strategies_for_session(session):
                self.starvation_tracker.counters['signals_evaluated'] += 1
                
                # Regime gate
                if not regime_allows(strategy, self.state.regime):
                    self.starvation_tracker.record_block(
                        strategy, 'regime'
                    )
                    continue
                
                # Kill switches
                ks = evaluate_kill_switches(self.state, current_spread)
                if ks:
                    self.starvation_tracker.record_block(strategy, 'ks')
                    continue
                
                # Generate signal (IMPORT FROM LIVE CODE)
                signal = strategy.evaluate(
                    self.state, bar_buffer, current_spread
                )
                if signal is None:
                    continue
                
                self.starvation_tracker.record_signal()
                
                # Risk sizing (IMPORT FROM LIVE CODE)
                lots = compute_position_size(
                    signal, self.state, current_spread
                )
                if lots is None:
                    self.starvation_tracker.record_block(
                        strategy, 'compound_gate'
                    )
                    continue
                
                # Place simulated order
                order = self.execution_sim.place_order(
                    signal, lots, current_spread
                )
                self.state.pending_orders.append(order)
                self.starvation_tracker.record_order()
            
            # 11. Record equity
            equity = self.state.compute_equity(m5_bar.close)
            self.equity_curve.append((timestamp, equity))
        
        return self.compile_results()
    
    def manage_positions(self, bar, timestamp):
        """
        Mirrors live position management:
        - Check stop loss hit
        - Check take profit hit  
        - Partial exit at threshold
        - BE activation
        - ATR trailing stop update
        """
        for pos in list(self.state.open_positions):
            # Stop loss check
            if pos.direction == 'BUY':
                if bar.low <= pos.current_sl:
                    self.close_position(
                        pos, pos.current_sl, 'STOP_LOSS', timestamp
                    )
                    continue
                if pos.tp and bar.high >= pos.tp:
                    self.close_position(
                        pos, pos.tp, 'TAKE_PROFIT', timestamp
                    )
                    continue
            else:  # SELL
                if bar.high >= pos.current_sl:
                    self.close_position(
                        pos, pos.current_sl, 'STOP_LOSS', timestamp
                    )
                    continue
                if pos.tp and bar.low <= pos.tp:
                    self.close_position(
                        pos, pos.tp, 'TAKE_PROFIT', timestamp
                    )
                    continue
            
            # R-multiple calculation
            current_r = compute_r_multiple(
                pos, bar.close, pos.stop_price_original
            )
            
            # Partial exit
            if (not pos.partial_done and 
                current_r >= PARTIAL_EXIT_R):
                self.partial_close(pos, bar.close, timestamp)
            
            # BE activation
            if (not pos.be_activated and 
                current_r >= BE_ACTIVATION_R):
                self.activate_breakeven(pos, bar)
            
            # ATR trail update
            if pos.be_activated:
                new_trail = compute_atr_trail(
                    pos, self.state.atr_m15, ATR_TRAIL_MULTIPLIER
                )
                if pos.direction == 'BUY':
                    pos.current_sl = max(pos.current_sl, new_trail)
                else:
                    pos.current_sl = min(pos.current_sl, new_trail)
    
    def close_position(self, pos, price, reason, timestamp):
        """Record closed trade with full audit."""
        pnl = compute_pnl(pos, price)  # same formula as live
        r_mult = compute_r_multiple(pos, price, pos.stop_price_original)
        
        trade_record = {
            'strategy': pos.strategy,
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': price,
            'entry_time': pos.entry_time,
            'exit_time': timestamp,
            'lots': pos.lots,
            'pnl': pnl,
            'r_multiple': r_mult,
            'exit_reason': reason,
            'regime_at_entry': pos.regime_at_entry,
            'regime_at_exit': self.state.regime,
        }
        
        self.trade_log.append(trade_record)
        self.strategy_pnl[pos.strategy].append(trade_record)
        self.state.update_balance(pnl)
        self.state.open_positions.remove(pos)
```

**Execution Simulator** — this is where realism matters:

```python
class ExecutionSimulator:
    def __init__(self, slippage_points=0.7):
        self.slippage = slippage_points
    
    def process_pending_orders(self, orders, bar, spread, state):
        """
        Check if pending orders would have filled during this bar.
        
        Rules:
        - BUY STOP: fills if bar.high >= order.price
          Fill price = order.price + slippage (adverse)
        - SELL STOP: fills if bar.low <= order.price
          Fill price = order.price - slippage (adverse)
        - BUY LIMIT: fills if bar.low <= order.price
          Fill price = order.price (or better)
        - SELL LIMIT: fills if bar.high >= order.price
          Fill price = order.price (or better)
        - MARKET: fills at bar.open + half_spread + slippage (BUY)
                   or bar.open - half_spread - slippage (SELL)
        """
        for order in list(orders):
            # Check expiry first
            if order.expiry and bar.time >= order.expiry:
                orders.remove(order)
                continue
            
            filled = False
            fill_price = None
            
            if order.type == 'BUY_STOP':
                if bar.high >= order.price:
                    fill_price = order.price + self.slippage
                    # Spread already included in order.price 
                    # per your v3.0 logic
                    filled = True
                    
            elif order.type == 'SELL_STOP':
                if bar.low <= order.price:
                    fill_price = order.price - self.slippage
                    filled = True
                    
            elif order.type == 'BUY_LIMIT':
                if bar.low <= order.price:
                    fill_price = order.price  # limit = exact or better
                    filled = True
                    
            elif order.type == 'SELL_LIMIT':
                if bar.high >= order.price:
                    fill_price = order.price
                    filled = True
                    
            elif order.type == 'MARKET':
                half_spread = spread / 2
                if order.direction == 'BUY':
                    fill_price = bar.open + half_spread + self.slippage
                else:
                    fill_price = bar.open - half_spread - self.slippage
                filled = True
            
            if filled:
                position = Position(
                    strategy=order.strategy,
                    direction=order.direction,
                    entry_price=fill_price,
                    entry_time=bar.time,
                    lots=order.lots,
                    stop_price_original=order.sl,
                    current_sl=order.sl,
                    tp=order.tp,
                    regime_at_entry=state.regime
                )
                state.open_positions.append(position)
                orders.remove(order)
                state.starvation_tracker.record_fill()
```

**Results Compilation**:

```python
class BacktestResults:
    def __init__(self, trade_log, equity_curve, strategy_pnl, config):
        self.trades = trade_log
        self.equity = equity_curve
        self.by_strategy = strategy_pnl
        self.config = config
    
    def summary(self):
        """Print comprehensive results."""
        print("=" * 70)
        print("BACKTEST RESULTS")
        print(f"Period: {self.config['start_date']} → {self.config['end_date']}")
        print(f"Initial Balance: ${self.config['initial_balance']:,.2f}")
        print(f"Final Balance:   ${self.equity[-1][1]:,.2f}")
        print(f"Total Return:    {self.total_return():.1%}")
        print(f"Total Trades:    {len(self.trades)}")
        print("=" * 70)
        
        # Per-strategy breakdown
        print(f"\n{'Strategy':<20} {'Trades':>6} {'WR':>7} "
              f"{'Exp(R)':>8} {'PF':>6} {'MaxDD':>8}")
        print("-" * 60)
        
        for strategy in ALL_STRATEGIES:
            trades = self.by_strategy.get(strategy, [])
            if not trades:
                continue
            
            wins = [t for t in trades if t['r_multiple'] > 0]
            losses = [t for t in trades if t['r_multiple'] <= 0]
            
            wr = len(wins) / len(trades) if trades else 0
            
            avg_win = (
                sum(t['r_multiple'] for t in wins) / len(wins) 
                if wins else 0
            )
            avg_loss = (
                abs(sum(t['r_multiple'] for t in losses)) / len(losses)
                if losses else 0
            )
            
            exp = (wr * avg_win) - ((1 - wr) * avg_loss)
            pf = (
                sum(t['pnl'] for t in wins) / 
                abs(sum(t['pnl'] for t in losses))
                if losses and sum(t['pnl'] for t in losses) != 0 
                else float('inf')
            )
            
            max_dd = self.compute_strategy_max_dd(trades)
            
            print(f"{strategy:<20} {len(trades):>6} {wr:>6.1%} "
                  f"{exp:>+7.3f} {pf:>6.2f} {max_dd:>7.1%}")
        
        # Correlation matrix
        print("\n\nSTRATEGY CORRELATION MATRIX (daily P&L)")
        self.print_correlation_matrix()
    
    def walk_forward_report(self, train_months=6, test_months=3):
        """
        Walk-forward validation.
        Splits data into rolling train/test windows.
        Reports out-of-sample performance separately.
        """
        windows = self.generate_wf_windows(train_months, test_months)
        
        print(f"\n{'='*70}")
        print("WALK-FORWARD VALIDATION")
        print(f"Train window: {train_months} months, "
              f"Test window: {test_months} months")
        print(f"{'='*70}")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_trades = [
                t for t in self.trades 
                if train_start <= t['entry_time'] < train_end
            ]
            test_trades = [
                t for t in self.trades 
                if test_start <= t['entry_time'] < test_end
            ]
            
            train_exp = compute_expectancy(train_trades)
            test_exp = compute_expectancy(test_trades)
            
            degradation = (
                (test_exp - train_exp) / train_exp * 100 
                if train_exp != 0 else 0
            )
            
            status = "✅" if test_exp > 0.10 else "⚠️" if test_exp > 0 else "❌"
            
            print(
                f"Window {i+1}: "
                f"Train {train_start:%Y-%m} → {train_end:%Y-%m} "
                f"(exp={train_exp:+.3f}R) | "
                f"Test {test_start:%Y-%m} → {test_end:%Y-%m} "
                f"(exp={test_exp:+.3f}R) | "
                f"Degradation: {degradation:+.0f}% {status}"
            )
```

**Data requirements — what you need to collect before running**:

```python
# Check data availability
def validate_data_for_backtest(start_date, end_date):
    """Run this before starting a backtest."""
    issues = []
    
    # M5 OHLCV — primary timeframe
    m5_count = count_bars('M5', start_date, end_date)
    expected_m5 = expected_bars('M5', start_date, end_date)
    if m5_count < expected_m5 * 0.95:
        issues.append(
            f"M5 data: {m5_count}/{expected_m5} bars "
            f"({m5_count/expected_m5:.0%} coverage)"
        )
    
    # Spread data
    spread_count = count_spread_entries(start_date, end_date)
    expected_spreads = (end_date - start_date).days * 288  # every 5 min
    if spread_count < expected_spreads * 0.80:
        issues.append(
            f"Spread data: {spread_count}/{expected_spreads} entries "
            f"({spread_count/expected_spreads:.0%} coverage). "
            f"Will use session-average fallback for gaps."
        )
    
    # Economic events
    event_count = count_events(start_date, end_date)
    if event_count < 10:
        issues.append(
            f"Only {event_count} economic events found. "
            f"R3 backtesting will be unreliable."
        )
    
    # DXY data
    dxy_count = count_dxy_data(start_date, end_date)
    if dxy_count == 0:
        issues.append("No DXY data. Regime engine will skip DXY correlation.")
    
    if issues:
        print("DATA WARNINGS:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("✅ All data sufficient for backtest")
    
    return len(issues) == 0
```

**How to start collecting data if you don't have enough historical M5**:

```python
# Add to your daily scheduler — start storing M5 bars
# Your system likely already stores M15/H1, but M5 may not be persisted
def store_m5_history():
    """
    Run once to backfill, then daily to maintain.
    MT5 typically provides 1-2 years of M5 data.
    """
    bars = mt5.copy_rates_range(
        "XAUUSD", mt5.TIMEFRAME_M5,
        datetime(2025, 1, 1),  # as far back as available
        datetime.now()
    )
    # Store to PostgreSQL
    insert_ohlcv_bars('M5', bars)
```

**Run command**:
```bash
# Backtest with default parameters
python backtest/engine.py --start 2025-01-01 --end 2026-03-31

# Walk-forward validation
python backtest/engine.py --start 2025-01-01 --end 2026-03-31 --walk-forward

# Single strategy isolation
python backtest/engine.py --start 2025-01-01 --end 2026-03-31 --strategy S1_LONDON_BRK
```

---

### 1B-2: Risk-of-Ruin Simulator

**Built on top of backtesting framework** — shares the trade log and equity curve infrastructure.

**What it does**: Takes your backtest results (or live trade history) and runs 10,000 Monte Carlo simulations with randomized trade ordering to answer: "What's the probability of hitting catastrophic drawdown under different conditions?"

```python
class RiskOfRuinSimulator:
    """
    Monte Carlo simulation for drawdown and ruin probability.
    
    Uses actual trade results (from backtest or live) as the 
    distribution, then randomizes order to simulate different
    possible sequences.
    
    KEY OUTPUTS:
    - Probability of hitting KS6 (12% drawdown)
    - Probability of hitting KS5 (-10% weekly) 
    - Expected max drawdown distribution
    - Worst-case equity path
    """
    
    def __init__(self, trade_results, initial_balance, n_simulations=10000):
        self.trades = trade_results  # list of {pnl, r_multiple, strategy}
        self.initial_balance = initial_balance
        self.n_sims = n_simulations
    
    def run_basic(self):
        """
        Basic Monte Carlo: shuffle trade order, 
        simulate equity path.
        """
        results = []
        pnl_values = [t['pnl'] for t in self.trades]
        
        for sim in range(self.n_sims):
            shuffled = random.sample(pnl_values, len(pnl_values))
            
            equity = self.initial_balance
            peak = equity
            max_dd = 0
            
            for pnl in shuffled:
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            
            results.append({
                'final_equity': equity,
                'max_drawdown': max_dd,
                'hit_ks6': max_dd >= 0.12,
                'hit_ruin': equity <= 0,
            })
        
        return self.analyze_results(results)
    
    def run_clustered(self, cluster_probability=0.3):
        """
        Clustered Monte Carlo: simulates correlated losses.
        
        With probability cluster_probability, the next trade 
        outcome matches the previous one (win→win, loss→loss).
        This models regime-dependent clustering that basic 
        shuffle misses.
        
        THIS IS THE IMPORTANT ONE — basic shuffle underestimates
        real drawdown risk because your trades cluster by regime.
        """
        results = []
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] <= 0]
        
        overall_wr = len(wins) / len(self.trades)
        
        for sim in range(self.n_sims):
            equity = self.initial_balance
            peak = equity
            max_dd = 0
            prev_win = random.random() < overall_wr
            
            for _ in range(len(self.trades)):
                # Cluster: with some probability, repeat outcome type
                if random.random() < cluster_probability:
                    is_win = prev_win  # same as last
                else:
                    is_win = random.random() < overall_wr
                
                if is_win:
                    pnl = random.choice(wins)
                else:
                    pnl = random.choice(losses)
                
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
                prev_win = is_win
            
            results.append({
                'final_equity': equity,
                'max_drawdown': max_dd,
                'hit_ks6': max_dd >= 0.12,
            })
        
        return self.analyze_results(results)
    
    def analyze_results(self, results):
        max_dds = [r['max_drawdown'] for r in results]
        
        report = {
            'simulations': len(results),
            'median_max_dd': np.median(max_dds),
            'p95_max_dd': np.percentile(max_dds, 95),
            'p99_max_dd': np.percentile(max_dds, 99),
            'prob_hit_ks6': sum(1 for r in results if r['hit_ks6']) / len(results),
            'prob_ruin': sum(1 for r in results if r.get('hit_ruin', False)) / len(results),
            'median_final_equity': np.median([r['final_equity'] for r in results]),
        }
        
        print("=" * 50)
        print("RISK OF RUIN ANALYSIS")
        print("=" * 50)
        print(f"Simulations:          {report['simulations']:,}")
        print(f"Median Max Drawdown:  {report['median_max_dd']:.1%}")
        print(f"95th %ile Max DD:     {report['p95_max_dd']:.1%}")
        print(f"99th %ile Max DD:     {report['p99_max_dd']:.1%}")
        print(f"Prob of hitting KS6:  {report['prob_hit_ks6']:.1%}")
        print(f"Prob of ruin:         {report['prob_ruin']:.2%}")
        print(f"Median Final Equity:  ${report['median_final_equity']:,.2f}")
        
        # Decision support
        if report['prob_hit_ks6'] > 0.15:
            print("\n⚠️  >15% chance of hitting KS6. "
                  "DO NOT scale to Phase 2 risk.")
        elif report['prob_hit_ks6'] > 0.05:
            print("\n⚠️  5-15% chance of hitting KS6. "
                  "Phase 2 risk marginal — consider wider KS6.")
        else:
            print("\n✅  <5% chance of hitting KS6. "
                  "Phase 2 scaling appears safe.")
        
        return report


# Usage (after backtest completes):
backtest_results = engine.run()
simulator = RiskOfRuinSimulator(
    backtest_results.trades, 
    initial_balance=10000
)

print("\n--- BASIC (independent trades) ---")
basic = simulator.run_basic()

print("\n--- CLUSTERED (correlated losses, 30% clustering) ---")
clustered = simulator.run_clustered(cluster_probability=0.3)

print("\n--- CLUSTERED (worst case, 50% clustering) ---")
worst = simulator.run_clustered(cluster_probability=0.5)
```

**When to run this**: After every significant backtest. Before scaling to Phase 2 risk. After any parameter change that affects sizing.

---

## PHASE 1C — BACKTEST-VALIDATED DEPLOYMENTS

Everything in this phase follows the same workflow:

```
1. Implement the logic in code
2. Run backtest WITH the change vs WITHOUT
3. Compare: expectancy, win rate, max DD, Sharpe, per-strategy P&L
4. If improvement confirmed → deploy to live
5. If unclear or negative → do not deploy
```

---

### 1C-1: Adaptive ATR Percentile Thresholds

**Implementation**:

```python
# Add to config.py
ATR_RECALIBRATION_ENABLED = True
ATR_BLEND_SHORT_DAYS = 90
ATR_BLEND_LONG_DAYS = 180
ATR_BLEND_SHORT_WEIGHT = 0.60
ATR_BLEND_LONG_WEIGHT = 0.40

# New module: calibration/atr_adaptive.py
def recalibrate_atr_thresholds():
    """
    Run every Sunday at midnight UTC.
    Computes blended ATR distribution and updates 
    regime engine thresholds.
    
    WHY BLENDED: 
    90-day alone whipsaws during transitional periods.
    180-day alone is too slow to adapt.
    60/40 blend balances responsiveness with stability.
    """
    now = datetime.utcnow()
    
    # Pull H1 ATR(14) values for both windows
    atr_90d = get_h1_atr_values(
        start=now - timedelta(days=ATR_BLEND_SHORT_DAYS),
        end=now
    )
    atr_180d = get_h1_atr_values(
        start=now - timedelta(days=ATR_BLEND_LONG_DAYS),
        end=now
    )
    
    if len(atr_90d) < 500 or len(atr_180d) < 1000:
        logger.warning(
            "Insufficient ATR data for recalibration. "
            "Using existing thresholds."
        )
        return None
    
    # Compute percentiles for each window
    pcts_short = {
        30: np.percentile(atr_90d, 30),
        55: np.percentile(atr_90d, 55),
        85: np.percentile(atr_90d, 85),
        95: np.percentile(atr_90d, 95),
    }
    pcts_long = {
        30: np.percentile(atr_180d, 30),
        55: np.percentile(atr_180d, 55),
        85: np.percentile(atr_180d, 85),
        95: np.percentile(atr_180d, 95),
    }
    
    # Blend
    thresholds = {}
    for pct in [30, 55, 85, 95]:
        thresholds[pct] = (
            ATR_BLEND_SHORT_WEIGHT * pcts_short[pct] + 
            ATR_BLEND_LONG_WEIGHT * pcts_long[pct]
        )
    
    # Sanity check: thresholds must be monotonically increasing
    if not (thresholds[30] < thresholds[55] < 
            thresholds[85] < thresholds[95]):
        logger.error(
            f"ATR recalibration produced non-monotonic thresholds: "
            f"{thresholds}. Keeping existing values."
        )
        return None
    
    # Log the change
    old = get_current_atr_thresholds()
    logger.info(
        f"ATR Recalibration:\n"
        f"  P30: {old[30]:.2f} → {thresholds[30]:.2f}\n"
        f"  P55: {old[55]:.2f} → {thresholds[55]:.2f}\n"
        f"  P85: {old[85]:.2f} → {thresholds[85]:.2f}\n"
        f"  P95: {old[95]:.2f} → {thresholds[95]:.2f}"
    )
    
    # Persist to DB and update runtime config
    save_atr_thresholds(thresholds)
    update_regime_engine_thresholds(thresholds)
    
    return thresholds


# Add to scheduler
scheduler.add_job(
    recalibrate_atr_thresholds,
    trigger='cron',
    day_of_week='sun',
    hour=0, minute=0,
    timezone='UTC'
)
```

**Database addition**:
```sql
CREATE TABLE system_state.atr_calibration_log (
    id SERIAL PRIMARY KEY,
    calibration_date DATE NOT NULL,
    p30_old DECIMAL(8,4),
    p30_new DECIMAL(8,4),
    p55_old DECIMAL(8,4),
    p55_new DECIMAL(8,4),
    p85_old DECIMAL(8,4),
    p85_new DECIMAL(8,4),
    p95_old DECIMAL(8,4),
    p95_new DECIMAL(8,4),
    atr_90d_count INT,
    atr_180d_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Backtest validation**: Run full backtest with static thresholds vs adaptive thresholds. Compare regime distribution (are you spending less time stuck in NO_TRADE/UNSTABLE?) and overall system performance.

---

### 1C-2: Micro State Engine

**Implementation**:

```python
# New module: regime/micro_state.py

class MicroState:
    """
    Fast-updating volatility state that complements the 
    slow regime engine (H4 ADX + H1 ATR).
    
    Updates every M15 bar (vs H4 for macro regime).
    
    PURPOSE: Catches rapid environment changes that the 
    slow regime misses. Most common scenario:
    regime = NORMAL_TRENDING, but M15 volatility just collapsed
    → signals fire into a dead market → losses.
    
    STATES:
    - MICRO_NORMAL: conditions match macro regime expectation
    - MICRO_SPIKE: volatility exploding (only S8 should trade)
    - MICRO_ILLIQUID: spread widening + vol dropping (block all)
    - MICRO_DEAD: volatility collapsed (reduce sizing)
    - MICRO_ACCELERATION: vol increasing toward macro regime (boost)
    """
    
    # These thresholds need calibration via backtest
    # Starting values based on XAUUSD M15 characteristics
    SPIKE_THRESHOLD = 2.0       # current ATR > 2× baseline
    DEAD_THRESHOLD = 0.4        # current ATR < 0.4× baseline
    ACCELERATION_THRESHOLD = 1.3 # current ATR > 1.3× baseline (gentler)
    SPREAD_ILLIQUID_RATIO = 2.0  # current spread > 2× median
    
    def __init__(self):
        self.current_state = 'MICRO_NORMAL'
        self.state_since = datetime.utcnow()
        self.history = []  # last 10 states for pattern detection
    
    def update(self, atr_m15_fast, atr_m15_baseline, 
               current_spread, median_spread):
        """
        Call every M15 bar close.
        
        atr_m15_fast: ATR(3) on M15 — very recent volatility
        atr_m15_baseline: ATR(14) on M15 — normal volatility
        current_spread: live spread
        median_spread: 24h rolling median spread
        """
        if atr_m15_baseline == 0:
            return self.current_state
        
        vol_ratio = atr_m15_fast / atr_m15_baseline
        spread_ratio = (
            current_spread / median_spread 
            if median_spread > 0 else 1.0
        )
        
        old_state = self.current_state
        
        # Priority order matters — check most dangerous first
        
        # 1. Illiquid: spread blowing out while vol drops
        #    This catches pre-event liquidity withdrawal
        if (spread_ratio > self.SPREAD_ILLIQUID_RATIO and 
            vol_ratio < 0.8):
            self.current_state = 'MICRO_ILLIQUID'
        
        # 2. Spike: extreme vol expansion
        elif vol_ratio > self.SPIKE_THRESHOLD:
            self.current_state = 'MICRO_SPIKE'
        
        # 3. Dead: vol collapsed
        elif vol_ratio < self.DEAD_THRESHOLD:
            self.current_state = 'MICRO_DEAD'
        
        # 4. Acceleration: vol building (good for breakouts)
        elif vol_ratio > self.ACCELERATION_THRESHOLD:
            self.current_state = 'MICRO_ACCELERATION'
        
        # 5. Normal
        else:
            self.current_state = 'MICRO_NORMAL'
        
        # Track state change
        if self.current_state != old_state:
            self.state_since = datetime.utcnow()
            self.history.append({
                'time': self.state_since,
                'from': old_state,
                'to': self.current_state,
                'vol_ratio': vol_ratio,
                'spread_ratio': spread_ratio
            })
            # Keep last 10
            self.history = self.history[-10:]
            
            logger.info(
                f"MICRO STATE: {old_state} → {self.current_state} "
                f"(vol_ratio={vol_ratio:.2f}, spread_ratio={spread_ratio:.2f})"
            )
        
        return self.current_state
    
    def get_sizing_modifier(self):
        """
        Returns a multiplier to apply on top of existing sizing.
        This is a MODIFIER, not a replacement.
        """
        modifiers = {
            'MICRO_NORMAL':       1.0,
            'MICRO_SPIKE':        0.5,   # only S8 should be full size
            'MICRO_ILLIQUID':     0.0,   # block all new entries
            'MICRO_DEAD':         0.6,   # reduce — low vol = low edge
            'MICRO_ACCELERATION': 1.0,   # normal — regime handles this
        }
        return modifiers.get(self.current_state, 1.0)
    
    def allows_strategy(self, strategy_name):
        """
        Per-strategy permission based on micro state.
        """
        if self.current_state == 'MICRO_ILLIQUID':
            return False  # block everything
        
        if self.current_state == 'MICRO_SPIKE':
            # Only S8 (spike continuation) and R3 (event-driven)
            # should trade during spikes
            return strategy_name in ['S8_ATR_SPIKE', 'R3_CAL_MOMENTUM']
        
        if self.current_state == 'MICRO_DEAD':
            # Block breakout strategies in dead vol
            # Mean reversion (S2) might still work
            blocked_in_dead = [
                'S1_LONDON_BRK', 'S6_ASIAN_BRK', 
                'S7_DAILY_STRUCT', 'S5_NY_COMPRESS'
            ]
            return strategy_name not in blocked_in_dead
        
        return True  # NORMAL and ACCELERATION allow everything


# Integration into your signal evaluation loop:
# (add after regime check, before strategy evaluation)

micro = MicroState()

# In your M15 update cycle:
def on_m15_close(m15_bar):
    atr_fast = compute_atr(m15_bars[-3:], period=3)   # last 3 bars
    atr_base = compute_atr(m15_bars[-14:], period=14)  # last 14 bars
    spread = get_current_spread()
    median_spread = get_median_spread_24h()
    
    micro.update(atr_fast, atr_base, spread, median_spread)

# In strategy evaluation:
def evaluate_strategies():
    for strategy in active_strategies:
        # ... existing regime check ...
        
        # NEW: Micro state check
        if not micro.allows_strategy(strategy.name):
            tracker.record_block(strategy.name, 'micro_state',
                               f'state={micro.current_state}')
            continue
        
        # ... rest of evaluation ...
        
        # In sizing:
        micro_modifier = micro.get_sizing_modifier()
        # Apply alongside existing multipliers
        final_lots = calculated_lots * micro_modifier
```

**Backtest validation**: Run with and without micro state. Key metrics to compare:
- **Loss rate on MICRO_DEAD/MICRO_ILLIQUID entries** — these should be your worst-performing entries. If micro state blocks them, overall WR should improve.
- **Opportunity cost** — how many profitable trades would micro state have blocked? The starvation tracker catches this.

**Threshold calibration**: The initial values (2.0× spike, 0.4× dead, etc.) are starting points. Use backtest to test ranges:

```python
# Sensitivity analysis
for spike_thresh in [1.5, 1.75, 2.0, 2.25, 2.5]:
    for dead_thresh in [0.3, 0.4, 0.5, 0.6]:
        micro = MicroState()
        micro.SPIKE_THRESHOLD = spike_thresh
        micro.DEAD_THRESHOLD = dead_thresh
        result = run_backtest(micro_state=micro)
        print(f"Spike={spike_thresh}, Dead={dead_thresh}: "
              f"Exp={result.expectancy:+.3f}R, "
              f"WR={result.win_rate:.1%}, "
              f"DD={result.max_dd:.1%}")
```

---

### 1C-3: Intraday Equity Throttle

**Implementation**:

```python
# Add to config.py
EQUITY_THROTTLE_ENABLED = True
EQUITY_THROTTLE_PCT = -0.02       # -2% in 2 hours
EQUITY_THROTTLE_WINDOW_HOURS = 2
EQUITY_THROTTLE_SIZE_MULT = 0.50  # halve new position sizes
EQUITY_THROTTLE_CHECK_INTERVAL = 15  # check every 15 minutes

# New module: risk/equity_throttle.py
class IntraDayEquityThrottle:
    """
    Intermediate risk control between normal operation and KS3.
    
    Current gap:
    Normal (100% sizing) → KS3 at -4% daily (0% new entries)
    
    With throttle:
    Normal (100%) → Throttle at -2%/2hr (50%) → KS3 at -4% (0%)
    
    This catches "bad days" 1-2 hours earlier than KS3.
    """
    
    def __init__(self):
        self.equity_snapshots = []  # (timestamp, equity) ring buffer
        self.throttle_active = False
        self.throttle_activated_at = None
    
    def record_equity(self, equity):
        """Call every 15 minutes from scheduler."""
        now = datetime.utcnow()
        self.equity_snapshots.append((now, equity))
        
        # Keep only last 24 hours of snapshots
        cutoff = now - timedelta(hours=24)
        self.equity_snapshots = [
            (t, e) for t, e in self.equity_snapshots if t > cutoff
        ]
    
    def check_throttle(self, current_equity):
        """
        Returns sizing multiplier: 1.0 (normal) or 0.5 (throttled).
        Call before sizing new positions.
        """
        now = datetime.utcnow()
        window_start = now - timedelta(hours=EQUITY_THROTTLE_WINDOW_HOURS)
        
        # Find equity at window start (closest snapshot)
        past_snapshots = [
            (t, e) for t, e in self.equity_snapshots 
            if t <= window_start
        ]
        
        if not past_snapshots:
            return 1.0  # not enough history yet
        
        # Get the most recent snapshot before window start
        _, equity_at_window_start = max(past_snapshots, key=lambda x: x[0])
        
        change_pct = (
            (current_equity - equity_at_window_start) / 
            equity_at_window_start
        )
        
        if change_pct <= EQUITY_THROTTLE_PCT:  # e.g., <= -0.02
            if not self.throttle_active:
                self.throttle_active = True
                self.throttle_activated_at = now
                logger.warning(
                    f"EQUITY THROTTLE ACTIVATED: "
                    f"{change_pct:+.2%} in last "
                    f"{EQUITY_THROTTLE_WINDOW_HOURS}h. "
                    f"New position sizes reduced to "
                    f"{EQUITY_THROTTLE_SIZE_MULT:.0%}"
                )
            return EQUITY_THROTTLE_SIZE_MULT
        
        # Auto-deactivate if equity recovers
        if self.throttle_active and change_pct > EQUITY_THROTTLE_PCT / 2:
            self.throttle_active = False
            logger.info(
                "EQUITY THROTTLE DEACTIVATED: Equity stabilized."
            )
        
        return 1.0 if not self.throttle_active else EQUITY_THROTTLE_SIZE_MULT


# Integration into sizing engine:
throttle = IntraDayEquityThrottle()

# Add to scheduler (every 15 minutes):
scheduler.add_job(
    lambda: throttle.record_equity(get_current_equity()),
    trigger='interval',
    minutes=EQUITY_THROTTLE_CHECK_INTERVAL
)

# In position sizing function:
def compute_position_size(signal, state, spread):
    # ... existing calculation ...
    
    # After all existing multipliers, before final clamping:
    throttle_mult = throttle.check_throttle(get_current_equity())
    calculated_lots *= throttle_mult
    
    # ... existing floor/cap logic ...
```

**Important**: The throttle resets with the session, not the day. If throttle activates during London and equity stabilizes by NY open, NY trades get full sizing. This prevents one bad London session from killing your entire NY opportunity.

---

### 1C-4: 3-Tier Exit

**Implementation** — only deploy after backtest confirms improvement:

```python
# Add to config.py
EXIT_TIER_ENABLED = True  # feature flag — can disable instantly

# Old config (keep for comparison):
# PARTIAL_EXIT_R = 2.0  # 50% at 2R
# REMAINDER = trail with 2.5× ATR

# New config:
EXIT_TIERS = [
    {'r_target': 1.5, 'close_pct': 0.33, 'label': 'TIER_1'},
    {'r_target': 3.0, 'close_pct': 0.33, 'label': 'TIER_2'},
    # Remaining 34% trails with ATR
]
BE_ACTIVATION_R = 1.5   # BE activates at first partial
ATR_TRAIL_MULTIPLIER = 2.5  # unchanged

# Implementation in position management:
def manage_position_exits(pos, current_price, atr_m15):
    """
    Replaces single partial exit with 3-tier system.
    """
    if not EXIT_TIER_ENABLED:
        # Fall back to original logic
        return manage_position_exits_original(pos, current_price, atr_m15)
    
    current_r = compute_r_multiple(
        pos, current_price, pos.stop_price_original
    )
    
    for tier in EXIT_TIERS:
        tier_label = tier['label']
        
        # Check if this tier already executed
        if tier_label in pos.completed_tiers:
            continue
        
        if current_r >= tier['r_target']:
            close_lots = round(pos.original_lots * tier['close_pct'], 2)
            
            # Ensure we don't close more than remaining
            close_lots = min(close_lots, pos.current_lots)
            
            if close_lots >= 0.01:  # minimum lot
                partial_close(pos, close_lots, current_price, tier_label)
                pos.completed_tiers.add(tier_label)
                
                logger.info(
                    f"{pos.strategy} {tier_label}: Closed {close_lots} lots "
                    f"at {current_r:.1f}R. Remaining: {pos.current_lots} lots"
                )
            
            # Activate BE on first tier
            if tier_label == 'TIER_1' and not pos.be_activated:
                activate_breakeven(pos)
    
    # Trail remaining position (the 34%)
    if pos.be_activated and pos.current_lots > 0:
        trail_stop = compute_atr_trail(
            pos, atr_m15, ATR_TRAIL_MULTIPLIER
        )
        if pos.direction == 'BUY':
            pos.current_sl = max(pos.current_sl, trail_stop)
        else:
            pos.current_sl = min(pos.current_sl, trail_stop)
```

**Backtest comparison query**: After running backtest with both exit systems, compare:

```python
def compare_exit_systems(results_old, results_new):
    print(f"{'Metric':<25} {'2-Tier (old)':>15} {'3-Tier (new)':>15} {'Delta':>10}")
    print("-" * 65)
    
    metrics = [
        ('Expectancy (R)', results_old.expectancy, results_new.expectancy),
        ('Win Rate', results_old.win_rate, results_new.win_rate),
        ('Avg Winner (R)', results_old.avg_win_r, results_new.avg_win_r),
        ('Avg Loser (R)', results_old.avg_loss_r, results_new.avg_loss_r),
        ('Profit Factor', results_old.profit_factor, results_new.profit_factor),
        ('Max Drawdown', results_old.max_dd, results_new.max_dd),
        ('Sharpe Ratio', results_old.sharpe, results_new.sharpe),
        ('Trades reaching 1.5R', results_old.pct_reaching_1_5r, results_new.pct_reaching_1_5r),
        ('Trades reaching 3.0R', results_old.pct_reaching_3r, results_new.pct_reaching_3r),
    ]
    
    for name, old, new in metrics:
        delta = new - old
        flag = "✅" if delta > 0 else "❌" if delta < 0 else "—"
        print(f"{name:<25} {old:>15.3f} {new:>15.3f} {delta:>+9.3f} {flag}")
```

**Decision rule**: Deploy 3-tier ONLY if it improves expectancy AND doesn't increase max drawdown. If it improves one but worsens the other, stick with current system.

---

## PHASE 2 — ARCHITECTURAL EVOLUTION

---

### 2-1: Opportunity Budget Engine

**Prerequisites before building**:

1. Complete the multiplier audit — map every sizing modifier and permission gate
2. Have backtest results showing strategy correlation data
3. Decide: budget is a PERMISSION gate, NOT a sizing multiplier

**Multiplier Audit Template** — fill this out first by reading through your actual code:

```
CURRENT SIZING STACK (verify against your code):
================================================
1. Base risk: 1% or 2%
2. Conviction: 0.75 / 1.0 / 1.25
3. KS4 countdown: 0.5 for 3 trades
4. Severity multiplier: from event risk
5. Spread multiplier: from spread ratio
6. Vol scalar: from EWMA ATR percentile
7. Reduction floor: max(0.50, severity × spread × vol)
8. Regime multiplier: 0.0 / 0.4 / 0.7 / 0.8 / 1.0 / 1.5
9. Session multiplier: 0.7 / 1.0
10. Micro state modifier: 0.0 / 0.5 / 0.6 / 1.0 (NEW from 1C-2)
11. Equity throttle: 0.5 / 1.0 (NEW from 1C-3)

CURRENT PERMISSION GATES (verify against your code):
====================================================
1. Regime allows strategy? (yes/no)
2. Kill switches clear? (yes/no per KS1-7)
3. Compound gate > 0.35? (yes/no)
4. Family correlation kill? (yes/no)
5. Session lot cap? (yes/no)
6. Micro state allows? (yes/no, NEW from 1C-2)
7. [PROPOSED] Budget available? (yes/no)
```

**The budget sits at position 7 in the permission stack** — same level as portfolio brain, after all signal-level checks, before sizing.

```python
# New module: risk/opportunity_budget.py

class OpportunityBudgetEngine:
    """
    Allocates finite risk capacity per regime.
    Prevents strategy pile-up during correlated conditions.
    
    THIS IS A PERMISSION GATE, NOT A SIZING MULTIPLIER.
    It answers: "Is there room for this trade?" (yes/no)
    It does NOT change lot sizes.
    
    Budget values should be calibrated using:
    1. Backtest strategy correlation matrix
    2. Observed concurrent position counts per regime
    3. Risk-of-ruin simulator under different budget configs
    """
    
    # Budget per regime (total capacity = 1.0 in normal conditions)
    REGIME_BUDGET = {
        'SUPER_TRENDING':   1.5,   # more room in strong trends
        'NORMAL_TRENDING':  1.0,   # standard
        'WEAK_TRENDING':    0.8,   # slightly constrained
        'RANGING_CLEAR':    0.6,   # fewer strategies relevant
        'UNSTABLE':         0.3,   # only S3/S7
        'NO_TRADE':         0.0,   # S7 pending only
    }
    
    # Cost per strategy entry
    # Higher cost = consumes more budget = fewer concurrent trades
    # 
    # CALIBRATION PRINCIPLE:
    # - Strategies that correlate highly with others → higher cost
    # - Independent strategies → lower cost
    # - Strategies with larger stops → higher cost
    STRATEGY_COST = {
        'S1_LONDON_BRK':      0.35,  # primary, large position
        'S1B_FAILED_BRK':     0.20,  # reversal, smaller
        'S1D_PYRAMID':        0.10,  # add-on, inherently linked to S1
        'S1E_PYRAMID':        0.10,  # add-on
        'S1F_POST_TK':        0.20,  # continuation
        'S2_MEAN_REV':        0.25,  # counter-trend, different risk profile
        'S3_STOP_HUNT_REV':   0.20,  # reversal
        'S4_LONDON_PULL':     0.25,  # trend continuation
        'S5_NY_COMPRESS':     0.25,  # breakout
        'S6_ASIAN_BRK':       0.15,  # smaller, earlier session
        'S7_DAILY_STRUCT':    0.15,  # structural, half-size
        'S8_ATR_SPIKE':       0.15,  # event-driven, half-size
        'R3_CAL_MOMENTUM':    0.10,  # independent family, coexists
    }
    
    def __init__(self):
        self.active_allocations = {}  # position_id → cost
    
    def get_available_budget(self, current_regime):
        """Total budget minus consumed."""
        total = self.REGIME_BUDGET.get(current_regime, 0.5)
        consumed = sum(self.active_allocations.values())
        return total - consumed
    
    def can_take_trade(self, strategy_name, current_regime):
        """
        Permission check: is there budget for this trade?
        Returns (bool, context_string)
        """
        cost = self.STRATEGY_COST.get(strategy_name, 0.20)
        available = self.get_available_budget(current_regime)
        
        if available >= cost:
            return True, f"Budget OK: {available:.2f} available, {cost:.2f} needed"
        else:
            return False, (
                f"Budget EXHAUSTED: {available:.2f} available, "
                f"{cost:.2f} needed. "
                f"Active: {list(self.active_allocations.keys())}"
            )
    
    def allocate(self, position_id, strategy_name):
        """Call when a trade is opened."""
        cost = self.STRATEGY_COST.get(strategy_name, 0.20)
        self.active_allocations[position_id] = cost
        logger.info(
            f"Budget allocated: {strategy_name} = {cost:.2f}. "
            f"Remaining: {self.get_available_budget('NORMAL_TRENDING'):.2f}"
        )
    
    def release(self, position_id):
        """Call when a trade is closed."""
        if position_id in self.active_allocations:
            cost = self.active_allocations.pop(position_id)
            logger.info(
                f"Budget released: {cost:.2f}. "
                f"Remaining: {self.get_available_budget('NORMAL_TRENDING'):.2f}"
            )
    
    def status(self, current_regime):
        """For monitoring/logging."""
        return {
            'regime': current_regime,
            'total_budget': self.REGIME_BUDGET.get(current_regime, 0),
            'consumed': sum(self.active_allocations.values()),
            'available': self.get_available_budget(current_regime),
            'active_strategies': list(self.active_allocations.keys()),
        }


# Integration:
budget = OpportunityBudgetEngine()

# In signal evaluation (after existing permission gates):
allowed, context = budget.can_take_trade(strategy.name, current_regime)
if not allowed:
    tracker.record_block(strategy.name, 'budget', context)
    continue

# When trade opens:
budget.allocate(position.ticket, position.strategy)

# When trade closes:
budget.release(position.ticket)
```

**How to calibrate STRATEGY_COST values**: Use your backtest correlation matrix. If S1 and S4 have daily PnL correlation > 0.6, their combined cost should be high enough that both can't run simultaneously in WEAK_TRENDING (budget 0.8), but can in NORMAL_TRENDING (budget 1.0).

---