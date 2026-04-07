# Leela-XAUUSD Backtest Fix Plan
**Repo:** `amitx13/Leela-XAUUSD` | **Branch:** `fix-V3.0`  
**Generated:** 2026-04-07  
**Purpose:** Structured task list for a coding agent or developer to implement all fixes and verify them.

---

## How to Use This Document

Each task below is a **self-contained unit**:
1. Read **WHAT** and **WHY** to understand the problem
2. Go to the **FILE** and **EXACT LOCATION** listed
3. Apply the **EXACT CODE FIX** shown (copy-paste ready)
4. Run the **VERIFICATION CHECKLIST** to confirm the fix works
5. Mark the task ✅ DONE

Tasks are ordered by priority. **Complete and verify all P1 tasks before starting P2.**

---

## Priority Reference

| Priority | Label | Meaning |
|----------|-------|---------|
| P1 | CRITICAL | Engine is broken without this fix. Results are invalid. |
| P2 | HIGH | Results are statistically misleading without this fix. |
| P3 | MEDIUM | Results are incomplete / diverge from live behavior. |
| P4 | ENHANCEMENT | Full coverage of live algo behavior. |

---

---

# P1 — CRITICAL FIXES
## (Complete ALL before running any backtest)

---

### TASK-1: Fix Daily Counters Resetting Every Bar
**Priority:** P1 — CRITICAL  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
In `engine.py`, the `_update_strategy_state()` method has a daily reset block that should only run at midnight (00:00 UTC). However, the reset statements are placed **outside** the `if bar_time.hour == 0 and bar_time.minute == 0:` block and run on **every single M5 bar** (every 5 minutes). This means:
- `s4_fired_today`, `s5_fired_today`, `r3_fired_today` etc. reset every 5 minutes — strategies that should fire ONCE per day can fire every bar
- `daily_pnl = 0.0` resets every bar — KS3 (daily loss circuit breaker) never accumulates loss
- `kill_switch_checker.reset_daily()` is called thousands of times per day — KS3/KS4 are non-functional

**FILE:** `backtest/engine.py`  
**METHOD:** `_update_strategy_state()`  
**LOCATE THIS CODE BLOCK** (currently sitting outside the if-block):

```python
# Reset daily counters and kill-switch baselines
self.state.s1_family_attempts_today = 0
self.state.s1d_fired_today = False
self.state.s1d_pyramid_count = 0
# ... (all the reset lines) ...
self.state.daily_pnl = 0.0
self.state.daily_trades = 0
self.state.daily_commission_paid = 0.0
self.kill_switch_checker.reset_daily(self.state.balance)
if bar_time.weekday() == 0:
    self.kill_switch_checker.reset_weekly(self.state.balance)
```

**EXACT FIX — Indent all reset statements inside the midnight if-block:**

```python
def _update_strategy_state(self, bar, context, upcoming_events):
    bar_time = bar.time
    session = self._get_session(bar_time)

    if session == "LONDON":
        if bar_time.hour == 7 and bar_time.minute == 55:
            # ... pre-london range calculation (keep as-is) ...
            pass

    elif session == "ASIAN":
        if bar_time.hour == 5 and bar_time.minute == 30:
            # ... asian range calculation (keep as-is) ...
            pass

    # ─── DAILY RESET — runs ONLY at midnight UTC ───────────────────────
    if bar_time.hour == 0 and bar_time.minute == 0:
        # Previous day OHLC
        m15_bars = self.bar_buffer.get_latest_bars('M15', 96)
        if len(m15_bars) >= 96:
            prev_high = max(b['high'] for b in m15_bars)
            prev_low = min(b['low'] for b in m15_bars)
            self.state.prev_day_ohlc = {
                'high': prev_high, 'low': prev_low,
                'open': m15_bars[0]['open'],
                'close': m15_bars[-1]['close']
            }

        # KS6: Drawdown circuit breaker (checked daily at midnight)
        ks6_triggered, ks6_reason = check_ks6_drawdown(
            self.state.equity, self.state.peak_equity, self.state.to_dict()
        )
        if ks6_triggered:
            if should_enable_ks6_auto_reset() and "AUTO_RESET" in ks6_reason:
                updated_state_dict = handle_ks6_trigger_backtest(
                    self.state.to_dict(),
                    self.processed_bars,
                    bar._asdict(),
                    (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
                )
                new_state = SimulatedState()
                for k, v in updated_state_dict.items():
                    if hasattr(new_state, k):
                        setattr(new_state, k, v)
                self.state = new_state
            else:
                self.state.trading_enabled = False
                self.state.shutdown_reason = ks6_reason

        # Reset daily counters
        self.state.s1_family_attempts_today = 0
        self.state.s1d_fired_today = False
        self.state.s1d_pyramid_count = 0
        self.state.s1e_pyramid_count = 0
        self.state.s1f_reentered_today = False
        self.state.s1f_post_tk_active = False
        self.state.s4_fired_today = False
        self.state.s4_ema_touched = False
        self.state.s1d_ema_touched_today = False
        self.state.s5_fired_today = False
        self.state.range_computed = False
        self.state.r3_fired_today = False
        self.state.s8_fired_today = False
        self.state.s2_fired_today = False
        self.state.s3_fired_today = False
        self.state.s6_placed_today = False
        self.state.s7_placed_today = False
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.daily_commission_paid = 0.0
        self.kill_switch_checker.reset_daily(self.state.balance)

        if bar_time.weekday() == 0:  # Monday
            self.kill_switch_checker.reset_weekly(self.state.balance)

    # ─── Event-related state (runs every bar — keep outside if-block) ──
    if upcoming_events:
        for event in upcoming_events:
            event_time = event.get('time')
            if event_time and (bar_time - event_time).total_seconds() <= 0:
                self.state.r3_pre_event_price = bar.close
                self.state.r3_pre_event_atr = context.get('atr_h1', 20)
```

**VERIFICATION CHECKLIST:**
- [ ] Add a test: log `self.state.s4_fired_today` value at bar 5 of a simulated day — should NOT be False if it was set True at bar 1
- [ ] Set `self.state.s4_fired_today = True` manually at bar 1 of a day, confirm it stays True for bars 2–287 (rest of day), then resets to False at bar 288 (midnight)
- [ ] Add a debug print: count how many times `kill_switch_checker.reset_daily` is called — should be exactly 1 per calendar day, not thousands

---

### TASK-2: Fix Same-Bar Fill Look-Ahead Bias
**Priority:** P1 — CRITICAL  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
In `execution_simulator.py`, `process_pending_orders()` is called in `_process_bar()` with orders generated in the SAME bar iteration. The fill logic allows a BUY_STOP order placed at bar `t` to be filled on bar `t` if `bar_high >= price`. In live trading, a stop order placed at 10:05 cannot be filled until 10:10 (the next bar). This gives the backtest an unfair advantage — it enters at the breakout price with perfect timing, while in live trading the entry is the next bar (potentially at a worse price).

**FILE:** `backtest/execution_simulator.py`  
**METHOD:** `process_pending_orders()`  
**LOCATE THIS SECTION** (the for-loop start):

```python
for order in pending:
    # Expiry check
    if order.expiry and bar_time > order.expiry:
        ...
```

**EXACT FIX — Add a one-bar delay guard as the first check:**

```python
for order in pending:
    # ONE-BAR DELAY: Orders placed on bar T cannot be filled until bar T+1.
    # This prevents same-bar fill look-ahead bias.
    if order.placed_time is not None and bar_time <= order.placed_time:
        remaining.append(order)
        continue

    # Expiry check
    if order.expiry and bar_time > order.expiry:
        logger.debug(f"Order expired: {order.strategy} {order.direction} @ {order.price:.2f}")
        continue
    ...
```

**ALSO VERIFY:** Confirm that `SimOrder` dataclass in `backtest/models.py` has a `placed_time` field. If it does not, add it:
```python
@dataclass
class SimOrder:
    ...
    placed_time: Optional[datetime] = None  # Add this field if missing
    ...
```

And confirm that `_create_order()` in `engine.py` populates `placed_time=bar_time`:
```python
def _create_order(self, signal, bar_time):
    return SimOrder(
        ...
        placed_time=bar_time,   # ← Confirm this line exists
        ...
    )
```

**VERIFICATION CHECKLIST:**
- [ ] Create a test: place a BUY_STOP order at bar_time=T where bar_high at T >= order.price. Confirm no fill is returned for bar T.
- [ ] Confirm the same order is filled at bar T+1 when bar_high T+1 >= order.price.
- [ ] Confirm expiry check still works: place an order at T with expiry=T+2; confirm it fills at T+1 if price is hit, and expires at T+3 if not.
- [ ] Run a short 1-week backtest and compare entry_time to the bar where the signal was generated — every entry_time should be >= signal_bar + 5min.

---

### TASK-3: Fix POINT_VALUE Inconsistency in engine.py
**Priority:** P1 — CRITICAL  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
`execution_simulator.py` correctly defines `POINT_VALUE = 100.0`. But `engine.py` has its own module-level constant `POINT_VALUE = 1.0` which is factually wrong for XAUUSD (1 USD price move on 1 standard lot = $100, not $1). This orphan constant is misleading and will cause bugs if any future code in `engine.py` uses it directly for P&L calculation. The `_finalize_backtest()` method currently hardcodes `100.0` inline, bypassing the constant — this works now but is fragile.

**FILE:** `backtest/engine.py`  
**LOCATE AT TOP OF FILE:**

```python
# Trading constants for XAUUSD
POINT_VALUE = 1.0  # XAUUSD: 1 point = $1 USD per standard lot   ← WRONG
COMMISSION_PER_LOT_RT = 7.00
```

**EXACT FIX:**

```python
# Trading constants for XAUUSD
POINT_VALUE = 100.0        # XAUUSD: $1 price move × 100 oz/lot = $100 per standard lot
COMMISSION_PER_LOT_RT = 7.00
```

**ALSO IN `_finalize_backtest()`**, replace the hardcoded `100.0` with the constant:

```python
# BEFORE:
pnl_usd = pnl_points * 100.0 * position.lot_size  # POINT_VALUE=100

# AFTER:
pnl_usd = pnl_points * POINT_VALUE * position.lot_size
```

**VERIFICATION CHECKLIST:**
- [ ] `grep -n "POINT_VALUE" backtest/engine.py` — should show `POINT_VALUE = 100.0` and usages of the constant, no hardcoded `1.0` or `100.0` for P&L math.
- [ ] Run `_finalize_backtest()` on a test position: LONG 1 lot, entry=2000.0, forced-close at 2010.0 → P&L should be $1000 gross ($10 points × $100/lot).
- [ ] Cross-check: same calculation in `execution_simulator.py` `_create_trade_record()` should yield identical P&L for the same scenario.

---

---

# P2 — HIGH PRIORITY FIXES
## (Complete after all P1 tasks pass. Results are statistically misleading without these.)

---

### TASK-4: Fix Sharpe Ratio — Per-Bar Returns vs Daily Returns
**Priority:** P2 — HIGH  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
`_generate_results()` in `engine.py` computes Sharpe ratio on per-M5-bar equity returns. Since most M5 bars have zero equity change (no trades closing), the standard deviation of returns is artificially suppressed. This inflates Sharpe by an estimated 3–8× compared to the industry-standard calculation. Additionally, the result is not annualized.

**FILE:** `backtest/engine.py`  
**METHOD:** `_generate_results()`  
**LOCATE THIS SECTION:**

```python
# Calculate Sharpe ratio (simplified)
if len(equity_values) > 1:
    returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = avg_return / std_return if std_return > 0 else 0
else:
    sharpe = 0
```

**EXACT FIX — Replace with daily-return Sharpe:**

```python
# Calculate Sharpe ratio — on DAILY returns, annualized
# Group equity_curve by calendar date, take end-of-day (last) equity per day
sharpe = 0.0
try:
    daily_equity = {}
    for ep in self.equity_curve:
        day_key = ep.time.date()
        daily_equity[day_key] = ep.equity  # overwrites with last bar of each day

    sorted_days = sorted(daily_equity.keys())
    if len(sorted_days) > 2:
        daily_equities = [daily_equity[d] for d in sorted_days]
        daily_returns = [
            daily_equities[i] / daily_equities[i-1] - 1
            for i in range(1, len(daily_equities))
        ]
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        # Annualize: multiply by sqrt(252 trading days)
        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0
except Exception as e:
    logger.warning(f"Sharpe calculation failed: {e}")
    sharpe = 0.0
```

**VERIFICATION CHECKLIST:**
- [ ] Run a backtest over at least 30 trading days. Print both the old (per-bar) Sharpe and the new (daily) Sharpe. The new value should be significantly lower (typically 3–8× lower) — this is CORRECT.
- [ ] Verify the new Sharpe is in a realistic range: a good systematic strategy scores 0.8–2.0 annualized Sharpe. Values > 5.0 on daily returns are suspect.
- [ ] Edge case: if only 1 trading day of data, confirm `sharpe = 0.0` (no crash).

---

### TASK-5: Separate Partial Exits from Full Closes in Analytics
**Priority:** P2 — HIGH  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
When a position hits 2R, a `PARTIAL_EXIT` TradeRecord is created and appended to `self.trades`. This partial close is always profitable (exits at 2R = always a win). `_generate_results()` counts it as a full trade win, inflating win rate and trade count. A 100-trade strategy with 40 full wins and 40 partial exits would report 80/140 = 57% win rate, when the true win rate (on fully closed trades) is 40/100 = 40%.

**FILE:** `backtest/engine.py`  
**METHOD:** `_generate_results()`  
**LOCATE THIS SECTION:**

```python
total_trades = len(self.trades)
winning_trades = len([t for t in self.trades if t.pnl_net_dollars > 0])
losing_trades = total_trades - winning_trades
win_rate = winning_trades / total_trades if total_trades > 0 else 0
```

**EXACT FIX:**

```python
# Separate full closes from partial exits
full_trades = [t for t in self.trades if t.exit_type != 'PARTIAL_EXIT']
partial_exits = [t for t in self.trades if t.exit_type == 'PARTIAL_EXIT']

total_trades = len(full_trades)       # Report only full closes as "trades"
winning_trades = len([t for t in full_trades if t.pnl_net_dollars > 0])
losing_trades = len([t for t in full_trades if t.pnl_net_dollars <= 0])
win_rate = winning_trades / total_trades if total_trades > 0 else 0

# For P&L, expectancy, and R-multiple stats — use ALL trades (full + partial)
all_closed_trades = self.trades
total_pnl = sum(t.pnl_net_dollars for t in all_closed_trades)
total_commission = sum(t.commission for t in all_closed_trades)
```

**ALSO update the results dict to include partial exit stats:**

```python
results = {
    'summary': {
        ...
        'total_trades': total_trades,             # full closes only
        'partial_exits': len(partial_exits),       # ADD THIS
        'total_closures': len(self.trades),        # ADD THIS (full + partial)
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate * 100,
        ...
    },
    ...
}
```

**VERIFICATION CHECKLIST:**
- [ ] Find a strategy (S1 or S4) that regularly hits 2R. Confirm its win_rate before and after this fix — it should be LOWER after the fix (more realistic).
- [ ] Confirm `total_trades + partial_exits == total_closures` in the output.
- [ ] Confirm `total_pnl` is unchanged — P&L should include partial exit profits.
- [ ] Confirm `expectancy` is now computed on `full_trades` only (avg_win / avg_loss on full closes).

---

### TASK-6: Add Overnight Swap Cost Modeling
**Priority:** P2 — HIGH  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
There is no overnight swap deduction anywhere in the simulator. XAUUSD long positions typically incur a swap of -$5 to -$15 per standard lot per night. Strategies S1, S4, and S5 can hold positions overnight. Over a 12-month backtest with ~200 overnight holds at 0.1 lot average, this is a real omission (~$140–$300 per lot-year).

**FILE:** `backtest/engine.py`  
**METHOD:** `_update_equity_curve()` or add a new `_apply_daily_swap()` call inside `_process_bar()`

**EXACT FIX — Add swap deduction at midnight for each open position:**

Step 1: Add a constant near the top of `engine.py`:
```python
SWAP_PER_LOT_PER_NIGHT = -7.0  # USD per standard lot per overnight hold (XAUUSD long typical)
```

Step 2: In `_process_bar()`, after the existing indicator/regime calculation, add a daily swap check:
```python
# Apply overnight swap at midnight
if bar_time.hour == 0 and bar_time.minute == 0:
    self._apply_overnight_swap(bar_time)
```

Step 3: Add the new method to `BacktestEngine`:
```python
def _apply_overnight_swap(self, bar_time: datetime) -> None:
    """Deduct overnight swap cost for all open positions."""
    total_swap = 0.0
    for ticket, position in self.execution_sim.open_positions.items():
        # Only charge swap if position was open before today
        if position.entry_time.date() < bar_time.date():
            swap_cost = SWAP_PER_LOT_PER_NIGHT * position.lot_size
            total_swap += swap_cost

    if total_swap != 0.0:
        self.state.equity += total_swap
        self.state.balance = self.state.equity
        self.state.daily_pnl += total_swap
        logger.debug(f"Overnight swap applied: ${total_swap:.2f} at {bar_time.date()}")
```

**VERIFICATION CHECKLIST:**
- [ ] Open a simulated LONG position at 23:00 UTC with 1.0 lot. Confirm at 00:00 UTC the equity decreases by exactly `SWAP_PER_LOT_PER_NIGHT * 1.0` = -$7.00.
- [ ] Confirm positions entered at 00:01 UTC on day D do NOT get charged swap at 00:00 UTC on day D (they were opened AFTER midnight).
- [ ] Run a 30-day backtest with at least one overnight holder — compare total P&L with and without this fix; the diff should equal approximately `(number of overnight holds) × avg_lots × $7`.

---

---

# P3 — MEDIUM PRIORITY
## (Complete after P2. These fix behavioral divergence from live trading.)

---

### TASK-7: Load Real Spread CSV to Enable Dynamic Spreads
**Priority:** P3 — MEDIUM  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
Without a real spread CSV, `_get_current_spread()` returns a static `2.0` points on every bar. Real XAUUSD spreads range from 1.5 (liquid London hours) to 50+ points (news events, Asian open). The backtest overestimates P&L during high-spread periods.

**FILE:** `backtest/data_feed.py`  
**CLASS:** `HistoricalSpreadFeed`

**WHAT TO DO:**
1. Obtain M5 bid/ask OHLC data from your broker for the backtest date range
2. Export as a CSV with columns: `time` (UTC ISO format), `spread` (ask - bid, in price points)
3. Place the file at `backtest_data/spreads_XAUUSD_M5.csv`
4. Verify `HistoricalSpreadFeed.load()` reads from this path and `_get_current_spread()` stops returning the 2.0 fallback

**VERIFICATION CHECKLIST:**
- [ ] After loading the CSV, add a debug log: print the spread value for 3 different times — one in London session, one in Asian session, one at a news spike time. Confirm they differ.
- [ ] Confirm the 2.0 fallback (`return 2.0`) in `_get_current_spread()` is NOT being hit during the main backtest loop (add a counter).
- [ ] Visually inspect: average spread during London session should be ~1.5–3 pts; during Asian it should be ~2–5 pts; during NFP news it should spike to 10–50 pts.

---

### TASK-8: Load Real Events CSV to Enable KS7 News Blackout
**Priority:** P3 — MEDIUM  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
`_ks7_enabled` is `False` by default and only activates if `event_feed.loaded_from_csv = True`. Without a real events CSV, the backtest trades through all news events. In live trading, KS7 blocks all trading 30min before and 30min after high-impact USD/gold events. This is the largest single behavioral divergence from live.

**FILE:** `backtest/data_feed.py`  
**CLASS:** `HistoricalEventFeed`

**WHAT TO DO:**
1. Download economic calendar data (Forex Factory CSV export or similar) for the backtest period
2. Filter to: USD and XAU events, impact = HIGH only
3. Format as CSV with columns: `time` (UTC), `currency`, `event`, `impact`
4. Place at `backtest_data/events_calendar.csv`
5. Ensure `HistoricalEventFeed.load()` sets `self.loaded_from_csv = True` on successful load

**VERIFICATION CHECKLIST:**
- [ ] After loading, confirm `engine._ks7_enabled == True`
- [ ] Find a known NFP date in your backtest range. Confirm `trading_enabled = False` in the 30 minutes before and after that event timestamp.
- [ ] Count total bars where KS7 was active — should be non-zero. Log this in the results summary.

---

### TASK-9: Enforce Walk-Forward as Default (Not Opt-In)
**Priority:** P3 — MEDIUM  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
`run.py` only runs walk-forward analysis when `--walk-forward` CLI flag is passed. By default, all results are in-sample only. No in-sample-only result should be used for live deployment decisions.

**FILE:** `backtest/run.py`

**OPTION A — Change default to always run walk-forward:**
```python
# BEFORE:
parser.add_argument("--walk-forward", action="store_true", ...)

# AFTER:
parser.add_argument("--no-walk-forward", action="store_true",
                    help="Skip walk-forward analysis (not recommended)")
```

Then in `main()`:
```python
# BEFORE:
wf_results = run_walk_forward(results, args)

# AFTER:
if not args.no_walk_forward:
    wf_results = run_walk_forward(results, args)
else:
    logger.warning("Walk-forward analysis skipped. In-sample results only — not suitable for live deployment decisions.")
    wf_results = None
```

**OPTION B (minimum) — Add a prominent warning when walk-forward is skipped:**
```python
if not args.walk_forward:
    logger.warning("=" * 60)
    logger.warning("WARNING: Walk-forward analysis not enabled.")
    logger.warning("All results are IN-SAMPLE ONLY.")
    logger.warning("Do not use these results for live deployment.")
    logger.warning("Re-run with --walk-forward for valid out-of-sample results.")
    logger.warning("=" * 60)
```

**VERIFICATION CHECKLIST:**
- [ ] Run `python -m backtest.run --start 2025-01-01 --end 2025-06-30` without `--walk-forward`. Confirm the warning is printed prominently.
- [ ] Run with `--walk-forward`. Confirm OOS results are generated and printed.
- [ ] Confirm walk-forward window count makes sense: 6 months ÷ (3 month train + 1 month test) = ~2 windows.

---

---

# P4 — ENHANCEMENT
## (Do these last. They give you complete coverage of live algo behavior.)

---

### TASK-10: Add DXY Proxy to Unlock SUPER_TRENDING Regime
**Priority:** P4 — ENHANCEMENT  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
`dxy_macro` is hardcoded `None` in `_process_bar()`. The `SUPER_TRENDING` regime requires `dxy_macro < -0.70`, so it is permanently unreachable in backtest. The 1.5× position size multiplier (the algo's highest-conviction mode) is completely untested.

**FILE:** `backtest/engine.py` and `backtest/data_feed.py`

**WHAT TO DO:**
1. Add a new `HistoricalDXYFeed` class in `data_feed.py`:
   - Load daily DXY index or UUP ETF OHLC CSV
   - Method: `get_daily_change(bar_time: datetime) -> Optional[float]` returns daily % change (e.g., -0.85 for -0.85%)
2. In `BacktestEngine.__init__()`, add: `self.dxy_feed = HistoricalDXYFeed(start_date, end_date)`
3. In `_process_bar()`, replace the hardcoded `None`:
```python
# BEFORE:
regime, size_multiplier = self.regime_classifier.classify_regime(
    ..., None, ...   # dxy_macro hardcoded None
)

# AFTER:
dxy_macro = self.dxy_feed.get_daily_change(bar_time)  # Returns None if no data
regime, size_multiplier = self.regime_classifier.classify_regime(
    ..., dxy_macro, ...
)
```

**VERIFICATION CHECKLIST:**
- [ ] Find a date in the backtest where DXY dropped > 0.70% in a single day. Confirm `SUPER_TRENDING` regime fires on that date when ADX is also elevated.
- [ ] Count total bars with `SUPER_TRENDING` regime — should be > 0 in any 6-month period.
- [ ] Confirm position sizes during `SUPER_TRENDING` are 1.5× the NORMAL_TRENDING size for the same signal.

---

### TASK-11: Verify strategies.py vs main.py Parity
**Priority:** P4 — ENHANCEMENT  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
`backtest/strategies.py` is 39 KB. `main.py` is 72 KB. The engine claims "Matches live system exactly" but this size delta has never been verified. Any strategy logic branch in `main.py` absent from `strategies.py` is a silent accuracy gap.

**FILES:** `backtest/strategies.py` vs `main.py` (root)

**WHAT TO DO:**
1. Extract all strategy evaluation functions from both files
2. For each of the 10 strategies (S1, S1B, S1D, S1E, S1F, S2, S3, S4, S5, R3, S6, S7, S8), compare:
   - Entry condition logic
   - Stop distance formula
   - TP targets
   - Filter conditions (time, regime, session gates)
3. Document every discrepancy found as a new sub-task

**QUICK DIFF COMMAND:**
```bash
# Extract all def evaluate_* or def check_* function names from both files:
grep -n "^def \|^    def " backtest/strategies.py | sort > /tmp/strategies_fns.txt
grep -n "^def \|^    def " main.py | sort > /tmp/main_fns.txt
diff /tmp/strategies_fns.txt /tmp/main_fns.txt
```

**VERIFICATION CHECKLIST:**
- [ ] Run the diff above. For any function present in `main.py` but absent from `strategies.py`, create a new TASK to port it.
- [ ] For any function present in both, manually compare the condition logic (look for different threshold values, different context key names, different order types).
- [ ] Document findings in a comment block at the top of `backtest/strategies.py`.

---

### TASK-12: Add Indicator Warm-Up Buffer Before Backtest Start Date
**Priority:** P4 — ENHANCEMENT  
**Status:** ❌ OPEN

**WHAT IS WRONG:**  
EWM-based indicators (ATR, RSI, EMA with `adjust=False`) need ~100 bars to converge. The engine starts cold at `start_date`, so the first ~50 H1 bars produce unreliable ATR/RSI values that may generate false signals.

**FILE:** `backtest/engine.py`  
**METHOD:** `run()`

**EXACT FIX — Load warm-up bars before the main loop:**
```python
def run(self):
    ...
    m5_data = self.data_feed.load()
    
    # Load warm-up bars (100 H1 bars = ~4 days before start_date)
    warmup_start = self.start_date - timedelta(days=5)
    warmup_feed = HistoricalDataFeed(warmup_start, self.start_date)
    warmup_data = warmup_feed.load()
    
    # Pre-populate bar buffer with warm-up data (don't generate signals)
    for bar in warmup_data.itertuples():
        self.bar_buffer.add_m5_bar(bar._asdict())
    logger.info(f"Warm-up complete: {len(warmup_data)} bars pre-loaded")
    
    # Main loop (unchanged)
    for i, bar in enumerate(m5_data.itertuples(), 1):
        ...
```

**VERIFICATION CHECKLIST:**
- [ ] Print `context['atr_m15']` for bars 1–50 of the backtest WITH warm-up. It should not be the default `15.0` — it should reflect actual market ATR.
- [ ] Print `context['atr_m15']` for bars 1–50 WITHOUT warm-up. Many values should be the default `15.0`.
- [ ] Confirm the warm-up data is NOT generating trades (no signals fired, no orders placed during warm-up phase).

---

---

# POST-IMPLEMENTATION VERIFICATION PROTOCOL

After all tasks for a given priority level are complete, run the following checks before moving to the next priority.

## After P1 Tasks (TASK-1, 2, 3)

```
1. Run a 7-day mini backtest (pick a recent volatile week).
2. Confirm:
   a. Each strategy fires AT MOST ONCE per day (s4_fired_today guard works)
   b. No entry_time == signal_bar_time (1-bar delay enforced)
   c. Forced-close P&L on a LONG 1-lot +$10 trade = $1000 gross
   d. daily_pnl accumulates correctly — does not reset at bar 2
3. If any check fails, do not proceed to P2.
```

## After P2 Tasks (TASK-4, 5, 6)

```
1. Run a 30-day backtest.
2. Confirm:
   a. Annualized Sharpe ratio is between 0.0 and 5.0 (inflated values indicate bug)
   b. total_trades + partial_exits == total_closures
   c. win_rate is lower than before the partial-exit fix (this is correct)
   d. Equity curve shows small step-downs at midnight (swap deductions)
3. Export trades CSV and spot-check 3 overnight positions for swap charges.
```

## After P3 Tasks (TASK-7, 8, 9)

```
1. Run a 3-month backtest with spread CSV and events CSV loaded.
2. Confirm:
   a. Spread varies by session (inspect spread_log in debug output)
   b. KS7 blackout fires around at least 3 news events
   c. Walk-forward OOS results are generated automatically
   d. OOS Sharpe is lower than IS Sharpe (expected — overfitting check)
3. Compare 3-month P&L with/without news blackout — the difference is the "news slippage budget"
```

## After P4 Tasks (TASK-10, 11, 12)

```
1. Run a full-period backtest (maximum available date range).
2. Confirm:
   a. SUPER_TRENDING fires > 0 times
   b. strategies.py parity diff shows no unported logic
   c. First 50 bars have non-default ATR values (warm-up working)
3. Final Trust Score check:
   - Is overall confidence now >= 75%?
   - If yes: results are suitable for live deployment consideration.
   - If no: identify remaining gaps and document them.
```

---

## Final Expected Results After All Fixes

| Metric | Before | After All Fixes | Notes |
|--------|--------|-----------------|-------|
| Daily guards (KS3, KS4) | ❌ Non-functional | ✅ Working | TASK-1 |
| Entry timing | ❌ Same-bar look-ahead | ✅ Next-bar realistic | TASK-2 |
| Forced-close P&L | ⚠️ Uses wrong constant | ✅ Uses POINT_VALUE=100 | TASK-3 |
| Sharpe ratio | ❌ 3–8× inflated | ✅ Annualized daily | TASK-4 |
| Win rate | ⚠️ Partial exits inflate | ✅ Full closes only | TASK-5 |
| Overnight P&L | ⚠️ Overstated | ✅ Swap deducted | TASK-6 |
| Spread modeling | ⚠️ Static 2.0 pt | ✅ Dynamic CSV | TASK-7 |
| News events | ❌ Trades through news | ✅ KS7 active | TASK-8 |
| Validation | ❌ In-sample only | ✅ Walk-forward OOS | TASK-9 |
| SUPER_TRENDING | ❌ Never fires | ✅ DXY-driven | TASK-10 |
| Strategy parity | ⚠️ Unverified | ✅ Diffed and synced | TASK-11 |
| Indicator warmup | ⚠️ Cold start | ✅ 100-bar pre-load | TASK-12 |
| **Overall Trust** | **~35%** | **~85%** | |

