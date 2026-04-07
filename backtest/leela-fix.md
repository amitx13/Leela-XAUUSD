# Leela XAUUSD — Backtest Bug Fix Specification
> **Branch:** `fix-V3.0` | **Verified against actual code:** Yes, every line number confirmed | **Author:** Perplexity audit, April 7 2026

---

## ⚠️ READ BEFORE TOUCHING ANYTHING

1. **Apply fixes in the exact order listed in Section 0.** Out-of-order application has caused 0-trade runs before (specifically: fixing C4 without simultaneously fixing Mo3).
2. **Each fix section has: exact file path → exact function name → exact search string → exact replacement.** Use the search string to locate the line, not line numbers (which shift as you edit).
3. **Never** run a full multi-month backtest until the 5-day validation check in Section 7 passes.
4. **Do not add any other improvements** while applying these fixes. One change group at a time.

---

## Section 0 — Fix Order (non-negotiable)

| Step | ID | File | Dependency |
|------|----|------|------------|
| 1 | C1 | `backtest/risk/position_sizing.py` | None — standalone |
| 2 | C3 | `backtest/engine.py` | Must be before C4 |
| 3 | **C4 + Mo3** | `backtest/engine.py` + `backtest/risk/position_sizing.py` | **Apply both in same commit. C4 alone = 0 trades.** |
| 4 | C2 | `backtest/risk/kill_switches.py` | After C3 is in place |
| 5 | M1 | `backtest/risk/kill_switches.py` + `backtest/engine.py` | After C2 |
| 6 | M2 | `backtest/engine.py` | Standalone |
| 7 | M5 | `backtest/engine.py` | After C2 |
| 8 | M3 | `backtest/engine.py` | Standalone |
| 9 | Mo1 | `backtest/engine.py` | Standalone |

---

## Section 1 — FIX C1: REDUCTION_FLOOR logic wrong (Critical)

### Why it exists
`REDUCTION_FLOOR = 0.50` was meant to prevent reducing risk below 50% **of base_risk**. Instead the code uses it as a minimum for `compound_multiplier` — which is then multiplied by full equity. Result: every trade risks 50% of equity minimum instead of 1%.

### Secondary damage
`MIN_CONDITION_MULTIPLIER = 0.35` (the compound gate that blocks bad-condition trades) can **never fire** because the floor of 0.50 is always above the gate threshold of 0.35. All bad-condition trades go through unchecked.

### File
`backtest/risk/position_sizing.py`

### Function
`PositionSizer.calculate_lot_size()`

### Locate the line (search for this exact string)
```python
        # 5. Reduction floor (minimum 0.50×)
        compound_multiplier = max(size_before_compound, REDUCTION_FLOOR)
```

### Replace with
```python
        # 5. Reduction floor — prevents reducing below 50% OF base_risk (not 50% of equity)
        compound_multiplier = max(size_before_compound, base_risk * REDUCTION_FLOOR)
```

### What changes
- Before: `compound_multiplier = max(0.01, 0.50) = 0.50` → `risk_amount = equity × 0.50`
- After: `compound_multiplier = max(0.01, 0.01 × 0.50) = max(0.01, 0.005) = 0.01` → `risk_amount = equity × 0.01`
- The compound gate `< 0.35` will now correctly fire when conditions are poor

### Do NOT touch
- `REDUCTION_FLOOR` constant value (stays 0.50)
- `MIN_CONDITION_MULTIPLIER` constant value (stays 0.35)
- Any other line in this function

### Verify after applying
Add temporary print and run 1-day backtest:
```python
# Inside calculate_lot_size(), after computing compound_multiplier:
print(f"[C1-VERIFY] compound_mult={compound_multiplier:.5f} risk_amount={equity * compound_multiplier:.2f}")
```
Expected: `compound_mult=0.01000 risk_amount=100.00` at $10k equity. **If you see `risk_amount=5000.00` the fix did not apply.**

---

## Section 2 — FIX C3: KS3 + KS5 disabled on Day 1 (Critical)

### Why it exists
`KillSwitchChecker.__init__()` sets `self.day_start_balance = 0.0` and `self.week_start_balance = 0.0`. The engine's `_initialize_state()` never seeds these with the actual starting balance. `check_ks3_daily_loss()` has a guard `if starting_balance <= 0: return False` — so KS3 silently returns false all of day 1. Same for KS5.

### File
`backtest/engine.py`

### Function
`BacktestEngine._initialize_state()`

### Locate the end of the function (search for this exact string — it's the last line of the method)
```python
        self.state.independent_lanes_occupied = {}
```

### Add these 2 lines IMMEDIATELY AFTER that line (still inside the method, same indentation level)
```python
        self.kill_switch_checker.reset_daily(self.initial_balance)
        self.kill_switch_checker.reset_weekly(self.initial_balance)
```

### What changes
- `kill_switch_checker.day_start_balance` is now `initial_balance` from bar 1
- `kill_switch_checker.week_start_balance` is now `initial_balance` from bar 1
- KS3 and KS5 are now active from the very first bar

### Do NOT touch
- `KillSwitchChecker.__init__()` — leave `self.day_start_balance = 0.0` as-is (it gets overwritten by the new call)
- The midnight reset block — already correctly calls `reset_daily()` and `reset_weekly()`, leave it alone

### Verify after applying
```python
# In check_ks3_daily_loss(), add temporary print:
print(f"[C3-VERIFY] start_bal={starting_balance:.2f}")
```
Expected on bar 1: `start_bal=10000.00`. **If you see `start_bal=0.00` the fix did not apply.**

---

## Section 3 — FIX C4 + Mo3: Portfolio checker never populated + never reset (Critical + Moderate)

> **⚠️ APPLY ALL 4 SUB-CHANGES IN THIS SECTION IN ONE COMMIT. Do not apply C4 without Mo3 or vice versa.**

### Why they exist (C4)
`_update_state_after_fill()` adds positions to `self.state.open_positions` but never calls `self.portfolio_checker.add_position()`. The checker's list stays `[]` forever, so VAR cap and session lots cap never trigger.

### Why they exist (Mo3)
The midnight block calls `self.kill_switch_checker.reset_daily()` but never calls `self.portfolio_checker.reset_daily()`. If C4 is fixed without this, `daily_var_used` accumulates across days indefinitely and blocks all trades from day 2 onward. **This is the exact cause of your previous 0-trades incident.**

### Also required (PortfolioRiskChecker.remove_position bug)
The existing `remove_position()` method does `if position in self.open_positions` — this is object identity comparison. Since we'll be passing a new dict, it will never match and positions will never be removed. Must be fixed as part of this same change.

---

### Change 3a — `backtest/engine.py` · `_update_state_after_fill()`

**Locate (search for this exact string):**
```python
        self.state.open_positions[position.ticket] = position
```

**Add these lines IMMEDIATELY AFTER (same indentation = 8 spaces):**
```python
        self.portfolio_checker.add_position({
            "strategy": position.strategy,
            "direction": position.direction,
            "lot_size": position.lot_size,
            "ticket": position.ticket,
            "atr_h1": self.state.__dict__.get("atr_h1", 20.0)
        })
```

---

### Change 3b — `backtest/engine.py` · `_update_state_after_trade()`

**Locate this block (search for this exact string):**
```python
        if trade.ticket in self.state.open_positions:
            position = self.state.open_positions[trade.ticket]
```

**Add these lines INSIDE the if-block, AFTER `position = self.state.open_positions[trade.ticket]`, BEFORE the strategy_family checks. Indentation = 12 spaces:**
```python
            self.portfolio_checker.remove_position({
                "strategy": position.strategy,
                "direction": position.direction,
                "lot_size": position.lot_size,
                "ticket": position.ticket,
                "atr_h1": self.state.__dict__.get("atr_h1", 20.0)
            })
```

---

### Change 3c — `backtest/engine.py` · midnight block · `_update_strategy_state()`

**Locate (search for this exact string):**
```python
            self.kill_switch_checker.reset_daily(self.state.balance)
```

**Add this line IMMEDIATELY AFTER (same indentation = 12 spaces):**
```python
            self.portfolio_checker.reset_daily()
```

---

### Change 3d — `backtest/risk/position_sizing.py` · `PortfolioRiskChecker.remove_position()`

**Locate the entire method (search for this exact string):**
```python
    def remove_position(self, position: Dict[str, Any]) -> None:
        """Remove position from tracking."""
        if position in self.open_positions:
            self.open_positions.remove(position)
            atr_h1 = position.get('atr_h1', 20.0)
            self.daily_var_used -= position.get('lot_size', 0) * atr_h1
```

**Replace the entire method body with:**
```python
    def remove_position(self, position: Dict[str, Any]) -> None:
        """Remove position from tracking — matches by ticket ID, not object identity."""
        ticket = position.get("ticket")
        existing = next((p for p in self.open_positions if p.get("ticket") == ticket), None)
        if existing:
            self.open_positions.remove(existing)
            atr_h1 = existing.get("atr_h1", 20.0)
            self.daily_var_used -= existing.get("lot_size", 0) * atr_h1
```

---

### Verify C4+Mo3 after applying

Add this temporary print to `check_portfolio_risk()`:
```python
print(f"[C4-VERIFY] open_pos={len(self.open_positions)} daily_var_used={self.daily_var_used:.4f}")
```

Run a 3-day backtest. Expected:
- Day 1: `open_pos` grows after fills, shrinks to 0 after all closes
- Day 2 bar 1: `daily_var_used=0.0000` ← proves Mo3 reset is working
- Trades should NOT be 0. **If trades=0, the portfolio VAR cap is still blocking — check that change 3c was applied.**

---

## Section 4 — FIX C2: KS6 checked on every M5 bar (Critical)

### Why it exists
`check_all_kill_switches()` calls `check_ks6_drawdown()` on every bar. This sets `trading_enabled=False` on every bar after a 20% drawdown. The recovery path (`handle_ks6_trigger_backtest`) only exists in the midnight block in `engine.py`, not inside `check_all_kill_switches()`. So after a 20% drawdown, trading is blocked intraday forever (until midnight), then re-enabled, then blocked again on the very next bar.

### File
`backtest/risk/kill_switches.py`

### Function
`KillSwitchChecker.check_all_kill_switches()`

### Locate this exact block (it is 6 lines)
```python
        # KS6: Drawdown circuit breaker
        ks6_triggered, ks6_reason = check_ks6_drawdown(equity, self.peak_equity)
        if ks6_triggered:
            results['triggered_switches'].append('KS6')
            results['state_updates']['trading_enabled'] = False
            results['state_updates']['shutdown_reason'] = ks6_reason
```

### Replace with (add explanatory comment, remove logic)
```python
        # KS6: Intentionally NOT checked here.
        # KS6 is a daily circuit breaker handled exclusively at midnight
        # in BacktestEngine._update_strategy_state() which has the correct
        # auto-reset recovery path. Checking it here causes an intraday
        # permanent block with no recovery.
```

### Do NOT touch
- `check_ks6_drawdown()` function definition — leave it intact
- The `self.peak_equity` update block above this section — leave it intact
- The midnight KS6 check in `engine.py` — leave it completely untouched

### Verify after applying
Run 3-day backtest. The log should show **zero** `Kill switches triggered: ['KS6']` lines on non-midnight bars. KS6 may still appear in midnight logs — that is correct behaviour.

---

## Section 5 — FIX M1: KS4 countdown burns per bar not per trade (Major)

### Why it exists
`check_ks4_loss_streak()` decrements `ks4_reduced_trades_remaining` inside itself. `check_all_kill_switches()` calls this function on every M5 bar. At 12 bars/hour, a 3-trade countdown evaporates in ~15 minutes instead of after 3 actual trades.

Two changes required: (a) stop decrementing in the bar-level function, (b) decrement after each trade close in the engine.

---

### Change 5a — `backtest/risk/kill_switches.py` · `check_ks4_loss_streak()`

**Locate (search for this exact string):**
```python
    if ks4_reduced_trades_remaining > 0:
        new_countdown = ks4_reduced_trades_remaining - 1
        return False, f"KS4: Size reduction active - {new_countdown} trades remaining", new_countdown
```

**Replace with:**
```python
    if ks4_reduced_trades_remaining > 0:
        # Countdown is decremented per trade-close in engine._update_state_after_trade(), not here.
        return False, f"KS4: Size reduction active - {ks4_reduced_trades_remaining} trades remaining", ks4_reduced_trades_remaining
```

---

### Change 5b — `backtest/engine.py` · `_update_state_after_trade()`

**Locate (search for this exact string — it's the consecutive_losses update block):**
```python
        if trade.pnl_net_dollars < 0:
            self.kill_switch_checker.update_trade_result(-1)
        else:
            self.kill_switch_checker.update_trade_result(1)
```

**Add these lines IMMEDIATELY AFTER (same indentation = 8 spaces):**
```python
        if self.kill_switch_checker.ks4_reduced_trades_remaining > 0:
            self.kill_switch_checker.ks4_reduced_trades_remaining -= 1
            self.state.ks4_reduced_trades_remaining = self.kill_switch_checker.ks4_reduced_trades_remaining
```

---

## Section 6 — FIX M2: state.weekly_pnl never resets on Monday (Major)

### Why it exists
The midnight block resets `self.state.daily_pnl = 0.0` but there is no `self.state.weekly_pnl = 0.0` in the Monday block. `kill_switch_checker.reset_weekly()` updates the checker's **baseline** but not the state's accumulated P&L. KS5 therefore checks an ever-growing `weekly_pnl` against a freshly-reset baseline — making it progressively harder to trigger after a profitable first week.

### File
`backtest/engine.py`

### Function
`BacktestEngine._update_strategy_state()`

### Locate (search for this exact string)
```python
            if bar_time.weekday() == 0:  # Monday
                self.kill_switch_checker.reset_weekly(self.state.balance)
```

### Add one line INSIDE the if-block, AFTER `reset_weekly()` (indentation = 16 spaces)
```python
                self.state.weekly_pnl = 0.0
```

### Final result should look like
```python
            if bar_time.weekday() == 0:  # Monday
                self.kill_switch_checker.reset_weekly(self.state.balance)
                self.state.weekly_pnl = 0.0
```

---

## Section 7 — FIX M5: KS6 auto-reset leaves orphan positions in execution_sim (Major)

### Why it exists
The midnight KS6 handler rebuilds `self.state` from a fresh `SimulatedState()`. But `self.execution_sim.open_positions` and `self.pending_orders` are not cleared. The execution simulator continues managing the old positions, and when they close they modify the equity of the brand-new state, creating ghost P&L.

### File
`backtest/engine.py`

### Function
`BacktestEngine._update_strategy_state()` — inside the midnight KS6 auto-reset block

### Locate (search for this exact string — it's the last line of the KS6 auto-reset block)
```python
                    self.state = new_state
```

### Add 2 lines IMMEDIATELY AFTER (same indentation = 20 spaces)
```python
                    self.execution_sim.open_positions.clear()
                    self.pending_orders.clear()
```

### Do NOT touch
- The `handle_ks6_trigger_backtest()` call above — leave it untouched
- The else-branch (non-auto-reset KS6) — leave it untouched

---

## Section 8 — FIX M3: Asian range live update uses wrong hour filter (Moderate)

### Why it exists
The live Asian range update at 05:30 UTC uses `0 <= hour < 6`, missing the 22:00–23:59 UTC portion of the session. The warmup backfill in the same file correctly uses `hour >= 22 or hour < 7`. The two are inconsistent.

### File
`backtest/engine.py`

### Function
`BacktestEngine._update_strategy_state()` — inside `elif session == "ASIAN":` block

### Locate (search for this exact string)
```python
                asian_bars = [b for b in m5_bars if 0 <= b['time'].hour < 6]
```

### Replace with
```python
                asian_bars = [b for b in m5_bars if b['time'].hour >= 22 or b['time'].hour < 7]
```

### Do NOT touch
- The `_backfill_ranges_from_warmup()` method — it already has the correct filter
- The `pre_london_range` calculation block above this — leave it untouched

---

## Section 9 — FIX Mo1: Sharpe ratio guard against insufficient data (Moderate)

### Why it exists
Sharpe is computed with `len(sorted_days) > 2`, so 3 calendar days (= 2 daily returns) is enough to produce an annualised number. With 4–5 days the result is statistically meaningless (e.g. –8.70 Sharpe from 3 data points).

### File
`backtest/engine.py`

### Function
`BacktestEngine._generate_results()` — inside the Sharpe calculation try block

### Locate (search for this exact string)
```python
            if len(sorted_days) > 2:
```

### Replace with
```python
            if len(sorted_days) >= 30:
```

And add a warning in the else-branch for short runs. Find the try/except block around Sharpe:

**Locate:**
```python
        except Exception as e:
            logger.warning(f"Sharpe calculation failed: {e}")
            sharpe = 0.0
```

**Add before that except block (after the if/else body, same indentation = 8 spaces):**
```python
        if len(sorted_days) > 0 and len(sorted_days) < 30:
            logger.warning(
                f"Sharpe ratio skipped: only {len(sorted_days)} trading days in run "
                f"(minimum 30 required for statistical validity). Set to 0."
            )
```

---

## Section 10 — Validation Protocol

> Run these checks **in order** before committing to a long backtest.

### Check A — Position sizing (after Step 1 C1)
```bash
python -m backtest.run --start 2025-08-01 --end 2025-08-02
```
Look in logs for `[C1-VERIFY]` prints.
- ✅ Pass: `compound_mult≈0.01000`, `risk_amount≈100.00` at $10k
- ❌ Fail: `risk_amount≈5000.00` — fix did not apply

### Check B — KS3 active day 1 (after Step 2 C3)
```bash
python -m backtest.run --start 2025-08-01 --end 2025-08-02
```
Look for `[C3-VERIFY]` print on the very first bar.
- ✅ Pass: `start_bal=10000.00`
- ❌ Fail: `start_bal=0.00` — fix did not apply

### Check C — Portfolio + daily reset (after Step 3 C4+Mo3)
```bash
python -m backtest.run --start 2025-08-01 --end 2025-08-03
```
Look for `[C4-VERIFY]` prints.
- ✅ Pass: positions tracked, day 2 bar 1 shows `daily_var_used=0.0000`, trade count > 0
- ❌ Fail: `trade count = 0` — portfolio VAR is blocking. Check change 3c was applied.
- ❌ Fail: `open_pos` never changes — change 3a was not applied.

### Check D — No KS6 spam (after Step 4 C2)
```bash
python -m backtest.run --start 2025-08-01 --end 2025-08-05 2>&1 | grep "KS6"
```
- ✅ Pass: KS6 lines appear only at 00:00 timestamps
- ❌ Fail: KS6 lines appear at random bars throughout the day — the block was not removed

### Check E — Full 5-day regression (all steps applied)
```bash
python -m backtest.run --start 2025-08-01 --end 2025-08-05
```
Expected after all fixes:
- Trade count: **10–40** (not 128 — that was inflated by C1)
- No `start_bal=0` warnings
- No KS6 spam
- No `0 trades executed` error
- Lot sizes in the 0.05–0.20 range (not always capped at 0.50)

### Final production run (only after Check E passes)
```bash
python -m backtest.run --start 2025-01-01 --end 2025-09-30
```
Do not add `--monte-carlo` until you verify trade count is 200+ from this run.

---

## Appendix — Files Modified Summary

| File | Changes | Sections |
|------|---------|---------|
| `backtest/risk/position_sizing.py` | C1: 1 line · C4/3d: 6 lines | §1, §3 |
| `backtest/risk/kill_switches.py` | C2: delete 6 lines · M1/5a: 2 lines | §4, §5 |
| `backtest/engine.py` | C3: 2 lines · C4/3a+3b+3c: ~15 lines · M1/5b: 3 lines · M2: 1 line · M5: 2 lines · M3: 1 line · Mo1: 4 lines | §2, §3, §5, §6, §7, §8, §9 |

**Total: 3 files, ~37 lines changed/added.**

