# Changelog — Leela XAUUSD

All notable changes to this project are documented here.
Format: `[Version] — Date — Description`

---

## [fix-V3.0] — 2026-04-05 — Full Backtest Parity with Live Algo

This release fixes **15 parity gaps** identified between the backtest framework
(`backtest/engine.py`) and the live trading system (`engines/`, `main.py`).
After these fixes the backtest faithfully models all 13 strategies, all 4
kill switches, the TLT macro bias gate, broker commission, and weekly PnL
tracking — achieving ~90%+ behavioural fidelity with the live algo.

### Strategies Added (Gaps 1–8)

| Gap | ID | Description |
|-----|----|-------------|
| GAP-1 | S1b | Failed Breakout Reversal — fires when a broken S1 attempt returns inside range and breaks the opposite boundary. Mirrors `evaluate_s1b_signal()` in `signal_engine.py`. |
| GAP-2 | S1d | M5 EMA20 Pullback Re-entry — pyramids into an open S1 family position on EMA20 body-close touches. Max 8 adds (SUPER) / 5 (NORMAL). 5-min limit-order expiry. |
| GAP-3 | S1e | Confirmed Winner Pyramid — single add after partial exit AND BE activation, 0.5× original size. Hard limit: one per S1 campaign. |
| GAP-4 | S1f | Post Time-Kill NY Re-entry — after London 16:30 TK, re-enters in NY on EMA20 body close. Max 1 per day. SUPER/NORMAL regimes only. |
| GAP-5 | S4  | London EMA20(H1) Pullback — enters at EMA20 proximity during London in trending regime, using DI+/DI- for direction. |
| GAP-6 | S5  | NY Session Compression Breakout — triggers after tight 2-hour London consolidation breaks out during NY session (13–17 UTC). |
| GAP-7 | S8  | ATR Spike Continuation (independent lane) — fires when M15 bar range ≥ 2×ATR(H1) with confirmed continuation bar. Concurrent with trend family. |
| GAP-8 | R3  | Calendar Momentum (independent lane) — rides directional momentum after a high-impact event produces a strong M15 bar. 30-minute time-exit. Concurrent with trend family. |

### Kill Switches Added (Gaps 9–10)

| Gap | Kill Switch | Description |
|-----|-------------|-------------|
| GAP-9  | KS5 | Weekly loss limit — halts all new entries when weekly PnL exceeds `KS5_WEEKLY_LOSS_LIMIT_PCT`. Mirrors live `check_ks5_weekly_loss()`. |
| GAP-10 | KS6 | Account drawdown kill switch — halts trading when equity drawdown from `peak_equity` exceeds `KS6_DRAWDOWN_LIMIT_PCT`. Mirrors live `check_ks6_drawdown()`. |

### Macro Bias Gate (Gap 11)

| Gap | Feature | Description |
|-----|---------|-------------|
| GAP-11 | TLT Macro Bias | Since the backtest cannot call the live TLT bond feed, a structural proxy is derived from `EMA20(H1)` + `DI+/DI-`: `LONG_ONLY` when trending up, `SHORT_ONLY` when trending down, `BOTH_PERMITTED` otherwise. Applied to S1, S1b, S1f, S4, S5, S8, and R3. Mirrors the live `calculate_macro_bias()` gate. |

### Signal Timing Fix (Gap 12)

| Gap | Feature | Description |
|-----|---------|-------------|
| GAP-12 | S1/S1b M5 Evaluation | S1 and S1b are now evaluated on **every M5 bar** (not just M15 completions), matching the live system's `m15_dispatch_job` tick cadence. Catches intra-M15 breakouts that M15-only evaluation would miss. |

### Commission Deduction (Gap 13)

| Gap | Feature | Description |
|-----|---------|-------------|
| GAP-13 | Broker Commission | `$7.00/lot` round-trip commission is now deducted on every closed trade. Stored in `TradeRecord.commission`. Cumulative tracked in `SimulatedState.total_commission` and `daily_commission_paid`. |

### KS4 Cooldown (Gap 14)

| Gap | Feature | Description |
|-----|---------|-------------|
| GAP-14 | KS4 Cooldown | After 3 consecutive losses, `MAX_S1_FAMILY_ATTEMPTS` is capped at 2 for the rest of that day. Mirrors the live `decrement_ks4_countdown()` behaviour. |

### Weekly PnL Tracking (Gap 15)

| Gap | Feature | Description |
|-----|---------|-------------|
| GAP-15 | Weekly PnL | `SimulatedState.weekly_pnl` is accumulated per trade and reset every Monday. Required by KS5 gate. |

---

### Files Changed

| File | Change |
|------|--------|
| `backtest/engine.py` | **Primary change** — all 15 gaps implemented. 84 KB canonical engine. |
| `backtest/models.py` | `SimulatedState` expanded: `weekly_pnl`, `consecutive_losses`, `s4_fired_today`, `s5_fired_today`, `failed_breakout_flag/direction`, `s1d/s1e pyramid counts`, `r3_*`, `s8_*` independent lane fields, `TradeRecord.commission/pnl_gross`. |
| `backtest/strategies.py` | `ALL_STRATEGIES` updated to include all 13 strategies. `STRATEGY_REGISTRY` and `STRATEGY_GROUPS` added. |
| `backtest/run.py` | Import pinned to `backtest.engine`; stale engine imports removed; comment added clarifying canonical engine. |
| `backtest/engine_enhanced.py` | **DELETED** — superseded by `engine.py`. |
| `backtest/engine_enhanced_fixed.py` | **DELETED** — superseded by `engine.py`. |

---

### Known Residual Approximations

These are accepted simplifications that cannot be perfectly replicated without
a live broker feed:

1. **TLT macro bias** — approximated via EMA20(H1) + DI directional proxy.
   The live system uses actual TLT bond price data.
2. **Economic calendar** — `HistoricalEventFeed` uses static/mocked event
   data rather than a real Forex Factory API feed. Blackout accuracy is
   approximate around NFP, FOMC, and other high-impact events.
3. **Spread feed** — `HistoricalSpreadFeed` infers spread from OHLC data;
   does not use actual broker tick spreads. Spread widening during news events
   is approximated.
4. **Slippage** — fixed at `slippage_points=0.7`. Live slippage is dynamic
   and broker-dependent.

Overall backtest fidelity: **~90%** with the live algo.

---

## [V2.x] — Prior Releases

See git log for earlier version history.
