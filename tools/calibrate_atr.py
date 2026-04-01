#!/usr/bin/env python3
"""
tools/calibrate_atr.py

Connects to MT5 via rpyc bridge, downloads 6 months of H1 XAUUSD,
calculates ATR(14, RMA), and prints the actual percentile distribution
for your IC Markets data feed.

Output: Suggested values for config.py ATR thresholds.

Usage:
    python tools/calibrate_atr.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pandas_ta as ta

def run_calibration():
    try:
        from utils.mt5_client import get_mt5
        mt5 = get_mt5()
        print("✅ MT5 connected via rpyc bridge")

        bars = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 130 * 24)
        if bars is None or len(bars) < 100:
            print("❌ Not enough H1 bars from MT5. Got:", len(bars) if bars else 0)
            sys.exit(1)

        source = "MT5 rpyc (IC Markets)"

    except Exception as e:
        print(f"⚠️  MT5 unavailable ({e}), falling back to yfinance...")
        import yfinance as yf
        raw = yf.download("GC=F", period="6mo", interval="1h", progress=False)
        if raw.empty:
            print("❌ yfinance fallback also failed. Connect MT5 first.")
            sys.exit(1)
        bars = raw[["Open", "High", "Low", "Close"]].rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        ).reset_index().rename(columns={"Datetime": "time"}).to_dict("records")
        source = "yfinance GC=F (FALLBACK — use MT5 for real IC Markets values)"

    df = pd.DataFrame(bars)
    df.columns = [c.lower() for c in df.columns]

    # MUST use mamode='RMA' — Wilder smoothing = TradingView + MT5 standard
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14, mamode="RMA")
    df.dropna(inplace=True)

    atr_values = df["atr"].values
    current_atr = atr_values[-1]

    print(f"\n── ATR(14, RMA) Calibration ─────────────────────────────────────")
    print(f"  Source:            {source}")
    print(f"  Bars analysed:     {len(atr_values):,}")
    print(f"  Period:            {df['time'].iloc[0] if 'time' in df.columns else 'N/A'}  →  latest")
    print(f"  Current ATR:       {current_atr:.2f} points")
    print()

    percentiles = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    print(f"  {'Percentile':<14} {'ATR Value (pts)':<18} {'Meaning'}")
    print(f"  {'─'*50}")
    for p in percentiles:
        v = np.percentile(atr_values, p)
        meaning = ""
        if p == 30:  meaning = "← quiet/low volatility"
        if p == 55:  meaning = "← SUPER_TRENDING threshold"
        if p == 85:  meaning = "← UNSTABLE threshold (ATR_PCT_UNSTABLE_THRESHOLD)"
        if p == 95:  meaning = "← NO_TRADE threshold (ATR_PCT_NO_TRADE_THRESHOLD)"
        print(f"  {p:<14} {v:<18.2f} {meaning}")

    # Current config values for comparison
    try:
        import config
        print(f"\n── Current config.py vs suggested ──────────────────────────────")
        print(f"  ATR_PCT_NO_TRADE_THRESHOLD:  {config.ATR_PCT_NO_TRADE_THRESHOLD} (spec says 95)")
        print(f"  ATR_PCT_UNSTABLE_THRESHOLD:  {getattr(config, 'ATR_PCT_UNSTABLE_THRESHOLD', 'NOT SET')} (spec says 85)")
        print(f"  ATR_PCT_SUPER_THRESHOLD:     {getattr(config, 'ATR_PCT_SUPER_THRESHOLD', 'NOT SET')} (spec says 55)")
    except Exception:
        pass

    # Suggested config values based on actual data
    no_trade_val  = np.percentile(atr_values, 90)
    unstable_val  = np.percentile(atr_values, 70)
    super_val     = np.percentile(atr_values, 55)
    quiet_val     = np.percentile(atr_values, 30)
    current_pct   = float((atr_values < current_atr).mean() * 100)

    print(f"\n── Suggested config.py values (based on your data) ─────────────")
    print(f"  ATR_PCT_UNSTABLE_THRESHOLD  = 85   # ≈ {unstable_val:.1f} pts on IC Markets")
    print(f"  ATR_PCT_NO_TRADE_THRESHOLD  = 95   # ≈ {no_trade_val:.1f} pts on IC Markets")
    print(f"  ATR_PCT_SUPER_THRESHOLD     = 55   # ≈ {super_val:.1f} pts on IC Markets")
    print(f"  ATR_PCT_QUIET_FLOOR         = 30   # ≈ {quiet_val:.1f} pts on IC Markets")
    print(f"\n  Current session ATR percentile: {current_pct:.1f}th")
    print(f"  → Regime implication: ", end="")
    if current_pct > 90:   print("NO_TRADE (volatility too high)")
    elif current_pct > 70: print("UNSTABLE")
    elif current_pct > 55: print("SUPER_TRENDING candidate (if ADX confirms)")
    elif current_pct > 30: print("NORMAL/WEAK TRENDING range")
    else:                   print("QUIET — ranging likely")
    print()

if __name__ == "__main__":
    run_calibration()