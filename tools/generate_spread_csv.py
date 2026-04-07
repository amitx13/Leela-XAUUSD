"""
tools/generate_spread_csv.py

Run ONCE to generate backtest_data/spreads_XAUUSD_M5.csv.
Produces session-aware XAUUSD M5 spreads covering Aug 2025 → Apr 2026.

Usage:
    python tools/generate_spread_csv.py

XAUUSD real-world spread profile used:
  Asian core (02-06 UTC):      ~3.0 pts  (low liquidity)
  Pre-London (06-07 UTC):      ~2.5 pts
  London open (07:00-07:15):   ~1.6 pts  (tight at open)
  London early (07:15-09:00):  ~1.8 pts
  London core (09-13 UTC):     ~1.7 pts  (most liquid)
  NY open (13-14 UTC):         ~1.5 pts  (tightest)
  NY active (14-17 UTC):       ~1.8 pts
  NY wind-down (17-19 UTC):    ~2.5 pts
  End of day (19-21 UTC):      ~3.2 pts
  Late NY / early Asian        ~3.8 pts
  Weekend:                     3.5 – 8.0 pts (near-zero liquidity)
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

start = pd.Timestamp("2025-08-01 00:00:00", tz="UTC")
end   = pd.Timestamp("2026-04-07 23:55:00", tz="UTC")
times = pd.date_range(start, end, freq="5min", tz="UTC")

rows = []
for t in times:
    h, m, wd = t.hour, t.minute, t.weekday()

    if wd >= 5:                           # Weekend
        spread = round(np.random.uniform(3.5, 8.0), 1)
    elif 22 <= h or h < 2:               # Late NY / early Asian
        spread = max(1.0, round(np.random.normal(3.5, 0.8), 1))
    elif 2 <= h < 6:                      # Asian core
        spread = max(1.0, round(np.random.normal(3.0, 0.7), 1))
    elif 6 <= h < 7:                      # Pre-London
        spread = max(1.0, round(np.random.normal(2.5, 0.5), 1))
    elif h == 7 and m < 15:              # London open spike (tightest)
        spread = max(1.0, round(np.random.normal(1.6, 0.5), 1))
    elif 7 <= h < 9:                      # London early
        spread = max(1.0, round(np.random.normal(1.8, 0.4), 1))
    elif 9 <= h < 13:                     # London core
        spread = max(1.0, round(np.random.normal(1.7, 0.35), 1))
    elif 13 <= h < 14:                    # NY open
        spread = max(1.0, round(np.random.normal(1.5, 0.4), 1))
    elif 14 <= h < 17:                    # NY active
        spread = max(1.0, round(np.random.normal(1.8, 0.4), 1))
    elif 17 <= h < 19:                    # NY wind-down
        spread = max(1.0, round(np.random.normal(2.5, 0.6), 1))
    elif 19 <= h < 21:                    # End of day
        spread = max(1.0, round(np.random.normal(3.2, 0.7), 1))
    else:                                 # Late NY
        spread = max(1.0, round(np.random.normal(3.8, 0.8), 1))

    rows.append({"time": t.isoformat(), "spread": spread})

out = Path(__file__).parent.parent / "backtest_data" / "spreads_XAUUSD_M5.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Done. Saved {len(rows):,} rows to {out}")
print(f"Spread distribution:")
df = pd.DataFrame(rows)
print(df['spread'].describe().round(2))
