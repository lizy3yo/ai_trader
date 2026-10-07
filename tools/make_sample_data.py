"""Generate synthetic XAUUSD-like M15 bars for offline backtest testing.

Not real market data — just a plausible random-walk with trends, sessions and
weekend gaps, so the full pipeline (strategy -> risk -> backtest -> journal)
can be exercised without an MT5 terminal.

Usage: python tools/make_sample_data.py [out_csv] [n_bars] [seed]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(out_csv: str = "data/sample_xauusd_m15.csv", n: int = 20000, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    # M15 timestamps, weekdays only
    idx = pd.date_range("2025-06-02", periods=int(n * 1.5), freq="15min")
    idx = idx[idx.weekday < 5][:n]

    price = 2650.0
    # regime-switching drift so there are real trends to catch
    drift = 0.0
    opens, highs, lows, closes = [], [], [], []
    for _ in range(len(idx)):
        if rng.random() < 0.004:                      # switch regime ~ every 250 bars
            drift = rng.normal(0.0, 0.35)
        vol = abs(rng.normal(1.6, 0.6)) + 0.4         # $ per bar, gold-ish for M15
        o = price
        pull = -0.0004 * (price - 2650.0)             # weak anchor near $2650
        c = o + drift + pull + rng.normal(0.0, vol)
        h = max(o, c) + abs(rng.normal(0.0, vol * 0.5))
        l = min(o, c) - abs(rng.normal(0.0, vol * 0.5))
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        price = c
        price = max(price, 500.0)                     # floor, just in case

    df = pd.DataFrame(
        {"time": idx, "open": opens, "high": highs, "low": lows, "close": closes}
    ).round(2)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} bars -> {out_csv} "
          f"({df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f})")
    return out_csv


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_xauusd_m15.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    make(out, n, seed)
