"""Shared helpers for the btcsw_* BTCUSD swing modules (not a strategy itself).

Everything is causal. Lookbacks are given in DAYS and converted to bars with
bars_per_day(timeframe) (24/7 crypto: H1 = 24, H4 = 6; pre-2022 broker data is
weekdays only, so a "day" of bars is then a trading day).

Null-hypothesis hook used by the btcsw edge test: when params null_prob > 0
the module replaces its ENTRY trigger with a seeded coin flip (same gates,
same stops, same exit_signal), so the comparison isolates entry timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_BPD = {"M30": 48, "H1": 24, "H4": 6, "D1": 1}


def bars_per_day(tf: str) -> int:
    return _BPD.get(str(tf).upper(), 24)


def day_end_flag(index: pd.DatetimeIndex, tf: str) -> np.ndarray:
    """True on the bar whose CLOSE is server midnight (last bar of the day)."""
    step = pd.Timedelta(hours=24 // bars_per_day(tf)) if bars_per_day(tf) <= 24 else pd.Timedelta(minutes=30)
    close_t = index + step
    return (close_t.hour == 0) & (close_t.minute == 0)


def uniform(index: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """splitmix64 of (minutes since epoch, seed) -> U[0,1), reproducible."""
    with np.errstate(over="ignore"):
        z = (index.asi8 // 60_000_000_000).astype(np.uint64)
        z = z + np.uint64(0x9E3779B97F4A7C15) * np.uint64(int(seed) + 7)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(2**53)
