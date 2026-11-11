"""Shared helpers for the g6i_ library (round 6, gold intraday). Not a strategy.

Convention (same as ls_): add_indicators() builds a causal signal column
"ls_sig" (+1 / -1 / 0) decided on each bar's CLOSE plus "atr"; evaluate()
reads the last row and returns ATR stops, so the factory's random baseline
gets identical stop geometry. Optional exit columns ls_xl / ls_xs and a bar
cap `hold` drive exit_signal (study must set "exit_from_strategy": true).

Tick volume: MT5 tick_volume has a strong intraday profile (Asia quiet, US
open loud) and grew ~5x over 2019-2026, so raw thresholds are meaningless.
rvol = volume / mean volume of the SAME time-of-day slot over the previous
`days` sessions (current bar excluded) -> causal, ~1.0 on a normal bar.

Event times: US releases at 08:30 ET (NFP, CPI) and FOMC statements at
14:00 ET converted to server time with zoneinfo (server = EET/EEST,
Europe/Athens rules). Normally 15:30 / 21:00 server; 14:30 / 20:00 in the
DST-mismatch weeks (US on summer time, EU not yet / no longer). Checked on
the data: tick-volume spikes sit on the computed bar (e.g. NFP 2022-11-04
and 2023-11-03 at 14:30, FOMC 2024-03-20 at 20:00).
"""
from __future__ import annotations

from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .lm_calendar_data import FOMC_DATES, NFP_DATES
from .g6i_cpi_data import CPI_DATES

SERVER_TZ = ZoneInfo("Europe/Athens")
NY_TZ = ZoneInfo("America/New_York")


def rvol(df: pd.DataFrame, days: int = 20) -> np.ndarray:
    """Time-of-day-normalised tick volume (causal). NaN if no volume column."""
    if "tick_volume" not in df:
        return np.full(len(df), np.nan)
    v = df["tick_volume"].astype(float)
    slot = df.index.hour * 60 + df.index.minute
    base = v.groupby(slot).transform(
        lambda s: s.shift(1).rolling(days, min_periods=max(days // 2, 3)).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (v / base).to_numpy()
    r[~np.isfinite(r)] = np.nan
    return r


@lru_cache(maxsize=None)
def event_times(kind: str, et_hm: str) -> pd.DatetimeIndex:
    """Server-time (naive) release moments for an event list."""
    lists = {"nfp": NFP_DATES, "cpi": CPI_DATES, "fomc": FOMC_DATES}
    out = []
    for d in lists[kind]:
        t = pd.Timestamp(f"{d} {et_hm}").tz_localize(NY_TZ).tz_convert(SERVER_TZ)
        out.append(t.tz_localize(None))
    return pd.DatetimeIndex(sorted(out))


def events(which: str) -> pd.DatetimeIndex:
    """'nfp' | 'cpi' | 'fomc' | 'data' (nfp+cpi) | 'all' -> server release times."""
    parts = {"nfp": ["nfp"], "cpi": ["cpi"], "fomc": ["fomc"], "data": ["nfp", "cpi"],
             "all": ["nfp", "cpi", "fomc"]}[which]
    idx = []
    for k in parts:
        idx.extend(event_times(k, "14:00" if k == "fomc" else "08:30"))
    return pd.DatetimeIndex(sorted(set(idx)))


def time_exit_cols(out: pd.DataFrame, step_min: int, exit_hm: str) -> None:
    """ls_xl/ls_xs = 1 on the first bar whose CLOSE is at/after exit_hm (server)."""
    h, m = (int(x) for x in exit_hm.split(":"))
    close_t = out.index + pd.Timedelta(minutes=step_min)
    tx = close_t.normalize() + pd.Timedelta(minutes=h * 60 + m)
    tx = tx.where(tx <= close_t, tx - pd.Timedelta(days=1))
    prev_close = pd.Series(close_t).shift(1).to_numpy()
    crossed = (tx.to_numpy() > prev_close).astype(float)
    out["ls_xl"] = crossed
    out["ls_xs"] = crossed


def hold_bars(minutes: int, step_min: int) -> int:
    """Holding cap in bars for a cap given in minutes (0 = off)."""
    return 0 if int(minutes) <= 0 else max(1, int(round(int(minutes) / step_min)))
