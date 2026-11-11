"""Opening-range breakout at a market's own open (ls_ library, session family).

The range is the high/low of the bars inside [open_hm, open_hm + or_min)
(server time; the day is anchored at open_hm). After the range is complete,
the FIRST close beyond it (+ buf_atr x ATR) up to open_hm + entry_min minutes
gives the trade in the breakout direction (first_only=1: only the first
breakout of the day, either side). Days with an incomplete range are skipped.

Exits: ATR stop/target from StrategyCfg plus session.flat_at in the study
(intraday, no swap). Set session.window to the entry window in the study so
the random baseline is time-matched.

params: open_hm, or_min, entry_min, buf_atr, first_only, max_rng_atr (0=off;
skip days whose range exceeds this x ATR), side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import ls_common as C

DEFAULTS = {"open_hm": "16:30", "or_min": 30, "entry_min": 180, "buf_atr": 0.0,
            "first_only": 1, "max_rng_atr": 0.0, "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = C.tf_minutes(scfg)
    start = C.hm(p["open_hm"])
    or_min, entry_min = int(p["or_min"]), int(p["entry_min"])
    did = C.day_id(out.index, start)
    rel = (C.open_mod(out.index) - start) % 1440
    in_rng = (rel + step) <= or_min
    after = (rel >= or_min) & ((rel + step) <= entry_min)

    h = out["high"].where(in_rng)
    l = out["low"].where(in_rng)
    rhi = h.groupby(did).transform("max").to_numpy()
    rlo = l.groupby(did).transform("min").to_numpy()
    cnt = pd.Series(in_rng.astype(int), index=out.index).groupby(did).transform("sum").to_numpy()
    full = cnt >= max(1, or_min // step)

    c = out["close"].to_numpy()
    a = out["atr"].to_numpy()
    buf = float(p["buf_atr"]) * a
    with np.errstate(invalid="ignore"):
        ok = after & full & np.isfinite(rhi) & np.isfinite(a)
        if float(p["max_rng_atr"]) > 0:
            ok &= (rhi - rlo) <= float(p["max_rng_atr"]) * a
        up = ok & (c > rhi + buf)
        dn = ok & (c < rlo - buf)
    ev = up | dn
    if int(p["first_only"]):
        ev = C.first_per_day(ev, did)
    else:
        prev_up = np.r_[False, up[:-1]]
        prev_dn = np.r_[False, dn[:-1]]
        ev = (up & ~prev_up) | (dn & ~prev_dn)
    sig = np.where(ev & up, 1, np.where(ev & dn, -1, 0))
    out["ls_sig"] = sig.astype(float)
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"ORB {p['open_hm']}/{p['or_min']}m")
