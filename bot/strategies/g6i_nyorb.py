"""NY-open break of the London-morning range WITH a pre-registered filter (g6i_).

Round-4 near miss (gid_orb, M15, long, range 10-15 server, entry 15:30-18:00,
buf 0.3 ATR, SL3/TP8 ATR; trainval t 1.6). Filters were fixed BEFORE any
result was seen (round 6):
  filt "vol"    the breakout bar's time-of-day-normalised tick volume
                (g6i_common.rvol) >= vmin (participation behind the break)
  filt "atrp"   gid_orb regime: 14-day daily ATR as % of price in the top
                half of its trailing 250-day distribution (q_min 0.5)
  filt "both"   both filters
Everything else is delegated to gid_orb (same params accepted).
params: gid_orb params + filt, vmin, vdays
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import FLAT, Signal, flat
from . import g6i_common as G
from . import gid_orb as O

DEFAULTS = {"filt": "vol", "vmin": 1.5, "vdays": 20,
            "range_start_h": 10, "range_end_h": 15, "entry_start_hm": "15:30",
            "entry_end_hm": "18:00", "buf_atr": 0.3, "sides": "long"}


def _cfgs(scfg: StrategyCfg):
    p = {**DEFAULTS, **scfg.params}
    inner = copy.copy(scfg)
    ip = {k: v for k, v in p.items() if k not in ("filt", "vmin", "vdays")}
    if p["filt"] in ("atrp", "both"):
        ip["regime"], ip["q_min"] = "atrp", 0.5
    inner.params = ip
    return p, inner


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p, inner = _cfgs(scfg)
    out = O.add_indicators(df, inner)
    out["rvol"] = G.rvol(df, int(p["vdays"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    _, inner = _cfgs(scfg)
    return O.warmup(inner)


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p, inner = _cfgs(scfg)
    sig = O.evaluate(ind_window, inner)
    if sig.direction == FLAT:
        return sig
    if p["filt"] in ("vol", "both"):
        v = ind_window["rvol"].iat[-1]
        if not np.isfinite(v) or v < float(p["vmin"]):
            return flat("break without volume")
    return sig
