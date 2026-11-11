"""Daily-level breakouts on intraday bars (ls_ library, volatility family):
Larry-Williams volatility breakout, prior-day high/low breakout, NR4/NR7 and
inside-day breakouts.

The trading day starts at anchor_hm (server time). Levels for today, from the
COMPLETED previous day only:
  level "open_k" : today's first open +/- k x previous day's range (Williams)
  level "prev_hl": previous day's high / low
Filters on the previous day: nr = 4 or 7 (its range is the narrowest of the
last N days), inside = 1 (inside the day before it), and it must have been a
full day (>= 80% of the usual bar count).
Trigger: the first close beyond a level during the day (first_only=1: only
the first breakout of the day, either side) and no later than entry_min
minutes after the anchor.
Exits: ATR stop/target + session.flat_at in the study (day trade) or none
(swing, ATR exits + max_hold_bars).

params: anchor_hm, level, k, nr, inside, entry_min, first_only, side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import ls_common as C

DEFAULTS = {"anchor_hm": "00:00", "level": "open_k", "k": 0.5, "nr": 0, "inside": 0,
            "entry_min": 1440, "first_only": 1, "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = C.tf_minutes(scfg)
    start = C.hm(p["anchor_hm"])
    did = C.day_id(out.index, start)
    st = C.prev_day_stats(out, did)
    rel = (C.open_mod(out.index) - start) % 1440
    if str(p["level"]) == "prev_hl":
        up, dn = st["pd_h"].to_numpy(), st["pd_l"].to_numpy()
    else:
        k = float(p["k"])
        up = st["td_open"].to_numpy() + k * st["pd_rng"].to_numpy()
        dn = st["td_open"].to_numpy() - k * st["pd_rng"].to_numpy()
    ok = (st["pd_full"].to_numpy() == 1) & ((rel + step) <= int(p["entry_min"]))
    if int(p["nr"]) in (4, 7):
        ok &= st[f"pd_nr{int(p['nr'])}"].to_numpy() == 1
    if int(p["inside"]):
        ok &= st["pd_inside"].to_numpy() == 1
    c = out["close"].to_numpy()
    with np.errstate(invalid="ignore"):
        u = ok & (c > up)
        d = ok & (c < dn)
    ev = u | d
    if int(p["first_only"]):
        ev = C.first_per_day(ev, did)
    else:
        ev = (u & ~np.r_[False, u[:-1]]) | (d & ~np.r_[False, d[:-1]])
    out["ls_sig"] = np.where(ev & u, 1.0, np.where(ev & d, -1.0, 0.0))
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"DLVL {p['level']}")
