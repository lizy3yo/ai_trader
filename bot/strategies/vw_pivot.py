"""Classic floor-trader pivot points (prefix vw_).

From the PRIOR COMPLETED day's high/low/close:

    P  = (H + L + C) / 3
    R1 = 2P - L        S1 = 2P - H
    R2 = P + (H - L)   S2 = P - (H - L)
    R3 = H + 2(P - L)  S3 = L - 2(H - P)

Causality: the day aggregates are built with groupby(day).max/min/last and
then SHIFTED one day, so a bar only ever sees days that finished before its
own day started. Truncating the frame anywhere cannot change a published
level (the same day-shift pattern ses_orb.py uses for the Asia range).

Modes
    bounce : price trades into a level and closes back off it (fade the level)
    break  : price closes through a level plus a buffer (trade the breakout)

Params (scfg.params)
    mode         : "bounce" | "break"
    levels       : which levels are live -- "s1r1" | "pp" | "s2r2" |
                   "s1r1s2r2" | "all" (all = P,S1,R1,S2,R2,S3,R3)
    touch_atr    : bounce -- how close the bar's extreme must come to the
                   level, in ATRs (>=0 means it may also pierce it)
    buf_atr      : break -- close must clear the level by this many ATRs
    day_anchor_h : server hour the pivot "day" rolls over at (0 = broker day)
    bias         : "none" | "above_p" (longs only above P, shorts only below)
    min_range_atr / max_range_atr : skip days whose prior-day range is
                   outside this many ATRs
    tp_mode      : "atr" (scfg.atr_tp_mult x ATR) or "next_level" (target the
                   next pivot in the trade's direction, clipped into
                   [tp_min_atr, tp_max_atr] x ATR)
    sides        : "both" | "long" | "short"

SL = scfg.atr_sl_mult x ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "mode": "bounce",
    "levels": "s1r1",
    "touch_atr": 0.15,
    "buf_atr": 0.10,
    "day_anchor_h": 0,
    "bias": "none",
    "min_range_atr": 0.0,
    "max_range_atr": 99.0,
    "tp_mode": "atr",
    "tp_min_atr": 0.5,
    "tp_max_atr": 6.0,
    "sides": "both",
}

_LEVEL_SETS = {
    "s1r1": ("S1", "R1"),
    "pp": ("P",),
    "s2r2": ("S2", "R2"),
    "s1r1s2r2": ("S1", "R1", "S2", "R2"),
    "all": ("P", "S1", "R1", "S2", "R2", "S3", "R3"),
}

# every level, ordered low -> high, for the "next level" target
_ORDER = ("S3", "S2", "S1", "P", "R1", "R2", "R3")


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 15)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)

    anchor = int(p["day_anchor_h"])
    dayid = pd.Series((out.index - pd.Timedelta(hours=anchor)).normalize(),
                      index=out.index)

    daily = pd.DataFrame({
        "h": out["high"].groupby(dayid).max(),
        "l": out["low"].groupby(dayid).min(),
        "c": out["close"].groupby(dayid).last(),
    })
    prior = daily.shift(1)          # <- only completed days are ever read

    ph = dayid.map(prior["h"])
    pl = dayid.map(prior["l"])
    pc = dayid.map(prior["c"])

    pp = (ph + pl + pc) / 3.0
    rng = ph - pl
    out["pv_range"] = rng
    out["P"] = pp
    out["R1"] = 2.0 * pp - pl
    out["S1"] = 2.0 * pp - ph
    out["R2"] = pp + rng
    out["S2"] = pp - rng
    out["R3"] = ph + 2.0 * (pp - pl)
    out["S3"] = pl - 2.0 * (ph - pp)
    return out


def warmup(scfg: StrategyCfg) -> int:
    bars_per_day = 1440 // _bar_minutes(scfg)
    return 2 * bars_per_day + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg) + 2:
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    rng = float(last["pv_range"])
    if not np.isfinite(rng) or rng <= 0:
        return flat("no prior day")
    if rng < float(p["min_range_atr"]) * a or rng > float(p["max_range_atr"]) * a:
        return flat("prior-day range filtered out")

    prices = {}
    for name in _ORDER:
        v = float(last[name])
        if np.isfinite(v):
            prices[name] = v
    if "P" not in prices:
        return flat("no pivots")

    live = _LEVEL_SETS.get(str(p["levels"]))
    if live is None:
        return flat(f"unknown level set {p['levels']}")

    c_last = float(last["close"])
    c_prev = float(prev["close"])
    hi, lo = float(last["high"]), float(last["low"])
    pp = prices["P"]
    mode = str(p["mode"])

    direction = 0
    desc = ""
    best_gap = float("inf")

    if mode == "bounce":
        tol = float(p["touch_atr"]) * a
        for name in live:
            lvl = prices.get(name)
            if lvl is None:
                continue
            # support bounce: dipped to/through the level, closed back above it
            if lo <= lvl + tol and c_last > lvl and c_last > c_prev:
                gap = abs(c_last - lvl)
                if gap < best_gap:
                    best_gap, direction, desc = gap, LONG, f"bounce off {name} {lvl:.2f}"
            # resistance bounce: poked it, closed back below
            if hi >= lvl - tol and c_last < lvl and c_last < c_prev:
                gap = abs(c_last - lvl)
                if gap < best_gap:
                    best_gap, direction, desc = gap, SHORT, f"rejected at {name} {lvl:.2f}"
    elif mode == "break":
        buf = float(p["buf_atr"]) * a
        for name in live:
            lvl = prices.get(name)
            if lvl is None:
                continue
            if c_prev <= lvl + buf < c_last:
                gap = abs(c_last - lvl)
                if gap < best_gap:
                    best_gap, direction, desc = gap, LONG, f"broke up through {name} {lvl:.2f}"
            if c_prev >= lvl - buf > c_last:
                gap = abs(c_last - lvl)
                if gap < best_gap:
                    best_gap, direction, desc = gap, SHORT, f"broke down through {name} {lvl:.2f}"
    else:
        return flat(f"unknown mode {mode}")

    if direction == 0:
        return flat("no pivot signal")

    if str(p["bias"]) == "above_p":
        if direction == LONG and c_last <= pp:
            return flat("long below P")
        if direction == SHORT and c_last >= pp:
            return flat("short above P")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    sl_dist = scfg.atr_sl_mult * a
    if str(p["tp_mode"]) == "next_level":
        ladder = sorted(prices.values())
        if direction == LONG:
            above = [v for v in ladder if v > c_last]
            tp_dist = (above[0] - c_last) if above else scfg.atr_tp_mult * a
        else:
            below = [v for v in ladder if v < c_last]
            tp_dist = (c_last - below[-1]) if below else scfg.atr_tp_mult * a
        tp_dist = min(max(tp_dist, float(p["tp_min_atr"]) * a), float(p["tp_max_atr"]) * a)
    else:
        tp_dist = scfg.atr_tp_mult * a
    if tp_dist <= 0 or sl_dist <= 0:
        return flat("degenerate SL/TP")

    side = "long" if direction == LONG else "short"
    return Signal(direction, c_last, sl_dist, tp_dist, a,
                  f"PIVOT {mode} {side}: {desc} (close {c_last:.2f}, P {pp:.2f})")
