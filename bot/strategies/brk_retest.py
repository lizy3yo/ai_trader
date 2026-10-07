"""Donchian breakout + retest-hold confirmation (breakouts family, brk_).

Plain breakout entries buy an extended close and bleed to mean reversion
(measured: PF 0.55-0.8 on this data). This variant waits for the breakout,
then enters only when price PULLS BACK to the broken channel edge and holds:

Long setup:
  1. fresh breakout: close crosses above the prior `entry_period`-bar high
     (toolbox donchian() is shift(1)-causal)
  2. within the next `wait_bars` bars, a bar's LOW dips to within
     `pb_atr` x ATR of the broken level...
  3. ...and that same bar CLOSES back above the level -> enter next bar.
Short is the mirror.

Knobs in scfg.params:
    entry_period  Donchian lookback in bars
    wait_bars     how long a broken level stays valid for a retest entry
    pb_atr        how close the pullback must come to the level (ATRs)
    direction     "both" / "long" / "short"

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config fields).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, flat, valid_atr

DEFAULTS = {
    "entry_period": 40,
    "wait_bars": 12,
    "pb_atr": 0.25,
    "direction": "both",
}


def _params(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _params(scfg)
    out = df.copy()
    hh, ll = donchian(out, int(p["entry_period"]))   # shift(1)-causal
    close = out["close"]
    fresh_up = (close > hh) & (close.shift(1) <= hh.shift(1))
    fresh_dn = (close < ll) & (close.shift(1) >= ll.shift(1))

    idx = np.arange(len(out), dtype=float)
    up_i = pd.Series(np.where(fresh_up, idx, np.nan), index=out.index).ffill()
    dn_i = pd.Series(np.where(fresh_dn, idx, np.nan), index=out.index).ffill()
    out["up_level"] = hh.where(fresh_up).ffill()   # the broken high
    out["dn_level"] = ll.where(fresh_dn).ffill()   # the broken low
    out["up_age"] = idx - up_i                     # bars since that breakout
    out["dn_age"] = idx - dn_i
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _params(scfg)
    return max(int(p["entry_period"]) + int(p["wait_bars"]) + 2, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _params(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    wait = int(p["wait_bars"])
    zone = float(p["pb_atr"]) * a

    go_long = go_short = False
    if p["direction"] in ("both", "long") and np.isfinite(last["up_level"]):
        lvl = float(last["up_level"])
        if (1 <= last["up_age"] <= wait
                and float(last["low"]) <= lvl + zone
                and float(last["close"]) > lvl):
            go_long = True
    if p["direction"] in ("both", "short") and np.isfinite(last["dn_level"]):
        lvl = float(last["dn_level"])
        if (1 <= last["dn_age"] <= wait
                and float(last["high"]) >= lvl - zone
                and float(last["close"]) < lvl):
            go_short = True
    if go_long and go_short:
        return flat("conflicting retests")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    if go_long:
        return Signal(LONG, float(last["close"]), sl_dist, tp_dist, a,
                      f"retest hold of DC{int(p['entry_period'])} break up")
    if go_short:
        return Signal(SHORT, float(last["close"]), sl_dist, tp_dist, a,
                      f"retest hold of DC{int(p['entry_period'])} break down")
    return flat("no retest setup")
