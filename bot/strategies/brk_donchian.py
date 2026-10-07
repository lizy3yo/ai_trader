"""Donchian channel breakout (breakouts family, prefix brk_).

Long:  a FRESH breakout above the highest high of the prior `entry_period`
       bars (toolbox donchian() is already shift(1)-causal, so the current
       bar's own extreme never triggers its own breakout).
Short: mirror against the lowest low.

Knobs in scfg.params:
    entry_period   Donchian lookback in bars (e.g. 20 / 40 / 55)
    confirm        "close" -> the CLOSE must be beyond the channel
                   "wick"  -> a high/low pierce is enough
    min_break_atr  extra margin beyond the channel, in ATR multiples
    trend_filter   1 -> longs only above EMA(scfg.ema_trend), shorts below
    direction      "both" / "long" / "short" (gold's secular bull makes
                   long-only a legitimate structural variant)

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config fields).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr

DEFAULTS = {
    "entry_period": 40,
    "confirm": "close",
    "min_break_atr": 0.0,
    "trend_filter": 0,
    "direction": "both",
}


def _params(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _params(scfg)
    out = df.copy()
    hh, ll = donchian(out, int(p["entry_period"]))   # shift(1)-causal
    out["dc_hh"] = hh
    out["dc_ll"] = ll
    out["atr"] = atr(out, scfg.atr_period)
    if int(p["trend_filter"]):
        out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _params(scfg)
    need = max(int(p["entry_period"]) + 2, scfg.atr_period)
    if int(p["trend_filter"]):
        need = max(need, 5 * scfg.ema_trend)
    return need + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _params(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (np.isfinite(last["dc_hh"]) and np.isfinite(last["dc_ll"])
            and np.isfinite(prev["dc_hh"]) and np.isfinite(prev["dc_ll"])):
        return flat("channel warming up")

    a = float(last["atr"])
    margin = float(p["min_break_atr"]) * a
    if p["confirm"] == "wick":
        brk_up = last["high"] > last["dc_hh"] + margin and prev["high"] <= prev["dc_hh"]
        brk_dn = last["low"] < last["dc_ll"] - margin and prev["low"] >= prev["dc_ll"]
    else:  # "close": the bar must CLOSE beyond the channel (filters pure wicks)
        brk_up = last["close"] > last["dc_hh"] + margin and prev["close"] <= prev["dc_hh"]
        brk_dn = last["close"] < last["dc_ll"] - margin and prev["close"] >= prev["dc_ll"]

    if int(p["trend_filter"]):
        if brk_up and not last["close"] > last["ema_trend"]:
            brk_up = False
        if brk_dn and not last["close"] < last["ema_trend"]:
            brk_dn = False
    if p["direction"] == "long":
        brk_dn = False
    elif p["direction"] == "short":
        brk_up = False

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    if brk_up:
        return Signal(LONG, float(last["close"]), sl_dist, tp_dist, a,
                      f"DC{int(p['entry_period'])} break up ({p['confirm']})")
    if brk_dn:
        return Signal(SHORT, float(last["close"]), sl_dist, tp_dist, a,
                      f"DC{int(p['entry_period'])} break down ({p['confirm']})")
    return flat("no breakout")
