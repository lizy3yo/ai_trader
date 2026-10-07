"""Bollinger-squeeze breakout (breakouts family, prefix brk_).

Setup:  band-width contraction — the PREVIOUS bar's normalized Bollinger
        band width must sit in the lowest `squeeze_pct` quantile of its own
        prior `squeeze_lookback` values (all shift(1)-ed: the entry bar's own
        expansion never disqualifies/qualifies itself).
Entry:  the newest closed bar CLOSES outside the PREVIOUS bar's band
        (band trigger is shift(1)-ed too, so the bar cannot drag its own
        band along and trigger on it), and the previous close was inside.

Knobs in scfg.params:
    bb_period        Bollinger SMA/std period
    bb_std           band width in standard deviations
    squeeze_lookback bars of band-width history for the contraction quantile
    squeeze_pct      quantile (0..1) the prior band width must be under
    trend_filter     1 -> longs only above EMA(scfg.ema_trend), shorts below
    direction        "both" / "long" / "short"

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config fields).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "squeeze_lookback": 100,
    "squeeze_pct": 0.3,
    "trend_filter": 0,
    "direction": "both",
}


def _params(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _params(scfg)
    out = df.copy()
    period = int(p["bb_period"])
    mid = sma(out["close"], period)
    sd = rolling_std(out["close"], period)
    upper = mid + float(p["bb_std"]) * sd
    lower = mid - float(p["bb_std"]) * sd
    bw = (upper - lower) / mid
    # everything the entry bar reacts to is from PRIOR bars:
    out["bb_up_prev"] = upper.shift(1)
    out["bb_dn_prev"] = lower.shift(1)
    bw_prev = bw.shift(1)
    thresh = bw_prev.rolling(int(p["squeeze_lookback"])).quantile(float(p["squeeze_pct"]))
    out["squeeze"] = (bw_prev <= thresh).astype(float)
    out["atr"] = atr(out, scfg.atr_period)
    if int(p["trend_filter"]):
        out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _params(scfg)
    need = max(int(p["bb_period"]) + int(p["squeeze_lookback"]) + 2, scfg.atr_period)
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
    if not (np.isfinite(last["bb_up_prev"]) and np.isfinite(last["bb_dn_prev"])
            and np.isfinite(prev["bb_up_prev"]) and np.isfinite(prev["bb_dn_prev"])):
        return flat("bands warming up")
    if not last["squeeze"] > 0:
        return flat("no squeeze")

    brk_up = last["close"] > last["bb_up_prev"] and prev["close"] <= prev["bb_up_prev"]
    brk_dn = last["close"] < last["bb_dn_prev"] and prev["close"] >= prev["bb_dn_prev"]

    if int(p["trend_filter"]):
        if brk_up and not last["close"] > last["ema_trend"]:
            brk_up = False
        if brk_dn and not last["close"] < last["ema_trend"]:
            brk_dn = False
    if p["direction"] == "long":
        brk_dn = False
    elif p["direction"] == "short":
        brk_up = False

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    if brk_up:
        return Signal(LONG, float(last["close"]), sl_dist, tp_dist, a,
                      f"squeeze break up (BB{int(p['bb_period'])}/{p['bb_std']})")
    if brk_dn:
        return Signal(SHORT, float(last["close"]), sl_dist, tp_dist, a,
                      f"squeeze break down (BB{int(p['bb_period'])}/{p['bb_std']})")
    return flat("squeeze, no break")
