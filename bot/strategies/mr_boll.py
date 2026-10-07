"""Mean reversion: Bollinger-band re-entry fade toward the middle band.

Long:  previous close BELOW the lower band, latest close back ABOVE it
       (the snap-back has already started) and still below the middle band
       (room left to the target) -> buy, aiming at the middle band.
Short: mirror around the upper band.

TP: tp_mode "mid" targets the current middle band (distance = mid - close);
    tp_mode "atr" uses scfg.atr_tp_mult x ATR instead.
SL: scfg.atr_sl_mult x ATR. min_tp_usd floors the target above the spread.

params (scfg.params):
    bb_period  : Bollinger SMA/std window
    bb_std     : band width in standard deviations
    tp_mode    : "mid" or "atr"
    min_tp_usd : skip signals whose TP distance is below this ($/oz)
    both_sides : 1 = fade both bands, 0 = longs only (lower band)
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "tp_mode": "mid",
    "min_tp_usd": 1.5,
    "both_sides": 1,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    n = int(p["bb_period"])
    k = float(p["bb_std"])
    out = df.copy()
    mid = sma(out["close"], n)
    sd = rolling_std(out["close"], n)
    out["mr_mid"] = mid
    out["mr_upper"] = mid + k * sd
    out["mr_lower"] = mid - k * sd
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(int(p["bb_period"]), 5 * scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (pd.notna(prev["mr_lower"]) and pd.notna(last["mr_lower"])):
        return flat("bands not formed")

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    min_tp = float(p["min_tp_usd"])
    close = float(last["close"])
    mid = float(last["mr_mid"])

    long_setup = (
        float(prev["close"]) < float(prev["mr_lower"])
        and close > float(last["mr_lower"])
        and close < mid
    )
    short_setup = (
        int(p["both_sides"])
        and float(prev["close"]) > float(prev["mr_upper"])
        and close < float(last["mr_upper"])
        and close > mid
    )

    if long_setup:
        tp_dist = (mid - close) if p["tp_mode"] == "mid" else scfg.atr_tp_mult * a
        if tp_dist < min_tp:
            return flat("TP below spread floor")
        return Signal(LONG, close, sl_dist, tp_dist, a,
                      f"BB({int(p['bb_period'])},{p['bb_std']}) lower-band re-entry")
    if short_setup:
        tp_dist = (close - mid) if p["tp_mode"] == "mid" else scfg.atr_tp_mult * a
        if tp_dist < min_tp:
            return flat("TP below spread floor")
        return Signal(SHORT, close, sl_dist, tp_dist, a,
                      f"BB({int(p['bb_period'])},{p['bb_std']}) upper-band re-entry")
    return flat("no band re-entry")
