"""Mean reversion: immediate fade of a z-score extreme vs an SMA.

z = (close - SMA(z_period)) / rolling_std(z_period).
Long when z <= -z_enter, short when z >= +z_enter (no confirmation bar --
this is the aggressive cousin of the Bollinger re-entry strategy, which
waits for the snap-back to start; comparing the two tests whether the
confirmation is worth its delay).

Optional trend filter: trend_ema > 0 only fades toward the EMA trend side
(longs only above it, shorts only below it).

SL/TP from shared StrategyCfg ATR multipliers; min_tp_usd floors the
target above the ~$0.55 spread.

params (scfg.params):
    z_period   : SMA / std window
    z_enter    : |z| that triggers a fade
    trend_ema  : 0 = off; else EMA period, fade only with the trend
    min_tp_usd : skip signals whose TP distance is below this ($/oz)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "z_period": 20,
    "z_enter": 2.0,
    "trend_ema": 0,
    "min_tp_usd": 1.5,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    n = int(p["z_period"])
    out = df.copy()
    mid = sma(out["close"], n)
    sd = rolling_std(out["close"], n)
    out["mr_z"] = (out["close"] - mid) / sd.replace(0.0, np.nan)
    if int(p["trend_ema"]) > 0:
        out["mr_trend"] = ema(out["close"], int(p["trend_ema"]))
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need = max(int(p["z_period"]), 5 * scfg.atr_period)
    if int(p["trend_ema"]) > 0:
        need = max(need, 5 * int(p["trend_ema"]))
    return need + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    z = last["mr_z"]
    if pd.isna(z):
        return flat("z not formed")
    z = float(z)

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    if tp_dist < float(p["min_tp_usd"]):
        return flat("TP below spread floor")

    close = float(last["close"])
    z_enter = float(p["z_enter"])
    use_trend = int(p["trend_ema"]) > 0
    trend_ok_long = (not use_trend) or close > float(last["mr_trend"])
    trend_ok_short = (not use_trend) or close < float(last["mr_trend"])

    if z <= -z_enter and trend_ok_long:
        return Signal(LONG, close, sl_dist, tp_dist, a,
                      f"z {z:+.2f} <= -{z_enter} fade up")
    if z >= z_enter and trend_ok_short:
        return Signal(SHORT, close, sl_dist, tp_dist, a,
                      f"z {z:+.2f} >= +{z_enter} fade down")
    return flat("no z extreme")
