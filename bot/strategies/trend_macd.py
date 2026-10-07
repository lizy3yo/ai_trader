"""MACD-signal trend entries with EMA(trend) filter (advanced trend family).

Long: MACD line crosses above its signal line while close > EMA(trend).
With dip_only=1 the cross must happen BELOW the zero line — i.e. we are
buying the end of a pullback inside an uptrend (crosses above zero are
late-chase entries and are skipped). Short is the mirror.

Shared knobs used from StrategyCfg: ema_trend, atr_period, atr_sl_mult,
atr_tp_mult.
params: macd_fast, macd_slow, macd_signal, dip_only (0/1).
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "dip_only": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    macd = ema(out["close"], int(p["macd_fast"])) - ema(out["close"], int(p["macd_slow"]))
    out["macd"] = macd
    out["macd_sig"] = ema(macd, int(p["macd_signal"]))
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    span = max(scfg.ema_trend, int(p["macd_slow"]) + int(p["macd_signal"]))
    return 5 * span + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    dip_only = bool(int(p["dip_only"]))

    crossed_up = prev["macd"] <= prev["macd_sig"] and last["macd"] > last["macd_sig"]
    crossed_dn = prev["macd"] >= prev["macd_sig"] and last["macd"] < last["macd_sig"]

    if (
        crossed_up
        and last["close"] > last["ema_trend"]
        and (not dip_only or last["macd"] < 0)
    ):
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"MACD({p['macd_fast']},{p['macd_slow']},{p['macd_signal']}) up-cross"
            f"{' below zero' if dip_only else ''}, above EMA{scfg.ema_trend}",
        )
    if (
        crossed_dn
        and last["close"] < last["ema_trend"]
        and (not dip_only or last["macd"] > 0)
    ):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"MACD({p['macd_fast']},{p['macd_slow']},{p['macd_signal']}) down-cross"
            f"{' above zero' if dip_only else ''}, below EMA{scfg.ema_trend}",
        )
    return flat("no MACD cross")
