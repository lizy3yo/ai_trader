"""Donchian-channel breakout continuation (advanced trend-following family).

Turtle-style: go LONG on the first close above the highest high of the prior
`lookback` bars (channel already excludes the current bar via shift(1) in the
shared donchian() helper), provided the close also sits above EMA(trend).
Short is the mirror through the lowest low. The first-cross condition
(previous close was still inside the channel) prevents re-firing on every
bar of a run.

Optional gates (all default OFF, all causal):
  long_only : 1 = never short
  min_atr   : absolute $ ATR floor — skip dead-volatility regimes

Shared knobs used from StrategyCfg: ema_trend, atr_period, atr_sl_mult,
atr_tp_mult.
params: lookback, long_only, min_atr.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr

DEFAULTS = {
    "lookback": 48,
    "long_only": 0,
    "min_atr": 0.0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    hh, ll = donchian(out, int(p["lookback"]))
    out["dc_hi"] = hh
    out["dc_lo"] = ll
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(5 * scfg.ema_trend, int(p["lookback"]) + 2, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if pd.isna(last["dc_hi"]) or pd.isna(prev["dc_hi"]):
        return flat("channel not formed")

    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat(f"ATR {a:.2f} below floor {p['min_atr']}")
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    lb = int(p["lookback"])

    broke_up = prev["close"] <= prev["dc_hi"] and last["close"] > last["dc_hi"]
    broke_dn = prev["close"] >= prev["dc_lo"] and last["close"] < last["dc_lo"]

    if broke_up and last["close"] > last["ema_trend"]:
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"close broke {lb}-bar high, above EMA{scfg.ema_trend}",
        )
    if broke_dn and last["close"] < last["ema_trend"] and not int(p["long_only"]):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"close broke {lb}-bar low, below EMA{scfg.ema_trend}",
        )
    return flat("no channel breakout")
