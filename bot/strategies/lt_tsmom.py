"""Time-series momentum (Moskowitz-Ooi-Pedersen) (lt_ trend family).

Whenever flat, go in the direction of the sign of the n-bar return (optionally
skipping the last `skip` bars). The holding period comes from
cfg.max_hold_bars (e.g. 21 D1 bars = monthly rebalance) with a wide
catastrophe ATR stop and no exit_signal, so the random baseline is simply
"random entries held for the same horizon with the same stop".
params: n (lookback bars), skip, side, trend
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from . import lt_common as C

DEFAULTS = {"n": 252, "skip": 0, "side": "long", "trend": 0}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    sk = int(p["skip"])
    ret = c.shift(sk) / c.shift(sk + int(p["n"])) - 1.0
    up, dn = ret > 0, ret < 0
    if int(p["trend"]):
        tu, td = C.trend_ok(out, scfg.ema_trend)
        up, dn = up & tu, dn & td
    return C.finish(out, C.combine(up, dn, p["side"]), scfg.atr_period)


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = int(p["n"]) + int(p["skip"]) + scfg.atr_period + 5
    return max(w, 2 * scfg.ema_trend) if int(p["trend"]) else w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg):
    return C.make_signal(ind_window, scfg, f"{_p(scfg)['n']}-bar momentum sign")
