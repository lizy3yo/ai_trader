"""Micro Bollinger-squeeze breakout on M5 (scalping family, prefix sc_).

Idea: Bollinger band-width contracting to (near) an N-bar low marks a
volatility squeeze; the first close OUTSIDE the band while the squeeze is
in effect is traded as a breakout with tight ATR exits.

Causality: bands at row i use closes i-period+1..i (rolling, causal). The
squeeze test compares the PREVIOUS bar's bandwidth to the rolling minimum of
bandwidth over the prior `squeeze_look` bars, both .shift(1)-ed, so the
current bar never sees itself. The "fresh break" test uses the previous
bar's close vs the previous bar's band. No full-frame statistics.

Params (scfg.params):
    bb_period    : Bollinger period (SMA + population std)
    bb_std      : band width in standard deviations
    squeeze_look : bandwidth must be near the low of this many PRIOR bars
    squeeze_tol  : prev bandwidth <= tol x rolling min (1.0 = exact low)
    sides        : "both" / "long" / "short"
    trend_filter : if 1, longs only above EMA(scfg.ema_trend), shorts below
    min_atr      : skip if ATR below this, in $ (dead tape)
    max_atr      : skip if ATR above this, in $ (news blowout)

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "squeeze_look": 48,
    "squeeze_tol": 1.10,
    "sides": "both",
    "trend_filter": 0,
    "min_atr": 0.0,
    "max_atr": 99.0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    period = int(p["bb_period"])
    mid = sma(out["close"], period)
    sd = rolling_std(out["close"], period)
    out["bb_up"] = mid + float(p["bb_std"]) * sd
    out["bb_dn"] = mid - float(p["bb_std"]) * sd
    bw = (out["bb_up"] - out["bb_dn"]) / mid

    # squeeze on bar i: bar i-1's bandwidth is within tol of the minimum
    # bandwidth over bars i-squeeze_look .. i-1 (all strictly past rows)
    look = int(p["squeeze_look"])
    bw_prev = bw.shift(1)
    bw_min_prior = bw.rolling(look).min().shift(1)
    out["squeeze"] = (bw_prev <= float(p["squeeze_tol"]) * bw_min_prior).astype(float)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    base = int(p["bb_period"]) + int(p["squeeze_look"]) + 5 * scfg.atr_period
    if int(p["trend_filter"]):
        base = max(base, 5 * scfg.ema_trend)
    return base + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")
    if a > float(p["max_atr"]):
        return flat("ATR above cap")
    if not (np.isfinite(last["bb_up"]) and np.isfinite(prev["bb_up"])):
        return flat("bands not ready")
    if float(last["squeeze"]) != 1.0:
        return flat("no squeeze")

    c, c_prev = float(last["close"]), float(prev["close"])
    broke_up = c > float(last["bb_up"]) and c_prev <= float(prev["bb_up"])
    broke_dn = c < float(last["bb_dn"]) and c_prev >= float(prev["bb_dn"])
    if not (broke_up or broke_dn):
        return flat("no band break")

    direction = LONG if broke_up else SHORT
    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    if int(p["trend_filter"]):
        trend = float(last["ema_trend"])
        if direction == LONG and c <= trend:
            return flat("break against trend")
        if direction == SHORT and c >= trend:
            return flat("break against trend")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    side = "long" if direction == LONG else "short"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"squeeze break {side}: close {c:.2f} outside band, ATR {a:.2f}")
