"""Displacement / momentum-continuation entries (fast_ family).

Idea: a single bar that travels far (range >= disp_atr x the PRIOR ATR), has
a dominant body and closes near its extreme is a footprint of real order
flow. Enter the NEXT bar's open in the same direction, aligned with a longer
EMA trend, and hold for a target big enough to clear the spread (3-5 x ATR).

Everything is judged on CLOSED bars only: the displacement bar is complete
when we evaluate it, the reference ATR is .shift(1)-ed so the bar is never
compared against a statistic that already contains it, and the entry happens
on the following bar's open (the backtester's standard fill).

Params (scfg.params):
    disp_atr      minimum bar range as a multiple of the PRIOR-bar ATR
    max_disp_atr  maximum bar range (skip news candles you cannot stop safely)
    body_frac     minimum |close-open| / range
    close_pos     close must sit in the top (long) / bottom (short) fraction
                  of the bar's range, e.g. 0.7 = top 30%
    confirm_bars  1 = the displacement bar alone; 2 = the prior bar must close
                  in the same direction too
    trend_mode    "none" | "ema" | "slope" | "ema_slope" (EMA(ema_trend))
    slope_len     bars used by the slope test
    dir           "both" / "long" / "short"
    min_atr       absolute $ ATR floor

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (current ATR, which already
includes the displacement bar, so stops widen with the volatility that made
the signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "disp_atr": 1.5,
    "max_disp_atr": 99.0,
    "body_frac": 0.5,
    "close_pos": 0.66,
    "confirm_bars": 1,
    "trend_mode": "ema",
    "slope_len": 20,
    "dir": "both",
    "min_atr": 0.0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["atr_ref"] = out["atr"].shift(1)      # never compare a bar to itself
    if str(p["trend_mode"]) != "none":
        e = ema(out["close"], scfg.ema_trend)
        out["ema_trend"] = e
        out["ema_prev"] = e.shift(int(p["slope_len"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need = max(scfg.atr_period + 2, int(p["confirm_bars"]) + 2)
    if str(p["trend_mode"]) != "none":
        need = max(need, 5 * scfg.ema_trend + int(p["slope_len"]))
    return need + 10


def _trend_ok(p: dict, direction: int, last) -> bool:
    mode = str(p["trend_mode"])
    if mode == "none":
        return True
    e = float(last["ema_trend"])
    if not np.isfinite(e):
        return False
    if mode in ("ema", "ema_slope"):
        if direction == LONG and not float(last["close"]) > e:
            return False
        if direction == SHORT and not float(last["close"]) < e:
            return False
    if mode in ("slope", "ema_slope"):
        ep = float(last["ema_prev"])
        if not np.isfinite(ep):
            return False
        if direction == LONG and not e > ep:
            return False
        if direction == SHORT and not e < ep:
            return False
    return True


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    a = float(last["atr"])
    ref = float(last["atr_ref"])
    if not np.isfinite(ref) or ref <= 0:
        return flat("no reference ATR")
    if a < float(p["min_atr"]):
        return flat("ATR below floor")

    hi, lo = float(last["high"]), float(last["low"])
    o, c = float(last["open"]), float(last["close"])
    rng = hi - lo
    if rng <= 0:
        return flat("zero-range bar")
    if rng < float(p["disp_atr"]) * ref:
        return flat("no displacement")
    if rng > float(p["max_disp_atr"]) * ref:
        return flat("displacement too violent")
    if abs(c - o) < float(p["body_frac"]) * rng:
        return flat("body too small")

    pos = (c - lo) / rng            # 1.0 = closed on the high
    thr = float(p["close_pos"])
    up = (c > o) and pos >= thr
    dn = (c < o) and pos <= (1.0 - thr)

    nb = int(p["confirm_bars"])
    if nb > 1 and len(ind_window) > nb:
        for k in range(2, nb + 1):
            b = ind_window.iloc[-k]
            if up and not float(b["close"]) > float(b["open"]):
                up = False
            if dn and not float(b["close"]) < float(b["open"]):
                dn = False

    d = str(p["dir"])
    if d == "long":
        dn = False
    elif d == "short":
        up = False
    if not (up or dn):
        return flat("no directional thrust")

    direction = LONG if up else SHORT
    if not _trend_ok(p, direction, last):
        return flat("against trend filter")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    side = "up" if up else "down"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"thrust {side} {rng / ref:.1f}xATR close@{pos:.2f} "
                  f"(tp {tp_dist:.2f})")
