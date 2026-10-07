"""Keltner / ATR-band breakout with wide targets (fast_ family).

Same objective as fast_channel (many trades, targets that clear the spread)
but the trigger is volatility-relative instead of price-extreme-relative: the
band is EMA(ema_len) +/- band_atr x ATR, so the entry threshold breathes with
volatility instead of being pinned to an old high.

Long:  first close above the upper band. Short: first close below the lower
band. `re_arm` requires the previous close to have been back inside the band,
which stops the signal re-firing every bar of a trend leg.

Params (scfg.params):
    ema_len      band centre EMA length
    band_atr     band half-width in ATRs (the frequency knob: 1.0 fires often,
                 2.5 rarely)
    dir          "both" / "long" / "short"
    trend_mode   "none" | "ema" (close on the right side of EMA(ema_trend))
                 | "slope" (band centre rising/falling over slope_len bars)
                 | "ema_slope"
    slope_len    bars used by the slope test
    re_arm       1 = require the prior close inside the band (fresh cross)
    min_atr      absolute $ ATR floor
    max_ext_atr  anti-chase: skip when the close is already this many ATRs
                 beyond the band

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR.
Causality: ewm/rolling/shift only — no full-frame statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "ema_len": 20,
    "band_atr": 1.5,
    "dir": "both",
    "trend_mode": "none",
    "slope_len": 20,
    "re_arm": 1,
    "min_atr": 0.0,
    "max_ext_atr": 99.0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    mid = ema(out["close"], int(p["ema_len"]))
    out["kc_mid"] = mid
    out["kc_mid_prev"] = mid.shift(int(p["slope_len"]))
    out["kc_up"] = mid + float(p["band_atr"]) * out["atr"]
    out["kc_dn"] = mid - float(p["band_atr"]) * out["atr"]
    if str(p["trend_mode"]) in ("ema", "ema_slope"):
        out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need = max(5 * int(p["ema_len"]) + int(p["slope_len"]), scfg.atr_period)
    if str(p["trend_mode"]) in ("ema", "ema_slope"):
        need = max(need, 5 * scfg.ema_trend)
    return need + 10


def _trend_ok(p: dict, direction: int, last) -> bool:
    mode = str(p["trend_mode"])
    if mode == "none":
        return True
    if mode in ("ema", "ema_slope"):
        e = float(last["ema_trend"])
        if not np.isfinite(e):
            return False
        if direction == LONG and not float(last["close"]) > e:
            return False
        if direction == SHORT and not float(last["close"]) < e:
            return False
    if mode in ("slope", "ema_slope"):
        m, mp = float(last["kc_mid"]), float(last["kc_mid_prev"])
        if not (np.isfinite(m) and np.isfinite(mp)):
            return False
        if direction == LONG and not m > mp:
            return False
        if direction == SHORT and not m < mp:
            return False
    return True


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (np.isfinite(last["kc_up"]) and np.isfinite(prev["kc_up"])):
        return flat("bands warming up")

    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")

    c, cp = float(last["close"]), float(prev["close"])
    up, dn = float(last["kc_up"]), float(last["kc_dn"])
    brk_up = c > up
    brk_dn = c < dn
    if int(p["re_arm"]):
        brk_up = brk_up and cp <= float(prev["kc_up"])
        brk_dn = brk_dn and cp >= float(prev["kc_dn"])

    d = str(p["dir"])
    if d == "long":
        brk_dn = False
    elif d == "short":
        brk_up = False
    if not (brk_up or brk_dn):
        return flat("no band break")

    ext = (c - up) if brk_up else (dn - c)
    if ext > float(p["max_ext_atr"]) * a:
        return flat("break already extended")

    direction = LONG if brk_up else SHORT
    if not _trend_ok(p, direction, last):
        return flat("against trend filter")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    side = "up" if brk_up else "down"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"KC{int(p['ema_len'])}/{float(p['band_atr']):.1f} break {side} "
                  f"(atr {a:.2f}, tp {tp_dist:.2f})")
