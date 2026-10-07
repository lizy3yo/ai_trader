"""Asia-hours micro range-fade on M5 (scalping family, prefix sc_).

Idea: in the quiet Asia session gold tends to oscillate inside a narrow
range. Buy closes at/near the low edge of a rolling N-bar range, sell
closes at/near the high edge, targeting the middle of the range (or a tight
ATR target). Which hours are actually traded is set by the harness session
window (e.g. "session.window": "01:30-08:00") — keep it away from the
00:00-01:00 rollover.

Causality: the range high/low are rolling extremes of the PRIOR range_look
bars (.shift(1)), so the current bar's own extreme never defines the edge
it is faded against. No groupby-day, no full-frame statistics.

Params (scfg.params):
    range_look   : rolling bars defining the range (36 = 3h on M5)
    edge_frac    : entry zone depth as a fraction of the range width
    min_w_atr    : skip if range width < this x ATR (no room after spread)
    max_w_atr    : skip if range width > this x ATR (trending night)
    max_poke_atr : skip if close is beyond the edge by more than this x ATR
                   (that is a breakout, not a fade)
    tp_mode      : "atr" (scfg.atr_tp_mult x ATR) or "mid" (range midpoint)
    min_tp       : with tp_mode="mid", skip if the target is closer than
                   this many $ (a sub-spread target is donating money)
    sides        : "both" / "long" / "short"

SL: scfg.atr_sl_mult x ATR (shared config knob).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "range_look": 36,
    "edge_frac": 0.15,
    "min_w_atr": 1.5,
    "max_w_atr": 8.0,
    "max_poke_atr": 0.5,
    "tp_mode": "atr",
    "min_tp": 0.70,
    "sides": "both",
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    look = int(p["range_look"])
    out["rng_hi"] = out["high"].rolling(look).max().shift(1)
    out["rng_lo"] = out["low"].rolling(look).min().shift(1)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    return int(p["range_look"]) + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    hi, lo = last["rng_hi"], last["rng_lo"]
    if not (np.isfinite(hi) and np.isfinite(lo)):
        return flat("range not ready")

    a = float(last["atr"])
    hi, lo = float(hi), float(lo)
    width = hi - lo
    if width < float(p["min_w_atr"]) * a:
        return flat("range too narrow")
    if width > float(p["max_w_atr"]) * a:
        return flat("range too wide")

    c = float(last["close"])
    edge = float(p["edge_frac"]) * width
    poke_cap = float(p["max_poke_atr"]) * a

    direction = 0
    if c <= lo + edge and c >= lo - poke_cap:
        direction = LONG
    elif c >= hi - edge and c <= hi + poke_cap:
        direction = SHORT
    if direction == 0:
        return flat("not at a range edge")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    sl_dist = scfg.atr_sl_mult * a
    if str(p["tp_mode"]) == "mid":
        mid = lo + 0.5 * width
        tp_dist = (mid - c) if direction == LONG else (c - mid)
        if tp_dist < float(p["min_tp"]):
            return flat("mid target too close")
    else:
        tp_dist = scfg.atr_tp_mult * a

    side = "long" if direction == LONG else "short"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"range fade {side}: close {c:.2f} at edge of [{lo:.2f}, {hi:.2f}]")
