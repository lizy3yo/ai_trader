"""Session-scanned z-score mean reversion (session/time-structure family).

Idea: fade a stretch away from the rolling mean — enter when the close
CROSSES INTO the extreme zone (prev inside, last outside). The strategy has
no internal clock: WHICH session it runs in is set from the grid via the
"session.window" key, making the trading window itself the experiment
(Asia-only vs London-only vs NY-only vs near-24h).

Params (scfg.params):
    bb_period : rolling mean/std lookback (bars)
    z_in      : z-score threshold that defines the extreme zone

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "bb_period": 32,
    "z_in": 2.0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    period = int(p["bb_period"])
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    mid = sma(out["close"], period)
    sd = rolling_std(out["close"], period)
    out["z"] = (out["close"] - mid) / sd.replace(0.0, np.nan)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    return max(int(p["bb_period"]), 5 * scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    z_last, z_prev = float(last["z"]), float(prev["z"])
    if not (np.isfinite(z_last) and np.isfinite(z_prev)):
        return flat("z not ready")

    z_in = float(p["z_in"])
    sl_dist = scfg.atr_sl_mult * float(last["atr"])
    tp_dist = scfg.atr_tp_mult * float(last["atr"])

    if z_prev >= -z_in and z_last < -z_in:
        return Signal(LONG, float(last["close"]), sl_dist, tp_dist, float(last["atr"]),
                      f"mean-rev long: z {z_last:.2f} crossed below -{z_in}")
    if z_prev <= z_in and z_last > z_in:
        return Signal(SHORT, float(last["close"]), sl_dist, tp_dist, float(last["atr"]),
                      f"mean-rev short: z {z_last:.2f} crossed above +{z_in}")
    return flat("no z extreme")
