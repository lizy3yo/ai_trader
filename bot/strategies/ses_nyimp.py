"""NY-open impulse momentum (session/time-structure family, prefix ses_).

Idea: the London/NY overlap ramp starts ~15:00 server time (the measured
volatility peak is 15:00-18:00). Measure the direction of the 15:00-15:59
impulse (open of the hour's first bar -> close of its last bar, all closed
bars, so it is fully known at 16:00) and enter WITH it (mode="follow") or
AGAINST it (mode="fade") right after the impulse hour completes.

Params (scfg.params):
    imp_h       : impulse hour on the server clock (default 15)
    imp_min_atr : |impulse| must be at least this many ATRs to count
    imp_max_atr : |impulse| cap (skip blow-off hours), default off (99)
    mode        : "follow" or "fade"
    grace_min   : entries allowed while the entry moment (= signal bar close
                  time) is inside [imp_h+1:00, imp_h+1:00 + grace_min)
    skip_dows   : list of weekdays (0=Mon .. 4=Fri) to skip

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "imp_h": 15,
    "imp_min_atr": 1.0,
    "imp_max_atr": 99.0,
    "mode": "follow",
    "grace_min": 30,
    "skip_dows": [],
}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    imp_h = int(p["imp_h"])
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)

    hours = out.index.hour
    day = out.index.normalize()
    in_imp = hours == imp_h
    imp_open = out["open"].where(in_imp).groupby(day).transform("first")
    imp_close = out["close"].where(in_imp).groupby(day).transform("last")

    # Fully-known impulse: visible on bars whose OPEN hour is strictly after
    # the impulse hour (every impulse bar has closed by then).
    known = hours >= imp_h + 1
    out["imp_move"] = (imp_close - imp_open).where(known)

    # Running impulse for bars INSIDE the impulse hour: this bar's own close
    # minus the hour's opening price. Causal — evaluate() only ever reads the
    # newest CLOSED bar, and each row uses only its own close and an earlier
    # open. Only the hour's LAST bar (close time == imp_h+1:00) is acted on.
    out["imp_live"] = (out["close"] - imp_open).where(in_imp)
    return out


def warmup(scfg: StrategyCfg) -> int:
    bars_per_day = 1440 // _bar_minutes(scfg)
    return 2 * bars_per_day + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=_bar_minutes(scfg))
    mins = entry_ts.hour * 60 + entry_ts.minute
    start = (int(p["imp_h"]) + 1) * 60
    if not (start <= mins < start + int(p["grace_min"])):
        return flat("outside entry window")
    if entry_ts.weekday() in p["skip_dows"]:
        return flat("skipped weekday")

    # At exactly imp_h+1:00 the signal bar is the impulse hour's last bar —
    # its imp_live IS the full-hour impulse. Later grace bars use imp_move.
    imp = last["imp_live"] if mins == start else last["imp_move"]
    if not np.isfinite(imp):
        return flat("no impulse measured")
    a = float(last["atr"])
    if abs(float(imp)) < float(p["imp_min_atr"]) * a:
        return flat("impulse too small")
    if abs(float(imp)) > float(p["imp_max_atr"]) * a:
        return flat("impulse too large")

    direction = LONG if float(imp) > 0 else SHORT
    if str(p["mode"]) == "fade":
        direction = -direction

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    return Signal(direction, float(last["close"]), sl_dist, tp_dist, a,
                  f"NY impulse {float(imp):+.2f} ({p['mode']}), ATR {a:.2f}")
