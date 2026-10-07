"""London-open range breakout (session/time-structure family, prefix ses_).

Idea: the quiet Asia session (server 01:00-08:59) builds a range; the London
open (09:00+) often resolves it directionally. We compute each day's Asia
high/low CAUSALLY (values only become available on bars at/after the range
end hour) and trade the first close that breaks the range +/- an ATR buffer,
during a limited morning entry window.

Params (scfg.params):
    range_start_h / range_end_h : Asia range = bars whose OPEN hour is in
                                  [range_start_h, range_end_h)  (server clock)
    entry_start_h / entry_end_h : entries allowed while the ENTRY moment
                                  (= signal bar close time) is inside
                                  [entry_start_h, entry_end_h)
    buf_atr        : breakout buffer beyond the range, in ATRs
    min_range_atr  : skip days whose Asia range is narrower than this x ATR
    max_range_atr  : skip days whose Asia range is wider than this x ATR
    min_range_bars : require at least this many bars inside the range window
    skip_dows      : list of weekdays (0=Mon .. 4=Fri) to skip
    mode           : "follow" (trade the break) or "fade" (bet the break fails)
    trend_filter   : if true, only take trades aligned with EMA(ema_trend)

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "range_start_h": 1,
    "range_end_h": 9,
    "entry_start_h": 9,
    "entry_end_h": 12,
    "buf_atr": 0.2,
    "min_range_atr": 0.0,
    "max_range_atr": 99.0,
    "min_range_bars": 8,
    "skip_dows": [],
    "mode": "follow",
    "trend_filter": False,
    "sides": "both",     # "both" / "long" / "short"
}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    hours = out.index.hour
    day = out.index.normalize()
    in_range = (hours >= int(p["range_start_h"])) & (hours < int(p["range_end_h"]))

    rng_hi = out["high"].where(in_range).groupby(day).transform("max")
    rng_lo = out["low"].where(in_range).groupby(day).transform("min")
    rng_n = pd.Series(in_range.astype(float), index=out.index).groupby(day).transform("sum")

    # Causality: the day's range only becomes visible on bars whose OPEN hour
    # is at/after range_end_h — i.e. after every range bar has closed.
    complete = hours >= int(p["range_end_h"])
    out["rng_hi"] = rng_hi.where(complete)
    out["rng_lo"] = rng_lo.where(complete)
    out["rng_n"] = rng_n.where(complete)
    return out


def warmup(scfg: StrategyCfg) -> int:
    bars_per_day = 1440 // _bar_minutes(scfg)
    # 5x ema_trend keeps the ewm seed weight negligible (trend_filter option)
    return max(2 * bars_per_day + 5 * scfg.atr_period, 5 * scfg.ema_trend) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    # Entry happens at the NEXT bar open == this bar's close time.
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=_bar_minutes(scfg))
    mins = entry_ts.hour * 60 + entry_ts.minute
    if not (int(p["entry_start_h"]) * 60 <= mins < int(p["entry_end_h"]) * 60):
        return flat("outside entry window")
    if entry_ts.weekday() in p["skip_dows"]:
        return flat("skipped weekday")

    hi, lo, n = last["rng_hi"], last["rng_lo"], last["rng_n"]
    if not (np.isfinite(hi) and np.isfinite(lo)) or n < int(p["min_range_bars"]):
        return flat("no complete Asia range")

    a = float(last["atr"])
    width = float(hi) - float(lo)
    if width < float(p["min_range_atr"]) * a:
        return flat("range too narrow")
    if width > float(p["max_range_atr"]) * a:
        return flat("range too wide")

    buf = float(p["buf_atr"]) * a
    up, dn = float(hi) + buf, float(lo) - buf
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    c_last, c_prev = float(last["close"]), float(prev["close"])

    broke_up = c_prev <= up < c_last
    broke_dn = c_prev >= dn > c_last
    if not (broke_up or broke_dn):
        return flat("no range break")

    fade = str(p["mode"]) == "fade"
    if broke_up:
        direction = SHORT if fade else LONG
        desc = f"Asia hi {float(hi):.2f}+buf {buf:.2f}"
    else:
        direction = LONG if fade else SHORT
        desc = f"Asia lo {float(lo):.2f}-buf {buf:.2f}"

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    if p["trend_filter"]:
        trend = float(last["ema_trend"])
        if direction == LONG and c_last <= trend:
            return flat("break against trend filter")
        if direction == SHORT and c_last >= trend:
            return flat("break against trend filter")

    side = "long" if direction == LONG else "short"
    return Signal(direction, c_last, sl_dist, tp_dist, a,
                  f"ORB {p['mode']} {side}: close {c_last:.2f} broke {desc}")
