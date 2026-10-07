"""Session open-range breakout (breakouts family, prefix brk_).

Build TODAY's high/low range over a fixed server-clock window (e.g. the
quiet Asia session 01:00-08:00), then trade the FIRST close beyond that
range later in the day (e.g. during the London ramp). The harness's
session.window controls WHICH hours may enter; the range itself comes from
`range_window`.

Causality: a bar can only reference a range window that is already COMPLETE
(entries require bar-time >= range end, and the range window excludes the
entry bar itself), and the first-break bookkeeping uses only prior bars of
the same day.

Knobs in scfg.params:
    range_window  "HH:MM-HH:MM" server-clock window that defines the range
    buffer_atr    close must exceed the range edge by this many ATRs
    direction     "both" / "long" / "short"

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config fields).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "range_window": "01:00-08:00",
    "buffer_atr": 0.0,
    "direction": "both",
}


def _params(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _hhmm_to_min(s: str) -> int:
    h, m = (int(x) for x in s.strip().split(":"))
    return h * 60 + m


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _params(scfg)
    out = df.copy()
    rs, re_ = (_hhmm_to_min(x) for x in str(p["range_window"]).split("-"))
    minutes = out.index.hour * 60 + out.index.minute
    in_rng = (minutes >= rs) & (minutes < re_)
    day = out.index.normalize()

    rng_hi = out["high"].where(in_rng).groupby(day).cummax().groupby(day).ffill()
    rng_lo = out["low"].where(in_rng).groupby(day).cummin().groupby(day).ffill()
    after = pd.Series(minutes >= re_, index=out.index)
    out["orb_hi"] = rng_hi.where(after)
    out["orb_lo"] = rng_lo.where(after)

    out["atr"] = atr(out, scfg.atr_period)
    buf = float(p["buffer_atr"]) * out["atr"]
    brk_up = ((out["close"] > out["orb_hi"] + buf) & out["orb_hi"].notna()).astype(int)
    brk_dn = ((out["close"] < out["orb_lo"] - buf) & out["orb_lo"].notna()).astype(int)
    # only the FIRST break of the day (in each direction) is a signal
    prior_up = brk_up.groupby(day).cummax().groupby(day).shift(1)
    prior_dn = brk_dn.groupby(day).cummax().groupby(day).shift(1)
    out["orb_long"] = ((brk_up == 1) & ~(prior_up.fillna(0) == 1)).astype(float)
    out["orb_short"] = ((brk_dn == 1) & ~(prior_dn.fillna(0) == 1)).astype(float)
    return out


def warmup(scfg: StrategyCfg) -> int:
    bars_per_day = 1440 // TIMEFRAME_MINUTES.get(scfg.timeframe, 30)
    return max(5 * scfg.atr_period, bars_per_day) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _params(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (np.isfinite(last["orb_hi"]) and np.isfinite(last["orb_lo"])):
        return flat("range not formed")

    go_long = last["orb_long"] > 0 and p["direction"] in ("both", "long")
    go_short = last["orb_short"] > 0 and p["direction"] in ("both", "short")

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    if go_long:
        return Signal(LONG, float(last["close"]), sl_dist, tp_dist, a,
                      f"open-range {p['range_window']} break up")
    if go_short:
        return Signal(SHORT, float(last["close"]), sl_dist, tp_dist, a,
                      f"open-range {p['range_window']} break down")
    return flat("no range break")
