"""btcid_window -- BTCUSD calendar/time-window entries (intraday research).

One entry per day at a fixed server-clock time, held for a fixed time
(cfg.max_hold_bars) and/or until session.flat_at. Covers three published
BTC timing ideas with optional causal filters:

  * post-rollover drift   : long at 00:00/00:30 server (~21-23 UTC)
  * NYSE-close momentum   : long at 23:00 server (16:00 ET) when the price is
                            above the max of the prior `hi_days` same-time
                            snapshots (10-day high), hold to 17:00 next day
  * Monday Asia-open      : weekdays=[0], entry 02:00 server (Sun 19:00 NY),
                            direction = sign of the prior `follow_h` hours

Server clock = EET (UTC+2 / +3 summer) = New York + 7h (verified: tick-volume
peak sits at 16:30 server in both winter and summer = the 09:30 ET open).

Signal fires on the bar whose CLOSE time == entry_hm, so the fill is the
open of the entry_hm bar. Everything is causal: daily aggregates are taken
from COMPLETED previous server days only.

Params:
  entry_hm  : "HH:MM" server time of the fill
  dir       : 1 long / -1 short (ignored when follow_h > 0)
  follow_h  : >0 = trade the sign of the last follow_h hours' return
  weekdays  : [] all, else allowed weekdays (Mon=0) of the ENTRY time
  hv        : 1 = previous day's range (%) > median of the prior `hv_days`
  hv_days   : lookback for that median
  trend     : 1 = previous daily close > SMA(trend_days) of daily closes (long)
              (mirror for shorts)
  trend_days: SMA length in days
  hi_days   : >0 = entry-time price must exceed the max of the prior
              hi_days same-time snapshots (long; min for shorts)
SL/TP: cfg.atr_sl_mult / cfg.atr_tp_mult x ATR (use wide values: the time
exit is the real exit; shared cfg keys so edge_test's random baseline gets
the identical geometry).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "entry_hm": "00:00",
    "dir": 1,
    "follow_h": 0,
    "weekdays": [],
    "hv": 0,
    "hv_days": 30,
    "trend": 0,
    "trend_days": 20,
    "hi_days": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _step(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def _entry_min(p) -> int:
    h, m = (int(x) for x in str(p["entry_hm"]).split(":"))
    return h * 60 + m


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = _step(scfg)

    # ---- previous completed server-day aggregates ----
    day = out.index.normalize()
    g = out.groupby(day)
    D = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last()})
    D["rng"] = (D["h"] - D["l"]) / D["c"] * 100.0
    D["rng_med"] = D["rng"].rolling(int(p["hv_days"])).median()
    D["sma"] = D["c"].rolling(int(p["trend_days"])).mean()
    prev = D.shift(1)                     # only completed days
    out["p_rng"] = prev["rng"].reindex(day).to_numpy()
    out["p_rng_med"] = prev["rng_med"].reindex(day).to_numpy()
    out["p_c"] = prev["c"].reindex(day).to_numpy()
    out["p_sma"] = prev["sma"].reindex(day).to_numpy()

    # ---- same-time snapshot extremes (for the N-day-high filter) ----
    hd = int(p["hi_days"])
    if hd > 0:
        close_t = out.index + pd.Timedelta(minutes=step)
        is_snap = (close_t.hour * 60 + close_t.minute) == _entry_min(p)
        snap = out["close"][is_snap]
        out["snap_max"] = snap.rolling(hd).max().shift(1).reindex(out.index).ffill()
        out["snap_min"] = snap.rolling(hd).min().shift(1).reindex(out.index).ffill()
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bpd = 1440 // _step(scfg)
    days = max(int(p["hv_days"]) if int(p["hv"]) else 0,
               int(p["trend_days"]) if int(p["trend"]) else 0,
               int(p["hi_days"])) + 2
    return max(days * bpd, scfg.atr_period * 3, int(p["follow_h"]) * 60 // _step(scfg) + 2) + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    step = _step(scfg)
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=step)
    if entry_ts.hour * 60 + entry_ts.minute != _entry_min(p):
        return flat("not entry time")
    wds = list(p["weekdays"])
    if wds and entry_ts.weekday() not in wds:
        return flat("weekday not allowed")

    c = float(last["close"])
    fh = int(p["follow_h"])
    if fh > 0:
        nb = fh * 60 // step
        past = ind_window["close"].iloc[-1 - nb]
        if ind_window.index[-1] - ind_window.index[-1 - nb] > pd.Timedelta(minutes=step * nb * 1.5):
            return flat("gap in follow window")
        direction = LONG if c > past else SHORT
    else:
        direction = LONG if int(p["dir"]) > 0 else SHORT

    if int(p["hv"]):
        if not (np.isfinite(last["p_rng"]) and np.isfinite(last["p_rng_med"])):
            return flat("hv filter not formed")
        if last["p_rng"] <= last["p_rng_med"]:
            return flat("previous day quiet")
    if int(p["trend"]):
        if not np.isfinite(last["p_sma"]):
            return flat("trend not formed")
        if (last["p_c"] - last["p_sma"]) * direction <= 0:
            return flat("against daily trend")
    if int(p["hi_days"]) > 0:
        ref = last["snap_max"] if direction == LONG else last["snap_min"]
        if not np.isfinite(ref) or (c - ref) * direction <= 0:
            return flat("not at N-day extreme")

    a = float(last["atr"])
    return Signal(direction, c, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"window {p['entry_hm']} {'long' if direction == LONG else 'short'}")
