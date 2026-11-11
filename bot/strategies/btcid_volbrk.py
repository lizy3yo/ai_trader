"""btcid_volbrk -- Larry-Williams volatility breakout, intraday (BTCUSD research).

Day = server day (00:00 = MT5 rollover). Levels for today:
    up = today's open + k * previous day's range (high - low)
    dn = today's open - k * previous day's range
The FIRST bar of the day that closes beyond either level triggers (one trade
per day at most: any earlier close beyond a level today blocks later ones).
Entry fills at the next bar's open; the position is meant to be closed by
session.flat_at (e.g. "23:30" on M30, "23:45" on M15) before the rollover,
so no swap is paid. SL/TP = cfg.atr_sl_mult / cfg.atr_tp_mult x ATR (use a
very wide TP: the time exit is the intended exit).

Causal: previous-day range and SMA use completed days only; today's open is
the first bar of the day; the first-breakout flag uses rows <= i.

Params:
  k          : range multiplier
  mode       : "both" / "long" / "short"
  trend      : 1 = longs only if prev daily close > SMA(trend_days), shorts mirror
  trend_days : daily SMA length
  cutoff_h   : no new signals from this server hour on (bar CLOSE time)
  min_rng    : previous-day range must be >= this % of price (0 = off)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "k": 0.5,
    "mode": "both",
    "trend": 0,
    "trend_days": 20,
    "cutoff_h": 20,
    "min_rng": 0.0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    day = out.index.normalize()
    g = out.groupby(day)
    D = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last()})
    D["sma"] = D["c"].rolling(int(p["trend_days"])).mean()
    prev = D.shift(1)
    # previous day must be the calendar day before (no gaps / weekend holes)
    contiguous = (D.index.to_series().diff() == pd.Timedelta(days=1))
    prev.loc[~contiguous] = np.nan
    out["p_rng"] = (prev["h"] - prev["l"]).reindex(day).to_numpy()
    out["p_c"] = prev["c"].reindex(day).to_numpy()
    out["p_sma"] = prev["sma"].reindex(day).to_numpy()
    out["d_open"] = g["open"].transform("first").to_numpy()
    k = float(p["k"])
    out["up"] = out["d_open"] + k * out["p_rng"]
    out["dn"] = out["d_open"] - k * out["p_rng"]
    brk = ((out["close"] > out["up"]) | (out["close"] < out["dn"])).astype(int)
    out["brk_n"] = brk.groupby(day).cumsum()      # breakouts so far today (incl. this bar)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bpd = 1440 // TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)
    days = (int(p["trend_days"]) if int(p["trend"]) else 1) + 2
    return max(days * bpd, scfg.atr_period * 3) + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last) or not np.isfinite(last["p_rng"]):
        return flat("not formed")
    step = TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)
    close_ts = ind_window.index[-1] + pd.Timedelta(minutes=step)
    if close_ts.normalize() != ind_window.index[-1].normalize():
        return flat("last bar of day")
    if close_ts.hour >= int(p["cutoff_h"]):
        return flat("past cutoff")
    if int(last["brk_n"]) != 1:
        return flat("not first breakout")
    c = float(last["close"])
    if float(p["min_rng"]) > 0 and last["p_rng"] / c * 100.0 < float(p["min_rng"]):
        return flat("previous day too quiet")
    if c > last["up"]:
        direction = LONG
    elif c < last["dn"]:
        direction = SHORT
    else:
        return flat("no breakout")
    mode = str(p["mode"])
    if (mode == "long" and direction != LONG) or (mode == "short" and direction != SHORT):
        return flat("side disabled")
    if int(p["trend"]):
        if not np.isfinite(last["p_sma"]) or (last["p_c"] - last["p_sma"]) * direction <= 0:
            return flat("against daily trend")
    a = float(last["atr"])
    return Signal(direction, c, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"vol breakout k={p['k']} {'long' if direction == LONG else 'short'}")
