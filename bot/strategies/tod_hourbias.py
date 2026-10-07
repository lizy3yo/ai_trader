"""tod_hourbias -- ADAPTIVE time-of-day directional drift (tod_ family).

The honest version of "is there an hour that just pays?".

Instead of the researcher eyeballing 24 hours on the train split and then
"validating" the winner (24 comparisons, guaranteed to produce a pretty
number), this module picks the hour CAUSALLY, bar by bar, from data that was
already available at that moment:

  for every time-of-day slot (hour, or hour:minute) we keep a running mean of
  the realised `hold_bars`-bar forward move of every PAST occurrence of that
  slot.  At decision time we look up the slot we are about to enter, and if
  its running mean drift is >= min_drift we go LONG, <= -min_drift we go
  SHORT.  Nothing about the future -- or about the researcher's hindsight --
  enters the decision.

Causality notes
  * `fwd` at row i is 100*(close[i]/open[i-hold+1] - 1): a move that has
    FULLY COMPLETED by the close of bar i.  It is attributed to the slot of
    bar i-hold+1, i.e. the slot the move STARTED in.
  * the running mean is an expanding (lookback=0) or rolling (lookback=N)
    mean over that slot's own series, INCLUDING the current row -- every
    term of which is a completed, already-observed move.  Row i uses rows
    <= i only, so add_indicators(df[:i+1]).iloc[-1] == add_indicators(df)[i].
  * evaluate() reads the drift stored on the most recent EARLIER bar whose
    slot matches the slot we are about to enter.  That bar closed at least a
    bar ago, so its value was public before we act.

Params (scfg.params)
  hold_bars   : horizon (bars) the drift statistic is measured over (>=1)
  lookback    : 0 = expanding mean over all history, N = last N occurrences
  min_obs     : minimum past occurrences of the slot before it may be traded
  min_drift   : |running mean| (percent of price) needed to fire
  by_slot     : 1 = per hour:minute slot, 0 = per clock hour
  mode        : "auto"  -> sign of the drift picks the side
                "long"  -> only take the long side of an auto signal
                "short" -> only take the short side
                "fixed" -> ignore the drift, trade `fixed_dir` in `hours`
                           (the deliberately data-snooped control)
  hours       : [] = every hour allowed, else only these ENTRY clock hours
  fixed_dir   : 1 / -1, used by mode="fixed"
  first_only  : 1 = only fire on the first bar of the entry hour
  trend_filter: 1 = require close on the right side of EMA(ema_trend)

SL/TP: scfg.atr_sl_mult / atr_tp_mult x ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "hold_bars": 1,
    "lookback": 0,
    "min_obs": 40,
    "min_drift": 0.0,
    "by_slot": 1,
    "mode": "auto",
    "hours": [],
    "fixed_dir": 1,
    "first_only": 0,
    "trend_filter": 0,
}

_SEARCH = 300          # bars of tail scanned for the matching slot (> 1 day)


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    hold = max(1, int(p["hold_bars"]))
    if int(p["by_slot"]):
        slot = out.index.hour * 60 + out.index.minute
    else:
        slot = out.index.hour * 60
    slot_s = pd.Series(np.asarray(slot, dtype=np.int64), index=out.index)
    out["slot"] = slot_s

    # completed hold-bar move, attributed to the slot it STARTED in
    fwd = 100.0 * (out["close"] / out["open"].shift(hold - 1) - 1.0)
    start_slot = slot_s.shift(hold - 1)
    out["start_slot"] = start_slot

    lb = int(p["lookback"])
    mo = max(1, int(p["min_obs"]))
    grp = fwd.groupby(start_slot)
    if lb > 0:
        d = grp.rolling(lb, min_periods=mo).mean()
    else:
        d = grp.expanding(min_periods=mo).mean()
    # groupby().rolling/expanding returns a (slot, time) MultiIndex -> realign
    out["drift"] = d.reset_index(level=0, drop=True).reindex(out.index)
    out["dcount"] = grp.cumcount() + 1.0
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bars_per_day = max(1, 1440 // _bar_minutes(scfg))
    need = (int(p["min_obs"]) + 2) * bars_per_day + int(p["hold_bars"])
    if int(p["trend_filter"]):
        need = max(need, 5 * scfg.ema_trend)
    return max(need, 5 * scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    step = _bar_minutes(scfg)
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=step)
    if int(p["first_only"]) and entry_ts.minute != 0:
        return flat("not first bar of hour")
    hours = list(p["hours"])
    if hours and entry_ts.hour not in hours:
        return flat("hour not allowed")

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    mode = str(p["mode"]).lower()

    if mode == "fixed":
        direction = LONG if int(p["fixed_dir"]) > 0 else SHORT
        why = f"fixed dir at hour {entry_ts.hour}"
    else:
        want = entry_ts.hour * 60 + entry_ts.minute if int(p["by_slot"]) else entry_ts.hour * 60
        tail = ind_window.iloc[-_SEARCH:]
        ss = tail["start_slot"].to_numpy()
        hit = np.nonzero(ss == want)[0]
        if hit.size == 0:
            return flat("slot never seen yet")
        drift = float(tail["drift"].to_numpy()[hit[-1]])
        obs = float(tail["dcount"].to_numpy()[hit[-1]])
        if not np.isfinite(drift) or obs < float(p["min_obs"]):
            return flat("slot has too little history")
        md = float(p["min_drift"])
        if drift >= md and md >= 0 and drift > 0:
            direction = LONG
        elif drift <= -md and drift < 0:
            direction = SHORT
        else:
            return flat(f"slot drift {drift:+.4f}% below threshold")
        if mode == "long" and direction != LONG:
            return flat("short side disabled")
        if mode == "short" and direction != SHORT:
            return flat("long side disabled")
        why = f"slot {want//60:02d}:{want%60:02d} drift {drift:+.4f}% over {obs:.0f} obs"

    if int(p["trend_filter"]):
        trend = float(last["ema_trend"])
        c = float(last["close"])
        if direction == LONG and c <= trend:
            return flat("long against trend")
        if direction == SHORT and c >= trend:
            return flat("short against trend")

    side = "long" if direction == LONG else "short"
    return Signal(direction, float(last["close"]), sl_dist, tp_dist, a,
                  f"tod {side}: {why}")
