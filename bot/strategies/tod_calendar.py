"""tod_calendar -- day-of-week and turn-of-month calendar effects (tod_ family).

Pure calendar entries: on an allowed WEEKDAY, inside an allowed
TURN-OF-MONTH window, take one directional trade at a fixed clock hour and
let ATR barriers manage it.  Nothing is derived from price except an
optional trend filter and the ATR that sizes the barriers, so if this makes
money the money comes from the calendar.

Every gate here is deterministic from the timestamp alone (weekday, calendar
day-of-month, days remaining in the calendar month), so causality is trivial
-- no rolling statistic is involved at all.  Note this uses CALENDAR days,
not "the Nth trading day", precisely because the trading-day count of a
month is only knowable in hindsight when holidays move.

Params (scfg.params)
  dows        : allowed entry weekdays, 0=Mon .. 4=Fri (default all)
  entry_hour  : clock hour (server) at which the entry is placed, or -1 to
                allow EVERY bar inside the session window (higher frequency:
                the one-position rule and cooldown_bars do the throttling)
  entry_min   : minute within that hour (0 unless you want a half-hour slot);
                ignored when entry_hour = -1
  tom_mode    : "off"   -> no turn-of-month gate
                "first" -> only calendar day-of-month <= tom_k
                "last"  -> only days-to-month-end < tom_k
                "both"  -> either of the two
                "mid"   -> the complement of "both" (the control)
  tom_k       : size of the turn-of-month window in calendar days
  dir         : "long", "short", or "tom" -- long inside the first-k window
                and short inside the last-k window (needs tom_mode="both";
                this is the two-sided turn-of-month trade, ~2x the frequency)
  trend_filter: 1 = require close on the right side of EMA(ema_trend)

SL/TP: scfg.atr_sl_mult / atr_tp_mult x ATR.
"""
from __future__ import annotations

import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "dows": [0, 1, 2, 3, 4],
    "entry_hour": 9,
    "entry_min": 0,
    "tom_mode": "off",
    "tom_k": 3,
    "dir": "long",
    "trend_filter": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need = 5 * scfg.atr_period
    if int(p["trend_filter"]):
        need = max(need, 5 * scfg.ema_trend)
    return need + 10


def _tom_ok(ts: pd.Timestamp, mode: str, k: int) -> bool:
    if mode == "off":
        return True
    dom = ts.day                                   # 1..31
    dim = ts.days_in_month
    dtme = dim - dom                               # 0 on the last calendar day
    first = dom <= k
    last = dtme < k
    if mode == "first":
        return first
    if mode == "last":
        return last
    if mode == "both":
        return first or last
    if mode == "mid":
        return not (first or last)
    return True


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=_bar_minutes(scfg))
    if entry_ts.weekday() not in list(p["dows"]):
        return flat("weekday not allowed")
    eh = int(p["entry_hour"])
    if eh >= 0 and (entry_ts.hour != eh or entry_ts.minute != int(p["entry_min"])):
        return flat("not the entry slot")
    if not _tom_ok(entry_ts, str(p["tom_mode"]).lower(), int(p["tom_k"])):
        return flat("outside turn-of-month window")

    dirp = str(p["dir"]).lower()
    if dirp == "tom":
        k = int(p["tom_k"])
        if entry_ts.day <= k:
            direction = LONG
        elif entry_ts.days_in_month - entry_ts.day < k:
            direction = SHORT
        else:
            return flat("dir=tom needs the first-k or last-k window")
    else:
        direction = LONG if dirp == "long" else SHORT
    if int(p["trend_filter"]):
        c, trend = float(last["close"]), float(last["ema_trend"])
        if direction == LONG and c <= trend:
            return flat("long against trend")
        if direction == SHORT and c >= trend:
            return flat("short against trend")

    a = float(last["atr"])
    side = "long" if direction == LONG else "short"
    return Signal(direction, float(last["close"]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a,
                  f"cal {side}: dow {entry_ts.weekday()} @ {entry_ts:%H:%M} "
                  f"dom {entry_ts.day} tom={p['tom_mode']}")
