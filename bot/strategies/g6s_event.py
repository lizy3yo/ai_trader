"""Scheduled-event drift UNION on intraday bars (round 6, g6s_).

Round 5 found two separate near misses on gold H1: the pre-FOMC drift (hold
from the US close before the decision day to just before the 14:00 ET
statement) and the pre-holiday drift (hold the last US session before an NYSE
holiday). Both are "risk premium ahead of a scheduled event" effects; this
module trades their union (plus optionally the pre-NFP hold, which exits just
before the 08:30 ET release) as ONE rule, each event with its own exit time.

Events (comma list in `events`):
  fomcpre  enter at the close (23:00 server) of the trading day before a
           scheduled FOMC decision day, exit at 21:00 server on decision day
  prehol   enter at the close of trading day A - prehol_nb - 1, exit at the
           close of A (A = last trading day before a scheduled NYSE holiday)
  nfppre   enter at the close of the day before the jobs report, exit at
           15:00 server on release day (release 15:30 server)
Trading days / holidays / event dates come from lm_calendar(_data) and are all
scheduled in advance (causal). Every NON-entry bar stores a planned exit one
trading day later at 22:00 server, so the random baseline (exit_from) holds
for a comparable ~1 day.
fomc_et=1: the fomcpre exit is the statement time 14:00 New York converted to
server time (Europe/Athens clock) per date, instead of a fixed 21:00 server.
They differ only in the ~2 weeks a year when US and EU daylight saving
disagree (then 21:00 server is one hour AFTER the statement).
params: events, prehol_nb, side ("long"/"short"), fomc_et
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat
from .lm_calendar import _anchors, trading_days

DEFAULTS = {"events": "fomcpre,prehol", "prehol_nb": 0, "side": "long", "fomc_et": 0}
ENTRY_MIN = 1380
EXIT_MIN = {"fomcpre": 1260, "prehol": 1380, "nfppre": 900}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    td = trading_days()
    idx = out.index
    step = idx.to_series().diff().mode().iloc[0] if len(idx) > 1 else pd.Timedelta(days=1)
    day = idx.normalize().values.astype("datetime64[D]")
    close_min = (idx.hour * 60 + idx.minute).values + int(step / pd.Timedelta(minutes=1))
    late = close_min >= ENTRY_MIN
    k = np.searchsorted(td, day, side="right") - 1          # last trading day <= bar date
    # default planned exit (random baseline / non-event bars): next trading day 22:00
    nxt = td[np.clip(k + 1, 0, len(td) - 1)]
    exit_at = (pd.DatetimeIndex(nxt) + pd.Timedelta(minutes=1320)).asi8.copy()
    entry = np.zeros(len(out), dtype=bool)
    for ev in [e.strip() for e in str(p["events"]).split(",") if e.strip()]:
        if ev == "fomcpre":
            a, nb, hold = _anchors(td, {"mode": "fomc"}), 0, 1
        elif ev == "nfppre":
            a, nb, hold = _anchors(td, {"mode": "nfp"}), 0, 1
        elif ev == "prehol":
            nb = int(p["prehol_nb"])
            a, hold = _anchors(td, {"mode": "preholiday"}), nb + 1
        else:
            raise ValueError(f"g6s_event: unknown event {ev!r}")
        e = a - nb - 1
        e = e[e >= 0]
        cand = np.isin(day, td[e]) & late
        first = cand & ~np.r_[False, (day[1:] == day[:-1]) & cand[:-1]]
        xd = td[np.clip(k + hold, 0, len(td) - 1)]
        ex = (pd.DatetimeIndex(xd) + pd.Timedelta(minutes=EXIT_MIN[ev])).asi8
        if ev == "fomcpre" and int(p["fomc_et"]):
            ny = (pd.DatetimeIndex(xd) + pd.Timedelta(hours=14)).tz_localize("America/New_York")
            ex = ny.tz_convert("Europe/Athens").tz_localize(None).asi8
        # an entry bar shared by two events keeps the later exit
        exit_at = np.where(first & entry, np.maximum(exit_at, ex), np.where(first, ex, exit_at))
        entry |= first
    out["ev_entry"] = entry
    out["ev_exit_at"] = exit_at
    out["ev_close"] = (idx + step).asi8
    return out


def warmup(scfg: StrategyCfg) -> int:
    return scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if not bool(ind_window["ev_entry"].iat[-1]):
        return flat("no event entry")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = SHORT if _p(scfg)["side"] == "short" else LONG
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, f"event {_p(scfg)['events']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    if bars_held + 1 > len(ind_window):
        return False
    target = ind_window["ev_exit_at"].iat[-(bars_held + 1)]   # stored on the signal bar
    return bool(ind_window["ev_close"].iat[-1] >= target)
