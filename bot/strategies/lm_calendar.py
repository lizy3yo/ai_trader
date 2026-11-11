"""Calendar & US-event windows (lm_ library, generic across markets).

Holds a position over a window of US TRADING days around an anchor day:
  entry at the close of trading day (A - n_before - 1), exit at the close of
  trading day (A + m_after)  ->  holds H = n_before + m_after + 1 trading days.
"Close" = the first bar of that server day whose CLOSE time >= entry_min /
exit_min (minutes after server midnight; default 23:00 server = 16:00 ET, the
US cash close). On D1 every daily bar qualifies, so D1 = enter at the next
day's open, exit at the open after day A + m_after.

Anchors (mode):
  tom        first trading day of each month (turn-of-the-month: n_before = last
             days of the old month held, m_after = extra new-month days)
  fomc       scheduled FOMC decision day (pre-FOMC drift, Lucca & Moench 2015);
             on H1, exit_min = 1260 (21:00 server = 14:00 ET) exits AT the
             announcement = pure pre-announcement drift
  nfp        US jobs-report release day
  preholiday last trading day before a regular NYSE holiday
  dow        every trading day whose weekday == dow (0 = Mon)
Trading days = weekdays minus NYSE holidays (lm_calendar_data), for every
market (the effects are US-driven). All dates are scheduled in advance, so
signals are causal.

exit_signal works for ANY entry bar (it reads the planned exit time stored on
the signal bar = H trading days later), so the random baseline in
tools/factory.py gets the identical holding period.

params: mode, n_before, m_after, dow, entry_min, exit_min, side ("long"/"short")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat
from .lm_calendar_data import FOMC_DATES, NFP_DATES, US_MARKET_HOLIDAYS

DEFAULTS = {"mode": "tom", "n_before": 1, "m_after": 2, "dow": 0,
            "entry_min": 1380, "exit_min": 1380, "side": "long"}

# closures that were NOT scheduled in advance: non-trading days, never anchors
_UNSCHEDULED = {"2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11",
                "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def trading_days() -> np.ndarray:
    days = pd.bdate_range("2000-01-03", "2027-12-31")
    hol = pd.to_datetime(US_MARKET_HOLIDAYS)
    return days[~days.isin(hol)].values.astype("datetime64[D]")


def _anchors(td: np.ndarray, p: dict) -> np.ndarray:
    mode = p["mode"]
    s = pd.DatetimeIndex(td)
    if mode == "tom":
        ym = s.year * 12 + s.month
        return np.flatnonzero(np.r_[True, ym[1:] != ym[:-1]])
    if mode in ("fomc", "nfp"):
        ev = pd.to_datetime(FOMC_DATES if mode == "fomc" else NFP_DATES).values.astype("datetime64[D]")
        idx = np.searchsorted(td, ev)
        ok = (idx < len(td)) & (td[np.minimum(idx, len(td) - 1)] == ev)
        return idx[ok]
    if mode == "preholiday":
        hol = [h for h in US_MARKET_HOLIDAYS if h not in _UNSCHEDULED]
        ev = pd.to_datetime(hol).values.astype("datetime64[D]")
        return np.unique(np.searchsorted(td, ev) - 1)
    if mode == "dow":
        return np.flatnonzero(s.weekday == int(p["dow"]))
    raise ValueError(f"lm_calendar: unknown mode {mode!r}")


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    td = trading_days()
    nb, ma = int(p["n_before"]), int(p["m_after"])
    hold = nb + ma + 1
    a = _anchors(td, p)
    e = a - nb - 1
    entry_days = td[e[(e >= 0)]]

    idx = out.index
    step = idx.to_series().diff().mode().iloc[0] if len(idx) > 1 else pd.Timedelta(days=1)
    day = idx.normalize().values.astype("datetime64[D]")
    close_t = idx + step
    close_min = (idx.hour * 60 + idx.minute).values + int(step / pd.Timedelta(minutes=1))
    late = close_min >= int(p["entry_min"])
    cand = np.isin(day, entry_days) & late
    prev_same = np.r_[False, (day[1:] == day[:-1]) & cand[:-1]]
    out["cal_entry"] = cand & ~prev_same

    k = np.searchsorted(td, day, side="right") - 1          # last trading day <= bar date
    xd = td[np.clip(k + hold, 0, len(td) - 1)]
    exit_at = pd.DatetimeIndex(xd) + pd.Timedelta(minutes=int(p["exit_min"]))
    out["cal_exit_at"] = exit_at.asi8
    out["cal_close"] = close_t.asi8
    return out


def warmup(scfg: StrategyCfg) -> int:
    return scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if not bool(ind_window["cal_entry"].iat[-1]):
        return flat("no calendar entry")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    p = _p(scfg)
    d = SHORT if p["side"] == "short" else LONG
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, f"calendar {p['mode']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    if bars_held + 1 > len(ind_window):
        return False
    target = ind_window["cal_exit_at"].iat[-(bars_held + 1)]   # stored on the signal bar
    return bool(ind_window["cal_close"].iat[-1] >= target)
