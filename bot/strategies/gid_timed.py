"""Timed intraday drift entry (gid_ family): enter at a fixed server time.

Mechanizes the published "gold rises in Asian hours" thesis (Blose &
Gondhalekar overnight-vs-daytime returns; CBS Chinese/Indian-hours drift):
buy shortly after the server-midnight rollover once spreads have normalized
and get out before the London open. Exit is either session.flat_at (live
use) or cfg.max_hold_bars (edge tests: random entries get identical holds).

Server clock = EET (UTC+2 winter / UTC+3 summer, EU DST rule; verified from
the 08:30 ET data spike sitting at 14:30 server in the US-DST-only weeks).
Daily reopen is server 01:00 (18:00 New York); London 08:00 = server 10:00.

Entry: the signal bar is the one whose CLOSE time equals entry_hm, so the
fill happens at the open of the bar starting at entry_hm.

Params (scfg.params):
    entry_hm   : "HH:MM" server entry time
    side       : "long" / "short"
    trend      : 1 = only when close is on the trade side of EMA(ema_trend)
    day_mom    : 1 = only when the previous server day closed up (long) /
                 down (short) vs its open  (daily momentum gate)
    skip_dows  : weekdays to skip (0=Mon .. 4=Fri)
SL/TP: scfg.atr_sl_mult / atr_tp_mult x ATR (use a wide SL/TP so the time
exit does the work).
"""
from __future__ import annotations

import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {"entry_hm": "01:15", "side": "long", "trend": 0, "day_mom": 0, "skip_dows": []}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    day = out.index.normalize()
    d_open = out["open"].groupby(day).transform("first")
    d_close = out["close"].groupby(day).transform("last")
    # previous COMPLETED day's direction, mapped onto every bar of the day
    dd = pd.Series((d_close - d_open).groupby(day).first())
    prev = dd.shift(1)
    out["prev_day_dir"] = prev.reindex(day).to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    return max(5 * scfg.ema_trend, scfg.atr_period, 2 * 1440 // _bar_minutes(scfg)) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=_bar_minutes(scfg))
    if entry_ts.strftime("%H:%M") != str(p["entry_hm"]):
        return flat("not entry time")
    if entry_ts.weekday() in p["skip_dows"]:
        return flat("skipped weekday")
    d = LONG if p["side"] == "long" else SHORT
    c = float(last["close"])
    if int(p["trend"]) and (c - float(last["ema_trend"])) * d <= 0:
        return flat("against trend")
    if int(p["day_mom"]):
        pdd = last["prev_day_dir"]
        if pd.isna(pdd) or float(pdd) * d <= 0:
            return flat("prev day against")
    a = float(last["atr"])
    return Signal(d, c, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"timed {p['side']} at {p['entry_hm']}")
