"""Killzone opening-drive momentum (killzone family, prefix kz_).

ICT-style "displacement" mechanized. Inside a killzone window on the broker
server clock (e.g. London open 09:00-12:00, NY 15:00-18:00) wait for an M5
bar whose BODY is at least disp_k x ATR — where ATR is taken from the
PREVIOUS bar, so a candle cannot inflate its own threshold — and optionally
whose body is at least body_frac of its full range (a real drive bar, not a
wick). Enter in the bar's direction ("follow") or against it ("fade") at the
next bar open.

nth = 1 : only the day's FIRST qualifying displacement bar (one-shot
          opening drive, one trade per day at most).
nth = 0 : every qualifying bar may enter (cooldown_bars spaces them).

Params (scfg.params):
    kz_start_h / kz_end_h : killzone = bars whose OPEN hour is in
                            [kz_start_h, kz_end_h) (server clock). Entry
                            fills happen one bar later, so the last possible
                            fill is at kz_end_h:00 sharp.
    disp_k       : body >= disp_k x ATR(previous bar)
    body_frac    : body/range >= this (0 disables the wick filter)
    nth          : see above (1 = first-only, 0 = all)
    mode         : "follow" (trade the drive) or "fade" (bet it retraces)
    sides        : "both" / "long" / "short"
    trend_filter : if true, only trade in the direction of EMA(ema_trend)
    min_atr      : skip entries when ATR (in $) is below this floor

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
Causality: ATR threshold uses .shift(1); the per-day displacement counter is
a groupby(day).cumsum(), which for row i only sees rows <= i of that day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "kz_start_h": 9,
    "kz_end_h": 12,
    "disp_k": 1.5,
    "body_frac": 0.0,
    "nth": 1,
    "mode": "follow",
    "sides": "both",
    "trend_filter": False,
    "min_atr": 0.0,
}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    atr_prev = out["atr"].shift(1)          # causal: yesterday's vol scale
    body = out["close"] - out["open"]
    rng = (out["high"] - out["low"]).replace(0.0, np.nan)

    hours = out.index.hour
    day = out.index.normalize()
    in_kz = (hours >= int(p["kz_start_h"])) & (hours < int(p["kz_end_h"]))

    big = body.abs() >= float(p["disp_k"]) * atr_prev          # NaN -> False
    solid = (body.abs() / rng).fillna(0.0) >= float(p["body_frac"])
    disp = in_kz & big & solid & (atr_prev > 0).fillna(False)

    out["disp_flag"] = disp.astype(float)
    # nth displacement bar of THIS day so far (cumsum = past+current rows only)
    out["disp_cnt"] = out["disp_flag"].groupby(day).cumsum()
    out["body"] = body
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
    if not valid_atr(last):
        return flat("bad ATR")
    if float(last["disp_flag"]) < 1.0:
        return flat("no displacement bar")

    nth = int(p["nth"])
    if nth >= 1 and int(last["disp_cnt"]) != nth:
        return flat("not the day's nth displacement")

    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")

    direction = LONG if float(last["body"]) > 0 else SHORT
    if str(p["mode"]) == "fade":
        direction = -direction

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    c = float(last["close"])
    if p["trend_filter"]:
        trend = float(last["ema_trend"])
        if direction == LONG and c <= trend:
            return flat("drive against trend filter")
        if direction == SHORT and c >= trend:
            return flat("drive against trend filter")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    side = "long" if direction == LONG else "short"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"KZ drive {p['mode']} {side}: body {float(last['body']):+.2f} "
                  f">= {p['disp_k']}xATR {a:.2f} (#{int(last['disp_cnt'])})")
