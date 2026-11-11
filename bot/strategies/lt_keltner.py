"""Keltner / ATR channel breakout (lt_ trend family).

Entry on the first close beyond EMA(n) +/- k x ATR(cfg.atr_period). Optional
trend gate close vs EMA(cfg.ema_trend). Exits: fixed ATR SL/TP, "mid"
(close back through EMA(n) = channel midline), or lt_common opp/ma.
params: n, k, side, trend, exit (fixed/mid/opp/ma), exit_n
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import atr, ema
from . import lt_common as C

DEFAULTS = {"n": 20, "k": 2.0, "side": "long", "trend": 0, "exit": "fixed", "exit_n": 20}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    mid = ema(c, int(p["n"]))
    a = atr(out, scfg.atr_period)
    hi, lo = mid + float(p["k"]) * a, mid - float(p["k"]) * a
    up = (c > hi) & (c.shift(1) <= hi.shift(1))
    dn = (c < lo) & (c.shift(1) >= lo.shift(1))
    if int(p["trend"]):
        tu, td = C.trend_ok(out, scfg.ema_trend)
        up, dn = up & tu, dn & td
    if p["exit"] == "mid":
        out["lt_xl"], out["lt_xs"] = c < mid, c > mid
    else:
        C.add_exit(out, p["exit"], p["exit_n"])
    return C.finish(out, C.combine(up, dn, p["side"]), scfg.atr_period)


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = 2 * int(p["n"]) + max(int(p["exit_n"]), scfg.atr_period) + 5
    return max(w, 2 * scfg.ema_trend) if int(p["trend"]) else w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg):
    p = _p(scfg)
    return C.make_signal(ind_window, scfg, f"Keltner EMA{p['n']} +/- {p['k']}ATR break")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_hook(ind_window, direction)
