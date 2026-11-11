"""N-bar (Donchian) breakout, generic across markets (lt_ trend family).

Entry on the FIRST close beyond the highest high / lowest low of the prior n
bars. Optional trend gate: close vs EMA(cfg.ema_trend) (matches the factory's
trend=1 random baseline). Exits: fixed ATR SL/TP, or a trailing exit via
lt_common.add_exit (mid / opp / ma) with a wide TP.
params: n, side (long/short/both), trend (0/1), exit (fixed/mid/opp/ma), exit_n
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import donchian
from . import lt_common as C

DEFAULTS = {"n": 55, "side": "long", "trend": 0, "exit": "fixed", "exit_n": 20}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    hh, ll = donchian(out, int(p["n"]))
    up = (c > hh) & (c.shift(1) <= hh.shift(1))
    dn = (c < ll) & (c.shift(1) >= ll.shift(1))
    if int(p["trend"]):
        tu, td = C.trend_ok(out, scfg.ema_trend)
        up, dn = up & tu, dn & td
    C.add_exit(out, p["exit"], p["exit_n"])
    return C.finish(out, C.combine(up, dn, p["side"]), scfg.atr_period)


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = max(int(p["n"]), int(p["exit_n"]), scfg.atr_period) + 5
    return max(w, 2 * scfg.ema_trend) if int(p["trend"]) else w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg):
    return C.make_signal(ind_window, scfg, f"{_p(scfg)['n']}-bar breakout")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_hook(ind_window, direction)
