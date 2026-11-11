"""Moving-average crossover (lt_ trend family).

Entry on the bar fast MA crosses slow MA (SMA or EMA). Optional trend gate
close vs EMA(cfg.ema_trend). Exits: fixed ATR SL/TP, "cross" (fast back
through slow = classic trend system, use a wide TP), or the lt_common
trailing modes (mid/opp/ma).
params: fast, slow, ma (sma/ema), side, trend, exit (fixed/cross/mid/opp/ma), exit_n
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import ema, sma
from . import lt_common as C

DEFAULTS = {"fast": 50, "slow": 200, "ma": "sma", "side": "long", "trend": 0,
            "exit": "cross", "exit_n": 20}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    f = sma if p["ma"] == "sma" else ema
    fa, sl = f(out["close"], int(p["fast"])), f(out["close"], int(p["slow"]))
    up = (fa > sl) & (fa.shift(1) <= sl.shift(1))
    dn = (fa < sl) & (fa.shift(1) >= sl.shift(1))
    if int(p["trend"]):
        tu, td = C.trend_ok(out, scfg.ema_trend)
        up, dn = up & tu, dn & td
    if p["exit"] == "cross":
        out["lt_xl"], out["lt_xs"] = fa < sl, fa > sl
    else:
        C.add_exit(out, p["exit"], p["exit_n"])
    return C.finish(out, C.combine(up, dn, p["side"]), scfg.atr_period)


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = max(int(p["slow"]) * (2 if p["ma"] == "ema" else 1), int(p["exit_n"]), scfg.atr_period) + 5
    return max(w, 2 * scfg.ema_trend) if int(p["trend"]) else w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg):
    p = _p(scfg)
    return C.make_signal(ind_window, scfg, f"{p['ma']}{p['fast']}/{p['slow']} cross")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_hook(ind_window, direction)
