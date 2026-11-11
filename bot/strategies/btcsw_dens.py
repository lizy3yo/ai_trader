"""BTCUSD swing: Donchian ENSEMBLE (after Zarattini/Pagani/Barbon,
"Catching crypto trends").

For each lookback L in LOOKBACK_DAYS a causal state machine is run: it goes
IN when the close breaks above the prior-L-day high and goes OUT when the close
falls below the trailing midpoint of the L-day channel. The ensemble score is
the mean state (0..1). Long when score >= thr, exit (exit_signal) when score
< xthr. Shorts mirror (side="short", lows / midpoint from below).

Stops: cfg.atr_sl_mult x ATR(cfg.atr_period) catastrophe stop (= vol-scaled,
defines 1R); TP = cfg.atr_tp_mult x ATR (set very wide: exit is the trail).
daily = 1: decide entries / exits only on the bar closing at server midnight.

params: thr, xthr, side, daily, set (lookback set id), null_prob, null_seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr
from .btcsw_common import bars_per_day, day_end_flag, uniform

SETS = {
    0: (10, 20, 30, 60, 90),
    1: (5, 10, 20, 40, 80),
    2: (20, 40, 60, 90, 120),
}

DEFAULTS = {"thr": 0.6, "xthr": 0.4, "side": "long", "daily": 1, "set": 0,
            "null_prob": 0.0, "null_seed": 0}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _state(close: np.ndarray, up: np.ndarray, mid: np.ndarray, d: int) -> np.ndarray:
    st = np.zeros(len(close))
    s = 0.0
    for i in range(len(close)):
        if np.isnan(up[i]) or np.isnan(mid[i]):
            s = 0.0
        elif s == 0.0 and (close[i] - up[i]) * d > 0:
            s = 1.0
        elif s == 1.0 and (close[i] - mid[i]) * d < 0:
            s = 0.0
        st[i] = s
    return st


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    bpd = bars_per_day(scfg.timeframe)
    d = SHORT if p["side"] == "short" else LONG
    out = df.copy()
    c = out["close"].to_numpy(dtype=float)
    states = []
    for L in SETS[int(p["set"])]:
        n = L * bpd
        hh = out["high"].rolling(n).max().shift(1)
        ll = out["low"].rolling(n).min().shift(1)
        mid = ((hh + ll) / 2.0).to_numpy(dtype=float)
        brk = (hh if d == LONG else ll).to_numpy(dtype=float)
        states.append(_state(c, brk, mid, d))
    out["score"] = np.mean(states, axis=0)
    out["atr"] = atr(out, scfg.atr_period)
    out["dend"] = day_end_flag(out.index, scfg.timeframe)
    out["u"] = uniform(out.index, int(p["null_seed"])) if float(p["null_prob"]) > 0 else 1.0
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(SETS[int(p["set"])]) * bars_per_day(scfg.timeframe) + 3 * scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if int(p["daily"]) and not bool(last["dend"]):
        return flat("not a day close")
    if not valid_atr(last):
        return flat("bad ATR")
    d = SHORT if p["side"] == "short" else LONG
    sc = float(last["score"])
    if float(p["null_prob"]) > 0:
        trig = float(last["u"]) < float(p["null_prob"]) and sc >= float(p["xthr"])
    else:
        trig = sc >= float(p["thr"])
    if not trig:
        return flat(f"score {sc:.2f}")
    a = float(last["atr"])
    return Signal(d, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"donchian ensemble {sc:.2f}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int,
                bars_held: int) -> bool:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if int(p["daily"]) and not bool(last["dend"]):
        return False
    return float(last["score"]) < float(p["xthr"])
