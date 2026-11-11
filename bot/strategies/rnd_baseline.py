"""Random-entry BASELINE (a null hypothesis, never a strategy to trade).

Fires with probability `prob` on any eligible bar with the shared ATR
stop/target. Used by tools/edge_test.py and tools/factory.py: a real
strategy must beat many seeds of this with the SAME session, direction,
stop/target geometry, holding rules, exit rule and costs, or its "edge" is
just being long an asset that went up (or paying less spread by luck).

params:
  prob, seed
  dir        "long" / "short" / "both" (both = random side per entry)
  trend      1 = only when close is on the trade's side of EMA(ema_trend)
  exit_from  optional strategy module name: reuse ITS exit_signal (and its
             indicators, built with exit_params) so trailing/indicator exits
             are matched too
  exit_params dict of that strategy's params
Deterministic per (bar time, seed), so it is causal and reproducible.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {"prob": 0.01, "seed": 0, "dir": "long", "trend": 0,
            "exit_from": "", "exit_params": {}}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _uniform(index: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """splitmix64 of (minutes since epoch, seed): independent uniforms per seed."""
    with np.errstate(over="ignore"):
        z = (index.asi8 // 60_000_000_000).astype(np.uint64)
        z = z + np.uint64(0x9E3779B97F4A7C15) * np.uint64(int(seed) + 1)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(2**53)


def _exit_module(p: dict):
    if not p["exit_from"]:
        return None
    from . import get_strategy
    return get_strategy(p["exit_from"])


def _exit_cfg(scfg: StrategyCfg, p: dict) -> StrategyCfg:
    c = copy.copy(scfg)
    c.params = dict(p["exit_params"])
    return c


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    mod = _exit_module(p)
    out = mod.add_indicators(df, _exit_cfg(scfg, p)) if mod else df.copy()
    out["rb_atr"] = atr(df, scfg.atr_period)
    out["rb_ema"] = ema(df["close"], scfg.ema_trend)
    out["rb_u"] = _uniform(out.index, p["seed"])
    out["rb_side"] = _uniform(out.index, int(p["seed"]) + 10_007)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = max(3 * scfg.ema_trend, scfg.atr_period) + 10 if int(p["trend"]) else scfg.atr_period + 10
    mod = _exit_module(p)
    if mod:
        w = max(w, mod.warmup(_exit_cfg(scfg, p)))
    return w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    a = float(last["rb_atr"])
    if not np.isfinite(a) or a <= 0 or float(last["rb_u"]) >= float(p["prob"]):
        return flat("no draw")
    if p["dir"] == "both":
        d = LONG if float(last["rb_side"]) < 0.5 else SHORT
    else:
        d = LONG if p["dir"] == "long" else SHORT
    if int(p["trend"]) and (float(last["close"]) - float(last["rb_ema"])) * d <= 0:
        return flat("against trend")
    # never enter where the borrowed exit would fire at once: otherwise a
    # "state" exit (close above its MA, inside the channel...) turns many
    # random entries into instant 1-bar losers and the null looks too weak
    mod = _exit_module(p)
    hook = getattr(mod, "exit_signal", None) if mod else None
    if hook is not None and hook(ind_window, _exit_cfg(scfg, p), d, 1):
        return flat("exit already active")
    return Signal(d, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a, "random")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = _p(scfg)
    mod = _exit_module(p)
    hook = getattr(mod, "exit_signal", None) if mod else None
    if hook is None:
        return False
    return bool(hook(ind_window, _exit_cfg(scfg, p), direction, bars_held))
