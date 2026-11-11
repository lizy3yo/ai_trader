"""User's RISK FRAMEWORK applied to any entry signal (2026-11-09).

Every trade = TREND + ENTRY + ATR STOP + fixed REWARD:RISK (never below 1:1.5):
  trend   only take the inner strategy's signal in the direction of
          EMA(cfg.ema_trend) (default 200): longs above it, shorts below it
  entry   the inner strategy's own entry signal at its default settings
          (params: inner = module name, inner_params = its settings)
  stop    cfg.atr_sl_mult x ATR(14) from entry (study values 1 / 1.5 / 2)
  target  cfg.atr_tp_mult x ATR = 1.5 / 2 / 3 x the stop
The inner strategy's own stop, target and exit rules are ignored: a trade ends
only at the stop or the target. The random baseline (dir both, trend 1, same
cfg.atr_* values) gets the identical trend filter, stop and target.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat

DEFAULTS = {"inner": "trend_donchian", "inner_params": {}, "use_trend": 1}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _inner(scfg: StrategyCfg):
    from . import get_strategy
    p = _p(scfg)
    c = copy.copy(scfg)
    c.params = dict(p["inner_params"])
    return get_strategy(p["inner"]), c


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    mod, icfg = _inner(scfg)
    out = mod.add_indicators(df, icfg)
    out["fw_atr"] = atr(df, scfg.atr_period).to_numpy()
    out["fw_ema"] = ema(df["close"], scfg.ema_trend).to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    mod, icfg = _inner(scfg)
    return max(int(mod.warmup(icfg)), 3 * scfg.ema_trend + 10)


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    mod, icfg = _inner(scfg)
    sig = mod.evaluate(ind_window, icfg)
    if sig.direction == 0:
        return sig
    last = ind_window.iloc[-1]
    a = float(last["fw_atr"])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    if int(_p(scfg)["use_trend"]) and (float(last["close"]) - float(last["fw_ema"])) * sig.direction <= 0:
        return flat("against the 200 EMA trend")
    return Signal(LONG if sig.direction > 0 else SHORT, float(last["close"]),
                  scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a, f"fw:{_p(scfg)['inner']}")
