"""GOLD swing: buy a pullback inside an established uptrend (H4 research,
prefix gsw_). Long only.

Trend : close > EMA(cfg.ema_trend) and the daily regime of gsw_donchian
        (regime_mode 0 none | 1 prior daily close > SMA(ma_days) |
        2 12-month momentum > 0 | 3 both).
Setup : within the last `arm_bars` bars a close made a fresh `dip_lb`-bar
        closing low (the pullback).
Entry : the first close back above the previous bar's high (resumption).
Exit  : SL = cfg.atr_sl_mult*ATR, TP = cfg.atr_tp_mult*ATR; with exit_mode=1
        also when the close exceeds the prior `exit_lb`-bar high (the swing
        high is retested); cfg.max_hold_bars = time stop.
`start_after` (research only): no entries before this timestamp.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, Signal, atr, ema, flat, valid_atr
from .gsw_donchian import _regime_ok, daily_regime

DEFAULTS = {
    "dip_lb": 10,
    "arm_bars": 3,
    "regime_mode": 1,
    "ma_days": 210,
    "mom_days": 252,
    "nan_pass": 1,
    "exit_mode": 0,
    "exit_lb": 20,
    "start_after": "",
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    n = int(p["dip_lb"])
    dip = c < c.rolling(n).min().shift(1)
    out["armed"] = dip.astype(float).rolling(int(p["arm_bars"])).max()
    out["ex_hi"] = out["high"].rolling(int(p["exit_lb"])).max().shift(1)
    out["ema_trend"] = ema(c, scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    out["reg_sma"], out["reg_mom"] = daily_regime(out, int(p["ma_days"]), int(p["mom_days"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(5 * scfg.ema_trend, int(p["dip_lb"]) + int(p["arm_bars"]) + 2,
               int(p["exit_lb"]) + 2, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    if p["start_after"] and ind_window.index[-1] < pd.Timestamp(p["start_after"]):
        return flat("before start")
    last, prev = ind_window.iloc[-1], ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    # armed on the PREVIOUS bar (pullback already happened), resumption now
    if not (prev["armed"] > 0 and last["close"] > prev["high"]):
        return flat("no pullback resumption")
    if last["close"] <= last["ema_trend"] or not _regime_ok(last, p, LONG):
        return flat("not in uptrend")
    a = float(last["atr"])
    return Signal(LONG, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"pullback ({p['dip_lb']}-bar low) resumed in uptrend")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = _p(scfg)
    if int(p["exit_mode"]) != 1:
        return False
    last = ind_window.iloc[-1]
    return bool(not pd.isna(last["ex_hi"]) and last["close"] > last["ex_hi"])
