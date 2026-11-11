"""Fixed, untuned cross-market TREND rule for the diversified-portfolio study
(results/lx_portfolio.py). Not tuned per market — the same rule everywhere.

Entry: close breaks the highest high (LONG) / lowest low (SHORT) of the prior
`entry` bars (first cross only). Exit: close crosses the opposite `exit`-bar
channel (Turtle-style trailing exit, via exit_signal), or the initial ATR stop
(cfg.atr_sl_mult x ATR(cfg.atr_period)); TP is set very wide (cfg.atr_tp_mult).

params: entry (100), exit (50), side ("both")
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, flat, valid_atr

DEFAULTS = {"entry": 100, "exit": 50, "side": "both"}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["hi_e"], out["lo_e"] = donchian(out, int(p["entry"]))
    out["hi_x"], out["lo_x"] = donchian(out, int(p["exit"]))
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(int(p["entry"]), int(p["exit"]), scfg.atr_period) + 3


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last, prev = ind_window.iloc[-1], ind_window.iloc[-2]
    if not valid_atr(last) or pd.isna(last["hi_e"]) or pd.isna(prev["hi_e"]):
        return flat("not formed")
    a = float(last["atr"])
    up = prev["close"] <= prev["hi_e"] and last["close"] > last["hi_e"]
    dn = prev["close"] >= prev["lo_e"] and last["close"] < last["lo_e"]
    if up and p["side"] != "short":
        return Signal(LONG, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a, "breakout up")
    if dn and p["side"] != "long":
        return Signal(SHORT, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a, "breakout down")
    return flat("no breakout")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    # cross of the trailing channel (equal to "close beyond channel" for real
    # trades; avoids instant exits for random-baseline entries)
    last, prev = ind_window.iloc[-1], ind_window.iloc[-2]
    if pd.isna(last["lo_x"]) or pd.isna(prev["lo_x"]):
        return False
    if direction == LONG:
        return bool(last["close"] < last["lo_x"] and prev["close"] >= prev["lo_x"])
    return bool(last["close"] > last["hi_x"] and prev["close"] <= prev["hi_x"])
