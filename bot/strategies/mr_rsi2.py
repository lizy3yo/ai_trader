"""Mean reversion: short-period RSI extreme, traded WITH the EMA trend.

Classic high-win-rate pullback design (Connors-style RSI(2)):
Long:  close above EMA(trend) and RSI(rsi_p) below buy_th  -> buy the dip,
       expecting a snap back up toward the trend.
Short: mirror (close below EMA(trend), RSI above 100 - buy_th), optional.

SL/TP from shared StrategyCfg ATR multipliers. For the intended high win
rate use a TP smaller than the SL (e.g. tp 1.0 x ATR, sl 2.5 x ATR).
A minimum TP floor (min_tp_usd) keeps targets clear of the ~$0.55 spread.

params (scfg.params):
    rsi_p      : RSI period (2-4 typical)
    buy_th     : RSI level under which longs fire (short level = 100 - buy_th)
    trend_ema  : EMA period of the trend filter
    both_sides : 1 = also short in downtrends, 0 = long-only
    min_tp_usd : skip signals whose TP distance is below this ($/oz)
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rsi, valid_atr

DEFAULTS = {
    "rsi_p": 2,
    "buy_th": 15.0,
    "trend_ema": 200,
    "both_sides": 1,
    "min_tp_usd": 1.5,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["mr_rsi"] = rsi(out["close"], int(p["rsi_p"]))
    out["mr_trend"] = ema(out["close"], int(p["trend_ema"]))
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(5 * int(p["trend_ema"]), 5 * scfg.atr_period, int(p["rsi_p"])) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    if tp_dist < float(p["min_tp_usd"]):
        return flat("TP below spread floor")

    r = float(last["mr_rsi"])
    close = float(last["close"])
    trend = float(last["mr_trend"])
    buy_th = float(p["buy_th"])
    sell_th = 100.0 - buy_th

    if close > trend and r < buy_th:
        return Signal(LONG, close, sl_dist, tp_dist, a,
                      f"RSI{int(p['rsi_p'])} {r:.0f}<{buy_th:.0f} dip in uptrend")
    if int(p["both_sides"]) and close < trend and r > sell_th:
        return Signal(SHORT, close, sl_dist, tp_dist, a,
                      f"RSI{int(p['rsi_p'])} {r:.0f}>{sell_th:.0f} pop in downtrend")
    return flat("no RSI extreme with trend")
