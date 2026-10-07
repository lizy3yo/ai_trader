"""Trend-following EMA crossover (the bot's original default).

Long:  EMA(fast) crosses above EMA(slow), close above EMA(trend),
       RSI below overbought.  Short is the mirror.
SL/TP: atr_sl_mult / atr_tp_mult x ATR (from shared StrategyCfg fields).

Knobs on scfg: ema_fast, ema_slow, ema_trend, rsi_period,
rsi_overbought, rsi_oversold (kept as first-class config fields since this
is the default strategy).
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import FLAT, LONG, SHORT, Signal, atr, ema, flat, rsi, valid_atr


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], scfg.ema_fast)
    out["ema_slow"] = ema(out["close"], scfg.ema_slow)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["rsi"] = rsi(out["close"], scfg.rsi_period)
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    # 5x the trend-EMA span leaves <0.007% weight on the ewm seed, so a
    # windowed live computation matches a full-history backtest computation
    return max(5 * scfg.ema_trend, scfg.ema_slow, scfg.rsi_period, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    sl_dist = scfg.atr_sl_mult * float(last["atr"])
    tp_dist = scfg.atr_tp_mult * float(last["atr"])

    crossed_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    crossed_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    if crossed_up and last["close"] > last["ema_trend"] and last["rsi"] < scfg.rsi_overbought:
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, float(last["atr"]),
            f"EMA{scfg.ema_fast}x{scfg.ema_slow} up, above EMA{scfg.ema_trend}, "
            f"RSI {last['rsi']:.0f}",
        )
    if crossed_dn and last["close"] < last["ema_trend"] and last["rsi"] > scfg.rsi_oversold:
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, float(last["atr"]),
            f"EMA{scfg.ema_fast}x{scfg.ema_slow} down, below EMA{scfg.ema_trend}, "
            f"RSI {last['rsi']:.0f}",
        )
    return flat("no crossover")
