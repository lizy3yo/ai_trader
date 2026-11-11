"""BTCUSD swing: N-day-high breakout / price-above-SMA regime, trailing exits.

Entry (entry param):
  "brk": close > highest close of the prior n_days (Quantpedia MAX(x) idea),
         optionally only while close > SMA(sma_days) (Detzel regime gate).
  "sma": close > SMA(sma_days) (pure regime: long while above).
Shorts = exact mirror (side = "short"), only as a check.

Exit (exit param, via exit_signal; fills at next bar open):
  0 = none: fixed ATR SL / TP only
  1 = close crosses the midpoint of the x_days Donchian channel (trailing)
  2 = close crosses the x_days Donchian lower band (Turtle-style)
  3 = close crosses SMA(sma_days) (regime exit)
The ATR SL (cfg.atr_sl_mult x ATR(cfg.atr_period)) is always on as the
catastrophe stop and defines 1R; set cfg.atr_tp_mult very wide for trailing.

daily = 1: entries and signal exits are only decided on the bar that closes
at server midnight (mimics the published daily-close rules, cuts whipsaw).
vol_cap > 0: skip entries when ATR(period)/ATR(10*period) > vol_cap.

params: entry, side, n_days, sma_days, exit, x_days, daily, vol_cap,
        null_prob, null_seed (edge-test hook, see btcsw_common).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, flat, sma, valid_atr
from .btcsw_common import bars_per_day, day_end_flag, uniform

DEFAULTS = {
    "entry": "brk",
    "side": "long",
    "n_days": 20,
    "sma_days": 0,
    "exit": 1,
    "x_days": 0,        # 0 = n_days // 2 (min 1)
    "daily": 1,
    "vol_cap": 0.0,
    "null_prob": 0.0,
    "null_seed": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _xdays(p: dict) -> int:
    return int(p["x_days"]) or max(1, int(p["n_days"]) // 2)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    bpd = bars_per_day(scfg.timeframe)
    out = df.copy()
    n = max(1, int(p["n_days"]) * bpd)
    out["hc"] = out["close"].rolling(n).max().shift(1)
    out["lc"] = out["close"].rolling(n).min().shift(1)
    xb = _xdays(p) * bpd
    hh, ll = donchian(out, xb)
    out["x_hi"], out["x_lo"] = hh, ll
    out["x_mid"] = (hh + ll) / 2.0
    sd = int(p["sma_days"])
    out["sma"] = sma(out["close"], sd * bpd) if sd > 0 else np.nan
    out["atr"] = atr(out, scfg.atr_period)
    if float(p["vol_cap"]) > 0:
        out["volr"] = out["atr"] / atr(out, 10 * scfg.atr_period)
    else:
        out["volr"] = 0.0
    out["dend"] = day_end_flag(out.index, scfg.timeframe)
    out["u"] = uniform(out.index, int(p["null_seed"])) if float(p["null_prob"]) > 0 else 1.0
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bpd = bars_per_day(scfg.timeframe)
    days = max(int(p["n_days"]), _xdays(p), int(p["sma_days"]))
    w = days * bpd + 2
    if float(p["vol_cap"]) > 0:
        w = max(w, 30 * scfg.atr_period)
    return max(w, 3 * scfg.atr_period) + 5


def _sgn(p: dict) -> int:
    return SHORT if p["side"] == "short" else LONG


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if int(p["daily"]) and not bool(last["dend"]):
        return flat("not a day close")
    if not valid_atr(last):
        return flat("bad ATR")
    d = _sgn(p)
    c = float(last["close"])
    sd = int(p["sma_days"])
    if sd > 0:
        if pd.isna(last["sma"]) or (c - float(last["sma"])) * d <= 0:
            return flat("regime gate")
    if float(p["vol_cap"]) > 0 and float(last["volr"]) > float(p["vol_cap"]):
        return flat("vol too high")

    if float(p["null_prob"]) > 0:
        # coin flip, but never on a bar where the exit rule would close the
        # trade straight away (keeps the null from handing itself free losses)
        trig = float(last["u"]) < float(p["null_prob"]) and not exit_signal(
            ind_window, scfg, d, 0)
    elif p["entry"] == "sma":
        trig = sd > 0          # gate above already required close beyond SMA
    else:
        ref = last["hc"] if d == LONG else last["lc"]
        trig = (not pd.isna(ref)) and (c - float(ref)) * d > 0
    if not trig:
        return flat("no trigger")
    a = float(last["atr"])
    return Signal(d, c, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"btcsw {p['entry']} {p['side']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int,
                bars_held: int) -> bool:
    p = _p(scfg)
    mode = int(p["exit"])
    if mode == 0:
        return False
    last = ind_window.iloc[-1]
    if int(p["daily"]) and not bool(last["dend"]):
        return False
    c = float(last["close"])
    if mode == 1:
        ref = last["x_mid"]
    elif mode == 2:
        ref = last["x_lo"] if direction == LONG else last["x_hi"]
    else:
        ref = last["sma"]
    if pd.isna(ref):
        return False
    return (c - float(ref)) * direction < 0
