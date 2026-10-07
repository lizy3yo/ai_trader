"""Short-lookback channel breakout with cost-clearing targets (fast_ family).

Goal of this family: TRADE OFTEN, but never with a target so small that the
$0.53-0.55 round-turn spread eats the edge. So the lookback is deliberately
short (20-120 bars on M15/M30 instead of the 360-H1-bar winner) while the
take-profit stays large (3-6 x ATR).

Long:  the first close that pushes above the highest high of the prior
       `lookback` bars (+ an optional ATR buffer). The toolbox donchian() is
       shift(1)-causal, so a bar can never break its own extreme, and the
       "prev close was still inside" test stops the signal re-firing on every
       bar of an extended run.
Short: the mirror through the lowest low.

Params (scfg.params):
    lookback       channel length in bars (the frequency knob)
    dir            "both" / "long" / "short"
    confirm        "close" (bar must CLOSE beyond the channel) or "wick"
    buf_atr        extra margin beyond the channel, in ATRs
    trend_mode     "none" | "ema" (close on the right side of EMA(ema_trend))
                   | "slope" (EMA(ema_trend) rising/falling over slope_len bars)
                   | "ema_slope" (both)
    slope_len      bars used by the slope test
    min_atr        absolute $ ATR floor (skip dead-volatility regimes)
    max_ext_atr    skip if the close is already this many ATRs beyond the
                   channel edge (anti-chase: the move is spent)
    min_width_atr  skip if the channel is narrower than this x ATR
    max_width_atr  skip if the channel is wider than this x ATR (a squeeze
                   preference: break out of compression, not out of chaos)

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR.
Causality: every column is rolling/shifted or an ewm recursion; no full-frame
statistics anywhere, so add_indicators(df[:i+1]).iloc[-1] == add_indicators(df).iloc[i].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr

DEFAULTS = {
    "lookback": 48,
    "dir": "both",
    "confirm": "close",
    "buf_atr": 0.0,
    "trend_mode": "none",
    "slope_len": 20,
    "min_atr": 0.0,
    "max_ext_atr": 99.0,
    "min_width_atr": 0.0,
    "max_width_atr": 99.0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _uses_trend(mode: str) -> bool:
    return str(mode) != "none"


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    hh, ll = donchian(out, int(p["lookback"]))       # already .shift(1)-safe
    out["dc_hi"] = hh
    out["dc_lo"] = ll
    out["atr"] = atr(out, scfg.atr_period)
    if _uses_trend(p["trend_mode"]):
        e = ema(out["close"], scfg.ema_trend)
        out["ema_trend"] = e
        out["ema_prev"] = e.shift(int(p["slope_len"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need = max(int(p["lookback"]) + 2, scfg.atr_period)
    if _uses_trend(p["trend_mode"]):
        # 5x the EMA span leaves a negligible ewm seed weight, so a windowed
        # live computation matches the full-history backtest computation
        need = max(need, 5 * scfg.ema_trend + int(p["slope_len"]))
    return need + 10


def _trend_ok(mode: str, direction: int, last) -> bool:
    mode = str(mode)
    if mode == "none":
        return True
    e = float(last["ema_trend"])
    if not np.isfinite(e):
        return False
    if mode in ("ema", "ema_slope"):
        if direction == LONG and not float(last["close"]) > e:
            return False
        if direction == SHORT and not float(last["close"]) < e:
            return False
    if mode in ("slope", "ema_slope"):
        ep = float(last["ema_prev"])
        if not np.isfinite(ep):
            return False
        if direction == LONG and not e > ep:
            return False
        if direction == SHORT and not e < ep:
            return False
    return True


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (np.isfinite(last["dc_hi"]) and np.isfinite(last["dc_lo"])
            and np.isfinite(prev["dc_hi"]) and np.isfinite(prev["dc_lo"])):
        return flat("channel warming up")

    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")

    width = float(last["dc_hi"]) - float(last["dc_lo"])
    if width < float(p["min_width_atr"]) * a:
        return flat("channel too narrow")
    if width > float(p["max_width_atr"]) * a:
        return flat("channel too wide")

    buf = float(p["buf_atr"]) * a
    up = float(last["dc_hi"]) + buf
    dn = float(last["dc_lo"]) - buf
    if str(p["confirm"]) == "wick":
        brk_up = float(last["high"]) > up and float(prev["high"]) <= float(prev["dc_hi"])
        brk_dn = float(last["low"]) < dn and float(prev["low"]) >= float(prev["dc_lo"])
    else:
        brk_up = float(last["close"]) > up and float(prev["close"]) <= float(prev["dc_hi"])
        brk_dn = float(last["close"]) < dn and float(prev["close"]) >= float(prev["dc_lo"])

    d = str(p["dir"])
    if d == "long":
        brk_dn = False
    elif d == "short":
        brk_up = False
    if not (brk_up or brk_dn):
        return flat("no breakout")

    c = float(last["close"])
    ext = (c - up) if brk_up else (dn - c)
    if ext > float(p["max_ext_atr"]) * a:
        return flat("break already extended")

    direction = LONG if brk_up else SHORT
    if not _trend_ok(p["trend_mode"], direction, last):
        return flat("against trend filter")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    side = "up" if brk_up else "down"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"DC{int(p['lookback'])} break {side} "
                  f"(atr {a:.2f}, tp {tp_dist:.2f})")
