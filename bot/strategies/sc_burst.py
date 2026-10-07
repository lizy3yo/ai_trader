"""Momentum-burst micro-scalp on M5 (scalping family, prefix sc_).

Idea: N consecutive strong same-direction closes with a meaningful net move
mark a short-horizon momentum burst. mode="follow" enters the continuation
at the next open with tight ATR exits; mode="fade" bets the burst exhausts
and enters against it.

Causality: every indicator column at row i uses only rows <= i (per-bar OHLC
arithmetic plus EWMA ATR). evaluate() reads only the last n_burst+1 CLOSED
rows of the window. No rolling extremes, no full-frame statistics.

Params (scfg.params):
    n_burst   : number of consecutive same-direction closes (2-3)
    body_atr  : each burst bar's |close-open| must be >= this x ATR
    net_atr   : |net move across the burst| must be >= this x ATR
    expand    : if 1, last burst bar's range must exceed the first's
    mode      : "follow" (continuation) or "fade" (exhaustion reversal)
    sides     : "both" / "long" / "short"
    min_atr   : skip if ATR below this (dead tape — spread dominates), in $
    max_atr   : skip if ATR above this (news blowout), in $

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "n_burst": 3,
    "body_atr": 0.30,
    "net_atr": 0.80,
    "expand": 0,
    "mode": "follow",
    "sides": "both",
    "min_atr": 0.0,
    "max_atr": 99.0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["body"] = out["close"] - out["open"]
    out["rng"] = out["high"] - out["low"]
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    return max(5 * scfg.atr_period, int(p["n_burst"]) + 2) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")
    if a > float(p["max_atr"]):
        return flat("ATR above cap")

    n = int(p["n_burst"])
    burst = ind_window.iloc[-n:]
    bodies = burst["body"].to_numpy(dtype=float)
    rngs = burst["rng"].to_numpy(dtype=float)
    if not np.all(np.isfinite(bodies)):
        return flat("bad bars")

    body_min = float(p["body_atr"]) * a
    all_up = bool(np.all(bodies >= body_min))
    all_dn = bool(np.all(bodies <= -body_min))
    if not (all_up or all_dn):
        return flat("no burst")

    net = float(burst["close"].iloc[-1]) - float(burst["open"].iloc[0])
    if abs(net) < float(p["net_atr"]) * a:
        return flat("net move too small")
    if int(p["expand"]) and n >= 2 and not rngs[-1] > rngs[0]:
        return flat("no range expansion")

    burst_dir = LONG if all_up else SHORT
    fade = str(p["mode"]) == "fade"
    direction = -burst_dir if fade else burst_dir

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    c = float(last["close"])
    tag = "fade" if fade else "follow"
    word = "up" if burst_dir == LONG else "down"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"burst {tag}: {n} {word} closes, net {net:+.2f} vs ATR {a:.2f}")
