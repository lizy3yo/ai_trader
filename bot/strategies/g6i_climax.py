"""Volume-climax reversal at an N-bar extreme (g6i_).

A bar that prints a new `n`-bar high (low), on rvol >= vmin (time-of-day
normalised tick volume, g6i_common.rvol), and rejects it with a long wick
(upper wick >= wick x bar range and close in the lower half for a top;
mirror for a bottom) marks exhaustion: late buyers absorbed -> fade it.
Exits: ATR stop/target, hold_min cap (exit_signal), session.flat_at.
params: n, vmin, vdays, wick, hold_min, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import g6i_common as G
from . import ls_common as C

DEFAULTS = {"n": 20, "vmin": 2.0, "vdays": 20, "wick": 0.5, "hold_min": 120, "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    rv = G.rvol(out, int(p["vdays"]))
    out["rvol"] = rv
    n = int(p["n"])
    h, l, o, c = (out[k] for k in ("high", "low", "open", "close"))
    rng = (h - l).replace(0, np.nan)
    top = h > h.shift(1).rolling(n).max()
    bot = l < l.shift(1).rolling(n).min()
    up_w = (h - np.maximum(o, c)) / rng
    dn_w = (np.minimum(o, c) - l) / rng
    pos = (c - l) / rng
    vok = rv >= float(p["vmin"])
    w = float(p["wick"])
    short = top & (up_w >= w) & (pos < 0.5) & vok
    long_ = bot & (dn_w >= w) & (pos > 0.5) & vok
    out["ls_sig"] = np.where(long_ & ~short, 1.0, np.where(short & ~long_, -1.0, 0.0))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = C.params(scfg, DEFAULTS)
    return int(p["n"]) + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), "vol climax")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held,
                            G.hold_bars(p["hold_min"], C.tf_minutes(scfg)))
