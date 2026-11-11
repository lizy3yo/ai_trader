"""Tick-volume-confirmed breakouts and quiet-to-active transitions (g6i_).

rvol = tick volume / same-time-of-day mean of the prior `vdays` sessions
(g6i_common.rvol), so the US-open volume burst is not mistaken for news.

mode "brk": close breaks the prior `n`-bar high (low) for the first time
            (previous close was inside) AND rvol of that bar >= vmin
            -> follow the break. vmin = 0 is the no-volume control.
mode "q2a": quiet-to-active: mean rvol of the prior `q` bars <= qmax, this
            bar rvol >= vmin and |close-open| >= body x ATR -> follow the
            bar's direction (a dormant market waking up with participation).
Exits: ATR stop/target (cfg.atr_sl_mult / atr_tp_mult), hold_min cap
(exit_signal; "exit_from_strategy": true), session.flat_at before the close.
Entry hours come from session.window. Generic across markets.
params: mode, n, vmin, vdays, q, qmax, body, hold_min, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import g6i_common as G
from . import ls_common as C

DEFAULTS = {"mode": "brk", "n": 20, "vmin": 2.0, "vdays": 20, "q": 8, "qmax": 0.7,
            "body": 0.8, "hold_min": 0, "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    rv = G.rvol(out, int(p["vdays"]))
    out["rvol"] = rv
    c, o = out["close"], out["open"]
    vok = (rv >= float(p["vmin"])) if float(p["vmin"]) > 0 else np.ones(len(out), bool)
    if p["mode"] == "brk":
        n = int(p["n"])
        hi = out["high"].shift(1).rolling(n).max()
        lo = out["low"].shift(1).rolling(n).min()
        up = (c > hi) & (c.shift(1) <= hi.shift(1))
        dn = (c < lo) & (c.shift(1) >= lo.shift(1))
        sig = np.where(up & vok, 1, np.where(dn & vok, -1, 0))
    elif p["mode"] == "q2a":
        q = int(p["q"])
        quiet = pd.Series(rv).shift(1).rolling(q, min_periods=q).mean().to_numpy() <= float(p["qmax"])
        body = (c - o).to_numpy()
        big = np.abs(body) >= float(p["body"]) * out["atr"].to_numpy()
        sig = np.where(quiet & vok & big, np.sign(body), 0)
    else:
        raise ValueError(f"g6i_volbrk: unknown mode {p['mode']!r}")
    out["ls_sig"] = np.nan_to_num(np.asarray(sig, float))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = C.params(scfg, DEFAULTS)
    return max(int(p["n"]), int(p["q"])) + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"volbrk {p['mode']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held,
                            G.hold_bars(p["hold_min"], C.tf_minutes(scfg)))
