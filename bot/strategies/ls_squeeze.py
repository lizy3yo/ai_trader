"""Bollinger-inside-Keltner squeeze breakout (ls_ library, volatility family).

Squeeze on bar i: BB(n, bb_k) lies inside KC(EMA n +/- kc_m x ATR n).
Armed while a squeeze of >= min_sq consecutive bars is ongoing or ended no
more than win bars ago. Trigger: the first close outside the Bollinger band
(previous close inside) -> trade the breakout side.
Exit: ATR stop/target; exit_mid=1 also exits when the close crosses back
through the band middle (SMA n) - use "exit_from_strategy": true.

params: n, bb_k, kc_m, min_sq, win, exit_mid, hold (bar cap, 0=off), side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr, ema
from . import ls_common as C

DEFAULTS = {"n": 20, "bb_k": 2.0, "kc_m": 1.5, "min_sq": 6, "win": 10, "exit_mid": 0,
            "hold": 0, "side": "both"}


def _run_len(flag: np.ndarray) -> np.ndarray:
    s = pd.Series(flag.astype(int))
    grp = (s != s.shift()).cumsum()
    return (s.groupby(grp).cumsum() * s).to_numpy()


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    n = int(p["n"])
    c = out["close"]
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    bu, bd = mid + float(p["bb_k"]) * sd, mid - float(p["bb_k"]) * sd
    km = ema(c, n)
    ka = atr(out, n)
    ku, kd = km + float(p["kc_m"]) * ka, km - float(p["kc_m"]) * ka
    sq = ((bu < ku) & (bd > kd)).to_numpy()
    run = _run_len(sq)
    q = sq & (run >= int(p["min_sq"]))
    idx = np.arange(len(out), dtype=float)
    last_q = pd.Series(np.where(q, idx, np.nan)).ffill().to_numpy()
    with np.errstate(invalid="ignore"):
        armed = (idx - last_q) <= int(p["win"])
        cv, buv, bdv = c.to_numpy(), bu.to_numpy(), bd.to_numpy()
        up = cv > buv
        dn = cv < bdv
        up_x = armed & up & ~np.r_[False, up[:-1]]
        dn_x = armed & dn & ~np.r_[False, dn[:-1]]
    out["ls_sig"] = np.where(up_x, 1.0, np.where(dn_x, -1.0, 0.0))
    if int(p["exit_mid"]):
        mv = mid.to_numpy()
        with np.errstate(invalid="ignore"):
            out["ls_xl"] = (cv < mv).astype(float)
            out["ls_xs"] = (cv > mv).astype(float)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = C.params(scfg, DEFAULTS)
    return 5 * max(scfg.atr_period, int(p["n"])) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), "SQZ")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held, int(p["hold"]))
