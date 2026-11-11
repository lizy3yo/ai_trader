"""Volatility-regime switch (ls_ library, volatility family).

Regime = rolling percentile rank of ATR% (ATR / close) over rank_n bars
(causal rolling rank, includes the current bar only).
  high vol (rank >= hi_q): trend rule - first close beyond the prior dc_n-bar
                           Donchian channel, trade the breakout
  low vol  (rank <= lo_q): fade rule - first close outside BB(bb_n, bb_k),
                           trade back toward the mean
mode "switch" uses both, "trend" / "fade" only one leg (to see which part
carries any edge). Exits: ATR stop/target, optional bar cap `hold`
(set "exit_from_strategy": true when hold > 0).

params: rank_n, hi_q, lo_q, dc_n, bb_n, bb_k, mode, hold, side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr, donchian
from . import ls_common as C

DEFAULTS = {"rank_n": 500, "hi_q": 0.6, "lo_q": 0.4, "dc_n": 20, "bb_n": 20, "bb_k": 2.0,
            "mode": "switch", "hold": 0, "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    c = out["close"]
    rk = (out["atr"] / c).rolling(int(p["rank_n"])).rank(pct=True).to_numpy()
    hh, ll = donchian(out, int(p["dc_n"]))
    m = c.rolling(int(p["bb_n"])).mean()
    sd = c.rolling(int(p["bb_n"])).std(ddof=0)
    bu = (m + float(p["bb_k"]) * sd).to_numpy()
    bd = (m - float(p["bb_k"]) * sd).to_numpy()
    cv, hv, lv = c.to_numpy(), hh.to_numpy(), ll.to_numpy()
    pc = np.r_[np.nan, cv[:-1]]
    with np.errstate(invalid="ignore"):
        hi = rk >= float(p["hi_q"])
        lo = rk <= float(p["lo_q"])
        t_up = hi & (cv > hv) & ~(pc > np.r_[np.nan, hv[:-1]])
        t_dn = hi & (cv < lv) & ~(pc < np.r_[np.nan, lv[:-1]])
        f_up = lo & (cv < bd) & ~(pc < np.r_[np.nan, bd[:-1]])   # oversold -> long
        f_dn = lo & (cv > bu) & ~(pc > np.r_[np.nan, bu[:-1]])   # overbought -> short
    mode = str(p["mode"])
    use_t, use_f = mode in ("switch", "trend"), mode in ("switch", "fade")
    lng = (t_up & use_t) | (f_up & use_f)
    sht = (t_dn & use_t) | (f_dn & use_f)
    out["ls_sig"] = np.where(lng & ~sht, 1.0, np.where(sht & ~lng, -1.0, 0.0))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = C.params(scfg, DEFAULTS)
    return max(int(p["rank_n"]), 5 * scfg.atr_period, int(p["dc_n"]), int(p["bb_n"])) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"VOLREG {p['mode']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held, int(p["hold"]))
