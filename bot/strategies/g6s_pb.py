"""H1 pullback INSIDE the Donchian trend regime (round 6, g6s_; long default).

Regime (the state the 360-bar Donchian trades in):
  regime=ema    close above EMA(cfg.ema_trend)
  regime=donch  close above EMA(cfg.ema_trend) AND a close above the prior
                `reg_n`-bar high happened within the last `reg_bars` bars
Entry: the FIRST close that is the lowest close of the last `dip_n` bars
(a 1-2 day dip on H1) while the regime holds. Exit: recovery = close back
above SMA(dip_n) (exit_signal), else ATR SL/TP or cfg.max_hold_bars.
Round 5's failed pullback was H4/D1 with no breakout-regime gate.
params: dip_n, regime, reg_n, reg_bars, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, sma

DEFAULTS = {"dip_n": 24, "regime": "donch", "reg_n": 360, "reg_bars": 240, "side": "long"}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    out["atr"] = atr(out, scfg.atr_period)
    e = ema(c, scfg.ema_trend)
    n = int(p["dip_n"])
    lo_c = c.rolling(n).min()
    hi_c = c.rolling(n).max()
    dip_l = (c <= lo_c) & (c.shift(1) > lo_c.shift(1))
    dip_s = (c >= hi_c) & (c.shift(1) < hi_c.shift(1))
    up_reg, dn_reg = c > e, c < e
    if p["regime"] == "donch":
        hh, ll = donchian(out, int(p["reg_n"]))
        rb = int(p["reg_bars"])
        brk_u = (c > hh).astype(float).rolling(rb, min_periods=1).max() > 0
        brk_d = (c < ll).astype(float).rolling(rb, min_periods=1).max() > 0
        up_reg, dn_reg = up_reg & brk_u, dn_reg & brk_d
    side = p["side"]
    L = (dip_l & up_reg).fillna(False) & (side != "short")
    S = (dip_s & dn_reg).fillna(False) & (side != "long")
    out["g_sig"] = np.where(L, 1, np.where(S, -1, 0)).astype("int64")
    m = sma(c, n)
    out["g_xl"], out["g_xs"] = c > m, c < m
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(5 * scfg.ema_trend, int(p["reg_n"]) + int(p["reg_bars"]), int(p["dip_n"])) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    s = int(ind_window["g_sig"].iat[-1])
    if s == 0:
        return flat("no dip")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = LONG if s > 0 else SHORT
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, f"{_p(scfg)['dip_n']}-bar dip in trend")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    col = "g_xl" if direction == LONG else "g_xs"
    return bool(ind_window[col].iat[-1])
