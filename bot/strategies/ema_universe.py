"""EMA universe under the user's framework (2026-11-09).

trend   close above EMA(trend_n) -> longs only; below -> shorts only
entry   "cross":    EMA(fast) crosses EMA(slow) in the trend direction
        "pullback": the previous bar closed on the wrong side of EMA(pb_n)
                    (a pullback through it) and this bar closes back on the
                    trend side of it
stop    cfg.atr_sl_mult x ATR(14)   (1 / 1.5 / 2 / 3)
target  cfg.atr_tp_mult x ATR       (1.5 / 2 / 3 x the stop)
Only the stop or the target closes a trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat

DEFAULTS = {"trend_n": 200, "entry": "cross", "fast": 20, "slow": 50, "pb_n": 20, "side": "both"}


def _p(scfg: StrategyCfg) -> dict:
    p = {**DEFAULTS, **scfg.params}
    if "pairs" in p:                      # "fast/slow" string from the study grid
        f, s = str(p["pairs"]).split("/")
        p["fast"], p["slow"] = int(f), int(s)
    return p


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    tr = ema(c, int(p["trend_n"])).to_numpy()
    cv = c.to_numpy()
    up, dn = cv > tr, cv < tr
    if p["entry"] == "cross":
        f, s = ema(c, int(p["fast"])).to_numpy(), ema(c, int(p["slow"])).to_numpy()
        d = f - s
        dp = np.r_[np.nan, d[:-1]]
        with np.errstate(invalid="ignore"):
            el, es = (d > 0) & (dp <= 0), (d < 0) & (dp >= 0)
    else:
        e = ema(c, int(p["pb_n"])).to_numpy()
        cp, ep = np.r_[np.nan, cv[:-1]], np.r_[np.nan, e[:-1]]
        with np.errstate(invalid="ignore"):
            el, es = (cp < ep) & (cv > e), (cp > ep) & (cv < e)
    sig = np.zeros(len(cv), dtype=np.int8)
    if p["side"] in ("long", "both"):
        sig[up & el] = 1
    if p["side"] in ("short", "both"):
        sig[dn & es & (sig == 0)] = -1
    out["atr"] = atr(out, scfg.atr_period)
    out["eu_sig"] = sig
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return 3 * max(int(p["trend_n"]), int(p["slow"]), int(p["pb_n"])) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    last = ind_window.iloc[-1]
    d, a = int(last["eu_sig"]), float(last["atr"])
    if d == 0 or not np.isfinite(a) or a <= 0:
        return flat("no signal")
    return Signal(LONG if d == 1 else SHORT, float(last["close"]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, "ema universe")
