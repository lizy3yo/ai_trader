"""Multi-timeframe: daily trend + intraday (H1/M30) trigger (lt_ MTF family).

Daily trend from COMPLETED server days, resampled causally from the traded
timeframe inside the module (lt_common.prior_daily):
  d_mode sma  prior day close vs SMA(d_n) of daily closes
         mom  sign of the d_n-day return (prior completed day)
         dch  prior day close in the upper / lower half of the d_n-day range
Intraday trigger in the daily direction:
  brk  first close beyond the prior n-bar high / low
  pb   RSI(n) < thr in a daily uptrend (short: > 100 - thr)
Exit: fixed ATR SL/TP; "dflip" = daily trend no longer agrees; "rec" = close
back over SMA(exit_n) (for pb); or lt_common mid/opp/ma.
params: d_mode, d_n, trig, n, thr, side, exit, exit_n
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import donchian, rsi
from . import lt_common as C

DEFAULTS = {"d_mode": "sma", "d_n": 50, "trig": "brk", "n": 24, "thr": 30.0,
            "side": "long", "exit": "fixed", "exit_n": 24}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _daily_trend(d: pd.DataFrame, mode: str, n: int) -> pd.DataFrame:
    c = d["close"]
    if mode == "sma":
        s = c - c.rolling(n).mean()
    elif mode == "mom":
        s = c / c.shift(n) - 1.0
    elif mode == "dch":
        hh, ll = d["high"].rolling(n).max(), d["low"].rolling(n).min()
        s = c - (hh + ll) / 2.0
    else:
        raise ValueError(f"unknown d_mode {mode!r}")
    return pd.DataFrame({"dtr": s})


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    dtr = C.prior_daily(out, lambda d: _daily_trend(d, p["d_mode"], int(p["d_n"])))["dtr"]
    du, dd = dtr > 0, dtr < 0
    n = int(p["n"])
    if p["trig"] == "brk":
        hh, ll = donchian(out, n)
        lc = (c > hh) & (c.shift(1) <= hh.shift(1))
        sc = (c < ll) & (c.shift(1) >= ll.shift(1))
    elif p["trig"] == "pb":
        r = rsi(c, n)
        lc, sc = r < float(p["thr"]), r > 100.0 - float(p["thr"])
    else:
        raise ValueError(f"unknown trig {p['trig']!r}")
    if p["exit"] == "dflip":
        out["lt_xl"], out["lt_xs"] = ~du, ~dd
    else:
        C.add_exit(out, p["exit"], p["exit_n"])
    return C.finish(out, C.combine(lc & du, sc & dd, p["side"]), scfg.atr_period)


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bpd = C.bars_per_day(scfg.timeframe)
    return max((int(p["d_n"]) + 2) * bpd, 5 * int(p["n"]), int(p["exit_n"]), scfg.atr_period) + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg):
    p = _p(scfg)
    return C.make_signal(ind_window, scfg, f"daily {p['d_mode']}{p['d_n']} + {p['trig']}({p['n']})")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_hook(ind_window, direction)
