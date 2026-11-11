"""Pullback in trend (lt_ pullback family).

Trend = close vs EMA(cfg.ema_trend) on this timeframe (same gate as the
factory's trend=1 random baseline). Long when the trend is up AND price makes
a short-term pullback; mirror for shorts. Triggers:
  nlow  close below the lowest close of the prior n bars (n-bar low close)
  rsi   RSI(n) < thr (short: > 100 - thr)
  ma    first close below SMA(n) (short: above)
Exit: "rec" = recovery, close back above SMA(exit_n) (short: below), plus
cfg.max_hold_bars time stop and a wide ATR stop; or "fixed" ATR SL/TP.
params: trig, n, thr, side, exit (rec/fixed/ma/mid), exit_n
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import rsi, sma
from . import lt_common as C

DEFAULTS = {"trig": "nlow", "n": 5, "thr": 10.0, "side": "long", "exit": "rec", "exit_n": 5}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def pullback(out: pd.DataFrame, trig: str, n: int, thr: float) -> tuple[pd.Series, pd.Series]:
    """(long dip, short rally) trigger columns, causal."""
    c = out["close"]
    if trig == "nlow":
        prior = c.shift(1)
        return c < prior.rolling(n).min(), c > prior.rolling(n).max()
    if trig == "rsi":
        r = rsi(c, n)
        return r < thr, r > 100.0 - thr
    if trig == "ma":
        m = sma(c, n)
        return (c < m) & (c.shift(1) >= m.shift(1)), (c > m) & (c.shift(1) <= m.shift(1))
    raise ValueError(f"unknown trigger {trig!r}")


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    dip, rally = pullback(out, p["trig"], int(p["n"]), float(p["thr"]))
    tu, td = C.trend_ok(out, scfg.ema_trend)
    C.add_exit(out, p["exit"], p["exit_n"])
    return C.finish(out, C.combine(dip & tu, rally & td, p["side"]), scfg.atr_period)


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(2 * scfg.ema_trend, 5 * int(p["n"]), int(p["exit_n"]), scfg.atr_period) + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg):
    p = _p(scfg)
    return C.make_signal(ind_window, scfg, f"{p['trig']}({p['n']}) pullback in EMA{scfg.ema_trend} trend")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_hook(ind_window, direction)
