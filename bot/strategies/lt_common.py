"""Shared helpers for the lt_* trend / pullback / multi-timeframe library
(not a strategy itself: no add_indicators/evaluate/warmup here).

Design used by every lt_ module (all vectorised, all causal):
  add_indicators() writes
    lt_sig  int  +1 long / -1 short / 0 none, decided on that bar's close
    atr          cfg.atr_period ATR (stops: cfg.atr_sl_mult / cfg.atr_tp_mult)
    lt_xl / lt_xs  bool, optional: close a long / short at the next open
  evaluate() and exit_signal() only read the LAST row (fast, identical in
  live and backtest). A very wide cfg.atr_tp_mult + an exit column = trailing.
Daily features on intraday data come from COMPLETED server days only
(prior_daily: day D bars see days <= D-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import LONG, SHORT, Signal, atr, ema, flat, sma

_BPD = {"M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}


def bars_per_day(tf: str) -> int:
    return _BPD.get(str(tf).upper(), 24)


def side_mask(sig: pd.Series, side: str) -> pd.Series:
    if side == "long":
        return sig.where(sig > 0, 0)
    if side == "short":
        return sig.where(sig < 0, 0)
    return sig


def trend_ok(out: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """(up, down) vs EMA(period) on this timeframe = the factory's trend=1 baseline."""
    e = ema(out["close"], period)
    return out["close"] > e, out["close"] < e


def combine(long_c: pd.Series, short_c: pd.Series, side: str) -> pd.Series:
    sig = pd.Series(np.where(long_c.fillna(False), 1, np.where(short_c.fillna(False), -1, 0)),
                    index=long_c.index, dtype="int64")
    return side_mask(sig, side)


def add_exit(out: pd.DataFrame, mode: str, n: int) -> None:
    """Exit columns. mode:
      fixed  none (SL/TP/max_hold only)
      mid    close crosses the midpoint of the last n bars' high-low range
      opp    close breaks the opposite n-bar channel (Turtle exit)
      ma     close crosses EMA(n) against the trade (trend trail)
      rec    recovery: long exits once close > SMA(n), short once close < SMA(n)
    """
    c = out["close"]
    n = int(n)
    if mode == "fixed":
        return
    if mode == "mid":
        mid = (out["high"].rolling(n).max() + out["low"].rolling(n).min()) / 2.0
        out["lt_xl"], out["lt_xs"] = c < mid, c > mid
    elif mode == "opp":
        hh = out["high"].rolling(n).max().shift(1)
        ll = out["low"].rolling(n).min().shift(1)
        out["lt_xl"], out["lt_xs"] = c < ll, c > hh
    elif mode == "ma":
        m = ema(c, n)
        out["lt_xl"], out["lt_xs"] = c < m, c > m
    elif mode == "rec":
        m = sma(c, n)
        out["lt_xl"], out["lt_xs"] = c > m, c < m
    else:
        raise ValueError(f"unknown exit mode {mode!r}")


def finish(out: pd.DataFrame, sig: pd.Series, atr_period: int) -> pd.DataFrame:
    out["lt_sig"] = sig.astype("int64")
    out["atr"] = atr(out, atr_period)
    return out


def make_signal(w: pd.DataFrame, scfg, reason: str) -> Signal:
    s = int(w["lt_sig"].iat[-1])
    if s == 0:
        return flat("no setup")
    a = float(w["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = LONG if s > 0 else SHORT
    return Signal(d, float(w["close"].iat[-1]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a, reason)


def exit_hook(w: pd.DataFrame, direction: int) -> bool:
    col = "lt_xl" if direction == LONG else "lt_xs"
    if col not in w.columns:
        return False
    v = w[col].iat[-1]
    return bool(v) if v == v else False


def prior_daily(df: pd.DataFrame, feat_fn) -> pd.DataFrame:
    """Resample intraday bars to server days, compute feat_fn(daily OHLC) ->
    DataFrame of daily features, and give every intraday bar of day D the
    features of the last COMPLETED day before D (shift(1) on trading days)."""
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last"}).dropna(subset=["close"])
    feat = feat_fn(d).shift(1)
    day = df.index.normalize()
    return feat.reindex(day).set_axis(df.index)
