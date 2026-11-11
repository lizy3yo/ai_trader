"""Short-horizon mean reversion (lm_ library, generic across markets).

One module, five classic entry rules (mode), each with its own published exit;
all signals are decided on the CLOSED bar and filled at the next open.
  rsi2    Connors RSI(2): long RSI(rsi_p) < rsi_th; exit close > SMA(exit_ma)
          or RSI > exit_rsi.                         (short: RSI > 100 - rsi_th)
  bb      Bollinger re-entry: previous close below SMA(bb_n) - bb_k*sd, this
          close back above it; exit close > SMA(bb_n) (middle band).
  nlow    N-bar low: close = lowest close of the last nlow_n bars; exit on the
          first up close (close > previous close).
  ibs     internal bar strength (close-low)/(high-low) < ibs_th; exit when
          IBS > ibs_exit.                            (short: IBS > 1 - ibs_th)
  zscore  z = (close - SMA(z_n)) / sd(z_n) < -z_th; exit when z >= -z_exit
          (z_exit 0 = back at the mean).
Every mode also exits after max_hold bars and keeps the shared ATR stop
(cfg.atr_sl_mult; set cfg.atr_tp_mult wide so the rule, not the TP, exits).
trend = 1: longs only above EMA(cfg.ema_trend), shorts only below it (the same
filter the random baseline uses with "trend": 1).
side: "long" / "short" / "both".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, rsi, sma

DEFAULTS = {"mode": "rsi2", "side": "long", "trend": 1, "max_hold": 10,
            "rsi_p": 2, "rsi_th": 10.0, "exit_ma": 5, "exit_rsi": 70.0,
            "bb_n": 20, "bb_k": 2.0, "nlow_n": 5,
            "ibs_th": 0.2, "ibs_exit": 0.8, "z_n": 20, "z_th": 2.0, "z_exit": 0.0}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    out["atr"] = atr(out, scfg.atr_period)
    mode = p["mode"]
    if mode == "rsi2":
        r = rsi(c, int(p["rsi_p"]))
        m = sma(c, int(p["exit_ma"]))
        th = float(p["rsi_th"])
        el, es = r < th, r > 100.0 - th
        xl = (c > m) | (r > float(p["exit_rsi"]))
        xs = (c < m) | (r < 100.0 - float(p["exit_rsi"]))
    elif mode == "bb":
        n, k = int(p["bb_n"]), float(p["bb_k"])
        mid, sd = sma(c, n), rolling_std(c, n)
        lo, up = mid - k * sd, mid + k * sd
        el = (c.shift(1) < lo.shift(1)) & (c > lo)
        es = (c.shift(1) > up.shift(1)) & (c < up)
        xl, xs = c > mid, c < mid
    elif mode == "nlow":
        n = int(p["nlow_n"])
        el = c <= c.rolling(n).min()
        es = c >= c.rolling(n).max()
        xl, xs = c > c.shift(1), c < c.shift(1)
    elif mode == "ibs":
        rng = (h - l).replace(0.0, np.nan)
        ibs = ((c - l) / rng).fillna(0.5)
        th, ex = float(p["ibs_th"]), float(p["ibs_exit"])
        el, es = ibs < th, ibs > 1.0 - th
        xl, xs = ibs > ex, ibs < 1.0 - ex
    elif mode == "zscore":
        n = int(p["z_n"])
        z = (c - sma(c, n)) / rolling_std(c, n).replace(0.0, np.nan)
        th, ex = float(p["z_th"]), float(p["z_exit"])
        el, es = z < -th, z > th
        xl, xs = z >= -ex, z <= ex
    else:
        raise ValueError(f"lm_meanrev: unknown mode {mode!r}")
    if int(p["trend"]):
        e = ema(c, scfg.ema_trend)
        el, es = el & (c > e), es & (c < e)
    side = p["side"]
    out["mr_long"] = el.fillna(False).astype(bool) & (side in ("long", "both"))
    out["mr_short"] = es.fillna(False).astype(bool) & (side in ("short", "both"))
    out["mr_xlong"] = xl.fillna(False).astype(bool)
    out["mr_xshort"] = xs.fillna(False).astype(bool)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = max(int(p["bb_n"]), int(p["z_n"]), int(p["nlow_n"]), int(p["exit_ma"]), scfg.atr_period) + 10
    if int(p["trend"]):
        w = max(w, 2 * scfg.ema_trend + 10)
    return w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    lg, sh = bool(ind_window["mr_long"].iat[-1]), bool(ind_window["mr_short"].iat[-1])
    if not (lg or sh):
        return flat("no setup")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = LONG if lg else SHORT
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, f"mr {_p(scfg)['mode']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    if bars_held >= int(_p(scfg)["max_hold"]):
        return True
    col = "mr_xlong" if direction == LONG else "mr_xshort"
    return bool(ind_window[col].iat[-1])
