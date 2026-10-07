"""Weighted-vote SIGNAL ENSEMBLE (ens_ family).

Several individually-weak, deliberately DIFFERENT directional signals each cast
a vote in {-1, 0, +1} on the last closed bar. The votes are combined into a
normalised score

    score = sum(w_i * v_i) / sum(w_i for enabled i)      ->  score in [-1, +1]

and a trade is only taken when enough of the committee agrees, i.e.
|score| >= k_thresh. Setting a component's weight to 0 removes it from both
the numerator AND the denominator, so k_thresh keeps its "fraction of the
committee" meaning whatever subset is enabled.

Components (all causal, all state-based unless noted)
  v_ema   sign(EMA(ema_fast) - EMA(ema_slow))                 fast/slow regime
  v_trend sign(close - EMA(ema_trend))                        slow trend
  v_dc    +1 close > prior dc_lb-bar high, -1 below prior low, else 0
  v_rsi   rsi_mode="mom": +1 rsi > 50+rsi_band, -1 rsi < 50-rsi_band
          rsi_mode="rev": +1 rsi < rsi_os (oversold=buy), -1 rsi > rsi_ob
  v_mom   sign(close - close.shift(roc_n))                    raw momentum
  v_ses   session bias: sign(close - today's FIRST bar open), but only while
          the bar's hour is inside [ses_h0, ses_h1); 0 elsewhere

Gate (not a vote)
  vol_q   require ATR >= vol_q quantile of the PRIOR vol_win ATRs (0 = off)

entry_mode
  "cross"  enter only on the bar where |score| first reaches k_thresh
           (score was on the wrong side of the threshold on the prior bar)
  "state"  enter on any bar where the score condition holds (the backtester's
           one-position rule + cooldown_bars spaces the trades) -> more trades

params: w_ema, w_trend, w_dc, w_rsi, w_mom, w_ses, k_thresh, dc_lb, roc_n,
        rsi_mode, rsi_band, rsi_ob, rsi_os, ses_h0, ses_h1, vol_q, vol_win,
        entry_mode, sides
Shared knobs used from StrategyCfg: ema_fast, ema_slow, ema_trend, rsi_period,
atr_period, atr_sl_mult, atr_tp_mult, cooldown_bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, rsi, valid_atr
from .hyb_common import atr_pct_ok

DEFAULTS = {
    "w_ema": 1.0,
    "w_trend": 1.0,
    "w_dc": 1.0,
    "w_rsi": 1.0,
    "w_mom": 0.0,
    "w_ses": 0.0,
    "k_thresh": 0.5,
    "dc_lb": 20,
    "roc_n": 10,
    "rsi_mode": "mom",
    "rsi_band": 10.0,
    "rsi_ob": 70.0,
    "rsi_os": 30.0,
    "ses_h0": 9,
    "ses_h1": 12,
    "vol_q": 0.0,
    "vol_win": 200,
    "entry_mode": "cross",
    "sides": "both",
}

_WEIGHT_KEYS = ("w_ema", "w_trend", "w_dc", "w_rsi", "w_mom", "w_ses")


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    close = out["close"]

    out["atr"] = atr(out, scfg.atr_period)

    # ---- component votes (each in {-1, 0, +1}) --------------------------
    ema_f = ema(close, scfg.ema_fast)
    ema_s = ema(close, scfg.ema_slow)
    ema_t = ema(close, scfg.ema_trend)
    out["v_ema"] = np.sign(ema_f - ema_s)
    out["v_trend"] = np.sign(close - ema_t)

    hh, ll = donchian(out, int(p["dc_lb"]))          # already .shift(1)-ed
    out["v_dc"] = np.where(close > hh, 1.0, np.where(close < ll, -1.0, 0.0))
    out["v_dc"] = pd.Series(out["v_dc"], index=out.index).where(hh.notna() & ll.notna())

    r = rsi(close, scfg.rsi_period)
    if str(p["rsi_mode"]).lower() == "rev":
        out["v_rsi"] = np.where(r < float(p["rsi_os"]), 1.0,
                                np.where(r > float(p["rsi_ob"]), -1.0, 0.0))
    else:
        band = float(p["rsi_band"])
        out["v_rsi"] = np.where(r > 50.0 + band, 1.0,
                                np.where(r < 50.0 - band, -1.0, 0.0))

    n = int(p["roc_n"])
    out["v_mom"] = np.sign(close - close.shift(n))

    # session bias: today's FIRST bar open is known from that first bar on,
    # so transform("first") is causal (a truncated frame yields the same value)
    day = out.index.normalize()
    day_open = out["open"].groupby(day).transform("first")
    hours = out.index.hour
    in_ses = (hours >= int(p["ses_h0"])) & (hours < int(p["ses_h1"]))
    out["v_ses"] = np.where(in_ses, np.sign(close - day_open), 0.0)

    # ---- weighted, normalised committee score ---------------------------
    weights = {k: float(p[k]) for k in _WEIGHT_KEYS}
    denom = sum(w for w in weights.values() if w > 0)
    if denom <= 0:
        out["score"] = np.nan
    else:
        acc = pd.Series(0.0, index=out.index)
        for key, w in weights.items():
            if w <= 0:
                continue
            col = "v_" + key[2:]
            acc = acc + w * out[col].astype(float)
        out["score"] = acc / denom

    out["vol_ok"] = atr_pct_ok(out["atr"], int(p["vol_win"]), float(p["vol_q"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need_vol = int(p["vol_win"]) + 1 if float(p["vol_q"]) > 0 else 0
    return max(
        5 * scfg.ema_trend, 5 * scfg.ema_slow, scfg.rsi_period, scfg.atr_period,
        int(p["dc_lb"]) + 2, int(p["roc_n"]) + 2, need_vol,
    ) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    s, s_prev = last["score"], prev["score"]
    if not np.isfinite(s):
        return flat("score not formed")
    if not bool(last["vol_ok"]):
        return flat("ATR below vol gate")

    k = float(p["k_thresh"])
    s = float(s)
    cross = str(p["entry_mode"]).lower() == "cross"
    if cross and not np.isfinite(s_prev):
        return flat("prev score not formed")
    sp = float(s_prev) if np.isfinite(s_prev) else 0.0

    long_ok = s >= k and (not cross or sp < k)
    short_ok = s <= -k and (not cross or sp > -k)
    if not (long_ok or short_ok):
        return flat(f"committee score {s:+.2f} vs k {k:.2f}")

    direction = LONG if long_ok else SHORT
    sides = str(p["sides"]).lower()
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    a = float(last["atr"])
    return Signal(
        direction, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
        f"ensemble score {s:+.2f} (k={k:.2f}) "
        f"[ema {last['v_ema']:+.0f} trend {last['v_trend']:+.0f} dc {last['v_dc']:+.0f} "
        f"rsi {last['v_rsi']:+.0f} mom {last['v_mom']:+.0f} ses {last['v_ses']:+.0f}]",
    )
