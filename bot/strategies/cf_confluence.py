"""Confluence engine: TREND x SETUP x TRIGGER (user request, 2026-11-08).

A trade needs all three layers to agree on the same side. Every building block
was fixed BEFORE testing (4 x 4 x 4 = 64 combinations):

TREND (direction of the bar)
  ema200     close above / below EMA200
  stack      EMA20 > EMA50 > EMA100 > EMA200 (mirror for shorts)
  structure  last two CONFIRMED swing highs and lows (pivot 5 bars each side):
             higher high + higher low = up; lower high + lower low = down
  supertrend Supertrend(ATR 10, mult 3) direction
SETUP (where; active if it occurred on this bar or the previous 2)
  ema        pullback: low touches EMA20 (or EMA50) while the close stays above
             EMA50 (longs; mirror for shorts)
  bos        retest of a break of structure: within 20 bars after a close
             above the last confirmed swing high, price comes back within
             0.5 x ATR of that broken level
  sr         horizontal support / resistance: low within 0.5 x ATR of the last
             confirmed swing low (support) without closing below it
  fib        pullback into the 50-61.8% zone of the last confirmed swing low ->
             highest high since
TRIGGER (when; on the closed bar)
  stoch      Stochastic %K(14,3) crosses above %D while %K < 40
  candle     bullish engulfing or hammer (lower wick >= 2x body, small upper wick)
  macd       MACD(12,26) line crosses above its 9 signal
  prevhigh   close above the previous bar's high
Shorts are exact mirrors. Entry next open; stop cfg.atr_sl_mult x ATR,
target cfg.atr_tp_mult x ATR (study default 2 / 4 = 1:2, the user's preference).
All swing points are used only after their confirmation bar (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat

DEFAULTS = {"trend": "ema200", "setup": "ema", "trigger": "prevhigh", "side": "both",
            "pivot": 5, "setup_win": 3}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _supertrend(h, l, c, a10, mult=3.0):
    n = len(c)
    d = np.zeros(n, dtype=np.int8)
    up = np.full(n, np.nan); dn = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(a10[i]):
            continue
        mid = (h[i] + l[i]) / 2
        bu, bd = mid - mult * a10[i], mid + mult * a10[i]
        if i == 0 or not np.isfinite(up[i - 1]):
            up[i], dn[i], d[i] = bu, bd, 1
            continue
        up[i] = max(bu, up[i - 1]) if c[i - 1] > up[i - 1] else bu
        dn[i] = min(bd, dn[i - 1]) if c[i - 1] < dn[i - 1] else bd
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < up[i] else 1
        else:
            d[i] = 1 if c[i] > dn[i] else -1
    return d


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(c)
    cs = out["close"]
    e20, e50, e100, e200 = (ema(cs, k).to_numpy() for k in (20, 50, 100, 200))
    a = atr(out, scfg.atr_period).to_numpy()
    a10 = atr(out, 10).to_numpy()
    k = int(p["pivot"])

    # ---- confirmed swing points (pivot at j known only at j + k) ----
    last_sh = np.full(n, np.nan); prev_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan); prev_sl = np.full(n, np.nan)
    hh_since_sl = np.full(n, np.nan); ll_since_sh = np.full(n, np.nan)
    lsh = psh = lsl = psl = np.nan
    hh = ll = np.nan
    for i in range(n):
        j = i - k
        if j - k >= 0:
            win_h, win_l = h[j - k: i + 1], l[j - k: i + 1]
            if h[j] == win_h.max():
                psh, lsh = lsh, h[j]
                ll = l[j:i + 1].min()          # lowest low since that swing high
            if l[j] == win_l.min():
                psl, lsl = lsl, l[j]
                hh = h[j:i + 1].max()          # highest high since that swing low
        if np.isfinite(hh):
            hh = max(hh, h[i])
        if np.isfinite(ll):
            ll = min(ll, l[i])
        last_sh[i], prev_sh[i], last_sl[i], prev_sl[i] = lsh, psh, lsl, psl
        hh_since_sl[i], ll_since_sh[i] = hh, ll

    # ---- TREND ----
    t = p["trend"]
    if t == "ema200":
        tr = np.where(c > e200, 1, np.where(c < e200, -1, 0))
    elif t == "stack":
        tr = np.where((e20 > e50) & (e50 > e100) & (e100 > e200), 1,
                      np.where((e20 < e50) & (e50 < e100) & (e100 < e200), -1, 0))
    elif t == "structure":
        with np.errstate(invalid="ignore"):
            tr = np.where((last_sh > prev_sh) & (last_sl > prev_sl), 1,
                          np.where((last_sh < prev_sh) & (last_sl < prev_sl), -1, 0))
    else:  # supertrend
        tr = _supertrend(h, l, c, a10)

    # ---- SETUP (raw on the bar, then 'active within setup_win bars') ----
    s = p["setup"]
    with np.errstate(invalid="ignore"):
        if s == "ema":
            sl_ = ((l <= e20) | (l <= e50)) & (c > e50)
            ss_ = ((h >= e20) | (h >= e50)) & (c < e50)
        elif s == "bos":
            # break = close crosses the last confirmed swing high; remember level 20 bars
            sl_ = np.zeros(n, bool); ss_ = np.zeros(n, bool)
            lvl_u = lvl_d = np.nan; age_u = age_d = 999
            for i in range(1, n):
                if np.isfinite(last_sh[i]) and c[i] > last_sh[i] >= c[i - 1]:
                    lvl_u, age_u = last_sh[i], 0
                if np.isfinite(last_sl[i]) and c[i] < last_sl[i] <= c[i - 1]:
                    lvl_d, age_d = last_sl[i], 0
                if 0 < age_u <= 20 and l[i] <= lvl_u + 0.5 * a[i] and c[i] > lvl_u - 0.5 * a[i]:
                    sl_[i] = True
                if 0 < age_d <= 20 and h[i] >= lvl_d - 0.5 * a[i] and c[i] < lvl_d + 0.5 * a[i]:
                    ss_[i] = True
                age_u += 1; age_d += 1
        elif s == "sr":
            sl_ = (l <= last_sl + 0.5 * a) & (c >= last_sl)
            ss_ = (h >= last_sh - 0.5 * a) & (c <= last_sh)
        else:  # fib
            rng_u = hh_since_sl - last_sl
            sl_ = (rng_u > 0) & (l <= hh_since_sl - 0.5 * rng_u) & (c >= hh_since_sl - 0.618 * rng_u)
            rng_d = last_sh - ll_since_sh
            ss_ = (rng_d > 0) & (h >= ll_since_sh + 0.5 * rng_d) & (c <= ll_since_sh + 0.618 * rng_d)
    w = int(p["setup_win"])
    setup_l = pd.Series(sl_).rolling(w, min_periods=1).max().to_numpy().astype(bool)
    setup_s = pd.Series(ss_).rolling(w, min_periods=1).max().to_numpy().astype(bool)

    # ---- TRIGGER ----
    g = p["trigger"]
    if g == "stoch":
        ll14 = pd.Series(l).rolling(14).min(); hh14 = pd.Series(h).rolling(14).max()
        kraw = 100 * (pd.Series(c) - ll14) / (hh14 - ll14).replace(0, np.nan)
        kk = kraw.rolling(3).mean(); dd = kk.rolling(3).mean()
        kk, dd = kk.to_numpy(), dd.to_numpy()
        kp, dp = np.r_[np.nan, kk[:-1]], np.r_[np.nan, dd[:-1]]
        with np.errstate(invalid="ignore"):
            gl = (kk > dd) & (kp <= dp) & (kk < 40)
            gs = (kk < dd) & (kp >= dp) & (kk > 60)
    elif g == "candle":
        body = np.abs(c - o); rng = h - l
        po, pc = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
        with np.errstate(invalid="ignore"):
            eng_l = (c > o) & (pc < po) & (c >= po) & (o <= pc)
            eng_s = (c < o) & (pc > po) & (c <= po) & (o >= pc)
            low_w = np.minimum(o, c) - l; up_w = h - np.maximum(o, c)
            ham = (rng > 0) & (low_w >= 2 * body) & (up_w <= body)
            star = (rng > 0) & (up_w >= 2 * body) & (low_w <= body)
        gl, gs = eng_l | ham, eng_s | star
    elif g == "macd":
        m = ema(cs, 12) - ema(cs, 26); sg = ema(m, 9)
        diff = (m - sg).to_numpy(); prev = np.r_[np.nan, diff[:-1]]
        with np.errstate(invalid="ignore"):
            gl, gs = (diff > 0) & (prev <= 0), (diff < 0) & (prev >= 0)
    else:  # prevhigh
        ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
        with np.errstate(invalid="ignore"):
            gl, gs = c > ph, c < pl

    sig = np.zeros(n, dtype=np.int8)
    if p["side"] in ("long", "both"):
        sig[(tr == 1) & setup_l & gl] = 1
    if p["side"] in ("short", "both"):
        sig[(tr == -1) & setup_s & gs & (sig == 0)] = -1
    sig[:250] = 0
    out["atr"] = a
    out["cf_sig"] = sig
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 260


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    last = ind_window.iloc[-1]
    d = int(last["cf_sig"])
    a = float(last["atr"])
    if d == 0 or not np.isfinite(a) or a <= 0:
        return flat("no confluence")
    p = _p(scfg)
    return Signal(LONG if d == 1 else SHORT, float(last["close"]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, f"{p['trend']}+{p['setup']}+{p['trigger']}")
