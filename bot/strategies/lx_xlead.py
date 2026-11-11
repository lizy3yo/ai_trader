"""CROSS-MARKET lead/lag signal (family: crossmkt).

A partner market's recent move, measured in its own volatility units, is used
to predict the traded market's next bars:

    u = log(P_t / P_{t-k}) / (sd_1bar(P, vol_win) * sqrt(k))      partner move
    v = log(C_t / C_{t-k}) / (sd_1bar(C, vol_win) * sqrt(k))      own move

mode "lead": trade sign(pol * u) when |u| >= thr        (partner leads)
mode "gap" : g = pol * u - v; trade sign(g) when |g| >= thr
             (own market lagged the partner's move -> catch-up)
pol = +1 "follow" / -1 "fade" (e.g. EURUSD up = USD weak -> metals bid: follow).
Signals only on bars with an exact-stamp partner bar; first bar of the
condition only (cross). Exit: cfg.max_hold_bars (time) or ATR SL/TP.

partner: a market name ("eurusd", "us500cash", ...) or a pair key ("crypto",
"metals", "indices") = the other leg of that pair. If the partner resolves to
the traded market itself the module stays flat.

LIVE: needs the partner symbol's closed MT5 bars on the same timeframe.
params: partner, mode, k, thr, pol, vol_win, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr
from . import lx_common as lc

DEFAULTS = {
    "partner": "eurusd",
    "mode": "lead",
    "k": 1,
    "thr": 1.5,
    "pol": "follow",
    "vol_win": 100,
    "side": "both",
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _norm_move(logp: pd.Series, k: int, win: int) -> pd.Series:
    r1 = logp.diff()
    sd = r1.rolling(win, min_periods=win).std(ddof=0).replace(0.0, np.nan)
    return (logp - logp.shift(k)) / (sd * np.sqrt(k))


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    partner = lc.resolve_partner(out, scfg.timeframe, p["partner"])
    pm = lc.load_market(partner, scfg.timeframe) if partner else None
    if pm is None:
        out["g"] = np.nan
        out["fresh"] = False
        return out
    k, win = int(p["k"]), int(p["vol_win"])
    # partner move computed on the PARTNER's own bars (causal there), then mapped
    # onto own bars by exact stamp only: the partner bar stamped t closes with ours
    u = _norm_move(np.log(pm["close"]), k, win).reindex(out.index)
    pol = 1.0 if p["pol"] == "follow" else -1.0
    if p["mode"] == "gap":
        v = _norm_move(np.log(out["close"]), k, win)
        g = pol * u - v
    else:
        g = pol * u
    ok = u.notna()
    out["g"] = g.where(ok)
    out["fresh"] = ok.to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return int(p["vol_win"]) + int(p["k"]) + scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last, prev = ind_window.iloc[-1], ind_window.iloc[-2]
    if not valid_atr(last) or not bool(last["fresh"]):
        return flat("no fresh partner bar")
    g = float(last["g"])
    if not np.isfinite(g):
        return flat("g not formed")
    gp = float(prev["g"])
    gp = gp if np.isfinite(gp) else 0.0
    t = float(p["thr"])
    if g >= t and gp < t:
        d = LONG
    elif g <= -t and gp > -t:
        d = SHORT
    else:
        return flat("no trigger")
    if (d == LONG and p["side"] == "short") or (d == SHORT and p["side"] == "long"):
        return flat("side off")
    a = float(last["atr"])
    return Signal(d, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"xlead {p['partner']} {p['mode']} g={g:+.2f}")
