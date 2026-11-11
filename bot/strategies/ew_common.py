"""Shared, causal Elliott Wave engine for the ew_ family (not a strategy).

Rules and sources: results/ew_rules.md (pre-registered before testing).

ZigZag (ATR reversal, non-repainting): in an up-leg a new high replaces the
running extreme; otherwise, if low <= extreme - k x ATR[t], the extreme is
CONFIRMED as a pivot high at bar t and a down-leg starts. Mirror for lows.
A pivot is known only from its confirmation bar on.

Wave scanners work on confirmed pivots only. Longs are coded once; shorts are
the same code run on the mirrored series (prices x -1, pivot types flipped).
Each scanner returns per-bar arrays: sig (+1/-1/0), stop level, target level
(structural target, NaN if none). Everything at row t uses rows <= t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat

PAD_ATR = 0.1       # stop pad beyond the wave extreme (as ms_)
MIN_SL_ATR = 0.3


def zigzag(h: np.ndarray, l: np.ndarray, a: np.ndarray, k: float):
    """Confirmed pivots: arrays (bar, price, type +1 high / -1 low, confirm_bar)."""
    n = len(h)
    pb, pp, pt, pc = [], [], [], []
    start = 0
    while start < n and not np.isfinite(a[start]):
        start += 1
    if start >= n:
        e = np.array([], dtype=np.int64)
        return e, np.array([]), e, e
    d = 1
    ext, ext_bar = h[start], start
    for t in range(start + 1, n):
        if d == 1:
            if h[t] > ext:
                ext, ext_bar = h[t], t
            elif l[t] <= ext - k * a[t]:
                pb.append(ext_bar); pp.append(ext); pt.append(1); pc.append(t)
                d, ext, ext_bar = -1, l[t], t
        else:
            if l[t] < ext:
                ext, ext_bar = l[t], t
            elif h[t] >= ext + k * a[t]:
                pb.append(ext_bar); pp.append(ext); pt.append(-1); pc.append(t)
                d, ext, ext_bar = 1, h[t], t
    return (np.array(pb, dtype=np.int64), np.array(pp, dtype=float),
            np.array(pt, dtype=np.int64), np.array(pc, dtype=np.int64))


def _armed_break(sig, stop, tgt, c, pc, j, n, level, stop_lvl, tgt_lvl, s):
    """Arm at confirmation of pivot j; fire at the first close beyond `level`
    (in mirrored units) before pivot j+1 is confirmed. One trade per setup."""
    t0 = pc[j]
    t1 = pc[j + 1] if j + 1 < len(pc) else n
    for t in range(t0, t1):
        if c[t] > level:
            if tgt_lvl is not None and c[t] >= tgt_lvl:
                return            # structural target already reached
            sig[t] = s
            stop[t] = s * stop_lvl
            tgt[t] = s * tgt_lvl if tgt_lvl is not None else np.nan
            return


def scan(model: str, c, pp, pt, pc, p: dict, s: int, sig, stop, tgt):
    """Fill sig/stop/tgt for one side. Prices here are already mirrored (x s)."""
    n = len(c)
    m = len(pp)
    for j in range(m):
        if pt[j] != -1:              # every setup ends on a confirmed LOW (mirrored)
            continue
        if model == "w3" and j >= 2:
            L0, H1, L2 = pp[j - 2], pp[j - 1], pp[j]
            w1 = H1 - L0
            if w1 <= 0 or not (L2 > L0):                       # R1
                continue
            r = (H1 - L2) / w1
            if not (p["r_lo"] <= r <= p["r_hi"]):
                continue
            _armed_break(sig, stop, tgt, c, pc, j, n, H1, L2, L2 + 1.618 * w1, s)
        elif model == "w5" and j >= 4:
            L0, H1, L2, H3, L4 = pp[j - 4: j + 1]
            w1, w3 = H1 - L0, H3 - L2
            if not (w1 > 0 and L2 > L0 and H3 > H1 and w3 >= w1 and L4 > H1):   # R1 R2 R3
                continue
            r = (H3 - L4) / w3
            if not (p["r_lo"] <= r <= p["r_hi"]):
                continue
            t = pc[j]                                   # entry when wave 4 is confirmed over
            target = L4 + w1                            # wave 5 = wave 1
            if c[t] >= target:
                continue
            sig[t], stop[t], tgt[t] = s, s * L4, s * target
        elif model == "abc" and j >= 6:
            Lpp, Hp, Lpre, H0, La, Hb, Lc = pp[j - 6: j + 1]
            if not (H0 > Hp and Lpre > Lpp):            # established trend (HH + HL)
                continue
            A = H0 - La
            if A <= 0:
                continue
            br = (Hb - La) / A
            if not (p["b_lo"] <= br <= p["b_hi"]):
                continue
            C = Hb - Lc
            if not (Lc < La and Lc > Lpp and p["r_lo"] <= C / A <= p["r_hi"]):
                continue
            _armed_break(sig, stop, tgt, c, pc, j, n, Hb, Lc, None, s)


def wave_signals(df: pd.DataFrame, model: str, k: float, p: dict, atr_period: int):
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    a = atr(df, atr_period).to_numpy(float)
    n = len(c)
    sig = np.zeros(n, dtype=np.int8)
    stop = np.full(n, np.nan)
    tgt = np.full(n, np.nan)
    pb, pp, pt, pc = zigzag(h, l, a, k)
    if p.get("side", "both") in ("both", "long"):
        scan(model, c, pp, pt, pc, p, 1, sig, stop, tgt)
    if p.get("side", "both") in ("both", "short"):
        s2 = np.zeros(n, dtype=np.int8)
        st2, tg2 = np.full(n, np.nan), np.full(n, np.nan)
        scan(model, -c, -pp, -pt, pc, p, -1, s2, st2, tg2)
        both = (sig != 0) & (s2 != 0)
        put = (s2 != 0) & (sig == 0)
        sig[put], stop[put], tgt[put] = s2[put], st2[put], tg2[put]
        sig[both] = 0                                  # conflicting setups -> no trade
    return sig, stop, tgt, a


# --------------------------------------------------------------- strategy glue
def add(df: pd.DataFrame, scfg: StrategyCfg, model: str, p: dict) -> pd.DataFrame:
    out = df.copy()
    sig, stop, tgt, a = wave_signals(out, model, float(p["k"]), p, scfg.atr_period)
    out["ew_atr"] = a
    out["ew_sig"] = sig.astype(float)
    out["ew_stop"] = stop
    out["ew_tgt"] = tgt
    return out


def rr_of(p: dict, scfg: StrategyCfg) -> float:
    """Fixed-R target: rr param if > 0, else cfg.atr_tp_mult / cfg.atr_sl_mult (so the
    random baseline, which uses those ATR knobs, gets the same reward:risk)."""
    r = float(p.get("rr", 0.0))
    return r if r > 0 else float(scfg.atr_tp_mult) / float(scfg.atr_sl_mult)


def signal(ind_window: pd.DataFrame, scfg: StrategyCfg, p: dict, label: str) -> Signal:
    last = ind_window.iloc[-1]
    d = int(last["ew_sig"])
    if d == 0:
        return flat("no wave setup")
    a = float(last["ew_atr"])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    c = float(last["close"])
    st = float(last["ew_stop"])
    pad = PAD_ATR * a
    sl = (c - (st - pad)) if d == LONG else ((st + pad) - c)
    sl = max(sl, MIN_SL_ATR * a)
    if p["target"] == "struct":
        g = float(last["ew_tgt"])
        tp = (g - c) if d == LONG else (c - g)
    else:
        tp = rr_of(p, scfg) * sl
    if not (np.isfinite(sl) and sl > 0 and np.isfinite(tp) and tp > 0):
        return flat("bad SL/TP")
    return Signal(LONG if d == LONG else SHORT, c, sl, tp, a, label)
