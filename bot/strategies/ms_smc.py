"""Complete SMC / ICT entry model (prefix ms_), rules in results/ms_rules.md.

    1. BIAS     swing-structure trend (LuxAlgo swing layer, swing_len bars) on
                this chart (bias "swing"), or the structure trend of the next
                higher timeframe from COMPLETED htf bars (bias "htf":
                M15->H4, H1->D1, H4->D1, size htf_len).
    2. SWEEP    the leg that the internal CHoCH breaks out of must have taken
                the previous internal pivot (sell-side liquidity for longs).
    3. CHoCH    internal structure (LuxAlgo internal layer, int_len bars)
                changes character in the bias direction (close through the
                last internal pivot while the internal trend was opposite).
    4. POI      the internal order block of that CHoCH (LuxAlgo
                storeOrderBlock) or the last LuxAlgo FVG inside the CHoCH leg
                (poi "fvg"). Setup lives `expiry` bars; first touch only.
    5. ZONE     the POI's entry edge must sit in discount (longs) / premium
                (shorts) of the bias dealing range (equilibrium 50%).
    6. ENTRY    on the close of the first bar that trades into the POI and
                closes back inside/above it (fill = next bar open), the bias
                must still agree.
    7. STOP     beyond the swept leg extreme (+ sl_pad_atr x ATR).
    8. TARGET   rr x risk (tp_mode "rr") or the opposing liquidity = the
                dealing-range extreme (tp_mode "liq", skipped below 1R).
Causal: structure points exist only after their confirmation bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat
from . import ms_common as M

DEFAULTS = {
    "swing_len": 50, "int_len": 5, "bias": "swing", "htf_len": 5,
    "sweep": 1, "pd": 1, "poi": "ob", "expiry": 20,
    "tp_mode": "rr", "rr": 0.0, "sl_pad_atr": 0.1, "min_sl_atr": 0.3,
    "side": "both",
}


def _rr(p: dict, scfg: StrategyCfg) -> float:
    """Reward:risk. rr <= 0 -> cfg.atr_tp_mult / cfg.atr_sl_mult, so the grid sets
    RR through the shared ATR knobs and the random baseline (which uses those
    ATR knobs for its own stops) gets the same stop/target geometry."""
    r = float(p["rr"])
    return r if r > 0 else float(scfg.atr_tp_mult) / float(scfg.atr_sl_mult)


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    a200 = atr(out, 200).to_numpy(float)
    o, h, l, c = M.arrays(out)
    n = len(out)

    sw = M.structure(o, h, l, c, int(p["swing_len"]), a200)
    it = M.structure(o, h, l, c, int(p["int_len"]), a200,
                     other_hi=sw["piv_hi"], other_lo=sw["piv_lo"])

    if p["bias"] == "htf":
        hs = M.htf_structure(out, M.HTF_OF[scfg.timeframe.upper()], int(p["htf_len"]))
        bias = hs["h_trend"].fillna(0).to_numpy(float)
        top, bot = hs["h_top"].to_numpy(float), hs["h_bot"].to_numpy(float)
    else:
        bias = sw["trend"].astype(float)
        top, bot = sw["top"], sw["bot"]
    eq = (top + bot) / 2.0

    use_fvg = p["poi"] == "fvg"
    fb, fs = M.fvg_flags(o, h, l, c) if use_fvg else (None, None)

    long_sig = np.zeros(n)
    short_sig = np.zeros(n)
    sig_stop = np.full(n, np.nan)
    sig_tgt = np.full(n, np.nan)
    expiry = int(p["expiry"])
    need_sweep = bool(int(p["sweep"]))
    need_pd = bool(int(p["pd"]))
    side = p["side"]

    La = Sa = False
    Lzl = Lzh = Lst = Szl = Szh = Sst = np.nan
    Lb = Sb = -1
    lb_fvg = (-1, np.nan, np.nan)   # last bullish FVG (bar, top, bottom)
    ls_fvg = (-1, np.nan, np.nan)

    for t in range(n):
        if use_fvg:
            if fb[t]:
                lb_fvg = (t, l[t], h[t - 2])
            if fs[t]:
                ls_fvg = (t, l[t - 2], h[t])
        # ---- existing long setup ----
        if La:
            if t - Lb > expiry or l[t] < Lst:
                La = False
            elif l[t] <= Lzh:
                ok = c[t] >= Lzl and bias[t] > 0
                if need_pd:
                    ok = ok and np.isfinite(eq[t]) and Lzh <= eq[t]
                if ok:
                    long_sig[t] = 1.0
                    sig_stop[t] = Lst
                    sig_tgt[t] = top[t]
                La = False
        if Sa:
            if t - Sb > expiry or h[t] > Sst:
                Sa = False
            elif h[t] >= Szl:
                ok = c[t] <= Szh and bias[t] < 0
                if need_pd:
                    ok = ok and np.isfinite(eq[t]) and Szl >= eq[t]
                if ok:
                    short_sig[t] = 1.0
                    sig_stop[t] = Sst
                    sig_tgt[t] = bot[t]
                Sa = False
        # ---- new setups from an internal CHoCH at this bar ----
        e = it["evt"][t]
        if e == 2 and bias[t] > 0 and side != "short" and (it["sweep"][t] or not need_sweep):
            s = it["brk_from"][t]
            if use_fvg:
                zb, zt, zl = lb_fvg[0], lb_fvg[1], lb_fvg[2]
                zone = (zl, zt) if zb >= s + 2 else (np.nan, np.nan)
            else:
                zone = (it["ob_lo"][t], it["ob_hi"][t])
            st = it["leg_ext"][t]
            if np.isfinite(zone[0]) and zone[1] > zone[0] and zone[1] > st and c[t] > zone[1]:
                La, Lzl, Lzh, Lst, Lb = True, max(zone[0], st), zone[1], st, t
        elif e == -2 and bias[t] < 0 and side != "long" and (it["sweep"][t] or not need_sweep):
            s = it["brk_from"][t]
            if use_fvg:
                zb, zt, zl = ls_fvg[0], ls_fvg[1], ls_fvg[2]
                zone = (zl, zt) if zb >= s + 2 else (np.nan, np.nan)
            else:
                zone = (it["ob_lo"][t], it["ob_hi"][t])
            st = it["leg_ext"][t]
            if np.isfinite(zone[0]) and zone[1] > zone[0] and zone[0] < st and c[t] < zone[0]:
                Sa, Szl, Szh, Sst, Sb = True, zone[0], min(zone[1], st), st, t

    out["long_sig"] = long_sig
    out["short_sig"] = short_sig
    out["sig_stop"] = sig_stop
    out["sig_tgt"] = sig_tgt
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = max(1000, 4 * int(p["swing_len"]))
    if p["bias"] == "htf":
        mult = M.TF_MIN[M.HTF_OF[scfg.timeframe.upper()]] if M.HTF_OF[scfg.timeframe.upper()] != "W1" else 10080
        w = max(w, 30 * int(p["htf_len"]) * mult // M.TF_MIN[scfg.timeframe.upper()])
    return w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    go_l, go_s = last["long_sig"] > 0, last["short_sig"] > 0
    if not (go_l or go_s) or (go_l and go_s):
        return flat("no SMC setup")
    a = float(last["atr"])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    c = float(last["close"])
    st = float(last["sig_stop"])
    pad = float(p["sl_pad_atr"]) * a
    d = LONG if go_l else SHORT
    sl = (c - (st - pad)) if d == LONG else ((st + pad) - c)
    sl = max(sl, float(p["min_sl_atr"]) * a)
    if p["tp_mode"] == "liq":
        tgt = float(last["sig_tgt"])
        tp = (tgt - c) if d == LONG else (c - tgt)
        if not np.isfinite(tp) or tp < sl:
            return flat("opposing liquidity < 1R")
    else:
        tp = _rr(p, scfg) * sl
    if not (np.isfinite(sl) and sl > 0 and tp > 0):
        return flat("bad SL/TP")
    return Signal(d, c, sl, tp, a, f"SMC {'long' if d == LONG else 'short'} POI retest, stop {st:.2f}")
