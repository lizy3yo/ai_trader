"""Supply & demand zones, Sam Seiden / Online Trading Academy style (prefix ms_).
Rules in results/ms_rules.md.

DEMAND (supply = mirror):
  base      1..base_max consecutive "boring" candles (body <= 50% of range)
            directly before the leg-out; a longer run is rejected (time at
            level is an odds enhancer: fewer candles = stronger)
  leg-out   the next candle is "exciting": bullish, body >= 50% of range,
            range >= strength_k x ATR, and it closes above the highest high of
            the base (price leaves the zone fast). Leg-in may be a rally (RBR)
            or a drop (DBR).
  zone      proximal = highest candle BODY of the base, distal = lowest wick
  freshness only the FIRST return to the zone is traded (any touch consumes
            it); a close beyond the distal kills it; zones live `expiry` bars
  entry     first bar whose low reaches the proximal and that does not trade
            through the distal; enter at its close (fill next open)
  stop      distal - sl_pad_atr x ATR;  target rr x risk (OTA minimum 3:1)
  odds      enh "trend": trade demand only while the higher-timeframe close is
            above its 20-period SMA (Seiden's 2012 trend tool), supply only
            below; enh "curve": demand only when the proximal sits in the lower
            third of the last 50 higher-timeframe bars' range (supply: upper
            third); "all" = trend + curve; "none" = raw zones.
Higher timeframe pairing (completed htf bars only): M15->H4, H1->D1,
H4->D1, D1->W1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat
from . import ms_common as M

DEFAULTS = {"base_max": 6, "boring": 0.5, "exciting": 0.5, "strength_k": 1.0,
            "expiry": 200, "rr": 0.0, "sl_pad_atr": 0.1, "min_sl_atr": 0.3,
            "enh": "none", "trend_n": 20, "curve_n": 50, "side": "both"}


def _rr(p: dict, scfg: StrategyCfg) -> float:
    """Reward:risk. rr <= 0 -> cfg.atr_tp_mult / cfg.atr_sl_mult, so the grid sets
    RR through the shared ATR knobs and the random baseline (which uses those
    ATR knobs for its own stops) gets the same stop/target geometry."""
    r = float(p["rr"])
    return r if r > 0 else float(scfg.atr_tp_mult) / float(scfg.atr_sl_mult)


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _htf_filters(df: pd.DataFrame, htf: str, trend_n: int, curve_n: int) -> pd.DataFrame:
    hb, key = M.htf_bars(df, htf)
    feat = pd.DataFrame({
        "h_close": hb["close"],
        "h_sma": hb["close"].rolling(trend_n).mean(),
        "h_hi": hb["high"].rolling(curve_n).max(),
        "h_lo": hb["low"].rolling(curve_n).min(),
    }, index=hb.index)
    return M.map_prior(feat, key, df.index)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    o, h, l, c = M.arrays(out)
    a = out["atr"].to_numpy(float)
    n = len(out)
    rng = h - l
    body = np.abs(c - o)
    with np.errstate(invalid="ignore", divide="ignore"):
        boring = (rng > 0) & (body <= float(p["boring"]) * rng)
        exc = (rng > 0) & (body >= float(p["exciting"]) * rng) & (rng >= float(p["strength_k"]) * a)
    # run length of consecutive boring candles ending at each bar
    run = np.zeros(n, dtype=np.int64)
    for t in range(n):
        run[t] = run[t - 1] + 1 if (boring[t] and t > 0) else (1 if boring[t] else 0)

    enh = str(p["enh"])
    htf = M.HTF_OF[scfg.timeframe.upper()]
    f = _htf_filters(out, htf, int(p["trend_n"]), int(p["curve_n"]))
    up_ok = (f["h_close"] > f["h_sma"]).to_numpy()
    dn_ok = (f["h_close"] < f["h_sma"]).to_numpy()
    hhi, hlo = f["h_hi"].to_numpy(float), f["h_lo"].to_numpy(float)
    use_trend = enh in ("trend", "all")
    use_curve = enh in ("curve", "all")

    long_sig = np.zeros(n)
    short_sig = np.zeros(n)
    sig_stop = np.full(n, np.nan)
    dem = []   # (proximal, distal, born)
    sup = []
    bmax = int(p["base_max"])
    expiry = int(p["expiry"])
    side = p["side"]

    for t in range(1, n):
        # ---- test fresh zones created before t ----
        keep = []
        best = None
        for z in dem:
            prox, dist, born = z
            if t - born > expiry or c[t] < dist:
                continue
            if l[t] <= prox:                      # first return consumes
                if l[t] > dist and (best is None or prox > best[0]):
                    best = z
                continue
            keep.append(z)
        dem = keep
        if best is not None:
            long_sig[t], sig_stop[t] = 1.0, best[1]
        keep = []
        best_s = None
        for z in sup:
            prox, dist, born = z
            if t - born > expiry or c[t] > dist:
                continue
            if h[t] >= prox:
                if h[t] < dist and (best_s is None or prox < best_s[0]):
                    best_s = z
                continue
            keep.append(z)
        sup = keep
        if best_s is not None:
            if long_sig[t]:
                long_sig[t] = 0.0
                sig_stop[t] = np.nan
            else:
                short_sig[t], sig_stop[t] = 1.0, best_s[1]

        # ---- new zone: base ending at t-1, leg-out candle t ----
        k = run[t - 1]
        if exc[t] and 1 <= k <= bmax:
            s = t - k
            bh, bl = h[s:t].max(), l[s:t].min()
            if c[t] > o[t] and c[t] > bh and side != "short":
                prox = float(np.max(np.maximum(o[s:t], c[s:t])))
                ok = (not use_trend or up_ok[t]) and (
                    not use_curve or (np.isfinite(hhi[t]) and prox <= hlo[t] + (hhi[t] - hlo[t]) / 3.0))
                if ok and prox > bl:
                    dem.append((prox, float(bl), t))
                    dem = dem[-10:]
            elif c[t] < o[t] and c[t] < bl and side != "long":
                prox = float(np.min(np.minimum(o[s:t], c[s:t])))
                ok = (not use_trend or dn_ok[t]) and (
                    not use_curve or (np.isfinite(hhi[t]) and prox >= hhi[t] - (hhi[t] - hlo[t]) / 3.0))
                if ok and prox < bh:
                    sup.append((prox, float(bh), t))
                    sup = sup[-10:]

    out["long_sig"] = long_sig
    out["short_sig"] = short_sig
    out["sig_stop"] = sig_stop
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    tf = scfg.timeframe.upper()
    htf = M.HTF_OF[tf]
    mult = (10080 if htf == "W1" else M.TF_MIN[htf]) // M.TF_MIN[tf]
    return max(300, (max(int(p["trend_n"]), int(p["curve_n"])) + 2) * mult)


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    go_l, go_s = last["long_sig"] > 0, last["short_sig"] > 0
    if not (go_l or go_s):
        return flat("no fresh zone touch")
    a = float(last["atr"])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    c = float(last["close"])
    st = float(last["sig_stop"])
    pad = float(p["sl_pad_atr"]) * a
    d = LONG if go_l else SHORT
    sl = (c - (st - pad)) if d == LONG else ((st + pad) - c)
    sl = max(sl, float(p["min_sl_atr"]) * a)
    tp = _rr(p, scfg) * sl
    if not (np.isfinite(sl) and sl > 0):
        return flat("bad SL")
    return Signal(d, c, sl, tp, a, f"S&D {'demand' if d == LONG else 'supply'} first touch, distal {st:.2f}")
