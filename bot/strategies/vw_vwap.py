"""Session-anchored VWAP: reversion, trend-ride and band-break (prefix vw_).

The day-trader's anchored VWAP. At the anchor hour a new session starts and
three expanding sums are reset:

    cw   = sum(w)            cwx = sum(w * tp)        cwx2 = sum(w * tp^2)
    vwap = cwx / cw          var = cwx2/cw - vwap^2   sigma = sqrt(var)

`tp` is the bar's typical price (h+l+c)/3, `w` a volume proxy. Everything is a
groupby-cumsum inside the session, so row i uses only rows <= i (causal by
construction: no full-frame statistic, no shift needed because the bar's own
close/high/low/volume ARE known at the moment the bar closes).

Modes
    revert     : price is >= band_k sigma away from VWAP and starts back
                 toward it  -> trade toward VWAP (classic mean reversion)
    ride       : price is on one side of VWAP (session trend) and pulls back
                 to touch it -> trade with the trend off the VWAP
    band_break : close crosses OUT through the +/- band_k sigma band
                 -> momentum continuation

Params (scfg.params)
    anchor_h      : server hour at which the VWAP resets (0 = broker day,
                    9 = London open, 8 = pre-London, ...)
    vol_mode      : "range" (w = bar range, a volume proxy), "equal"
                    (w = 1, plain typical-price VWAP) or "tick"
                    (real tick_volume from data/gold_m5_vol.csv, M5 only)
    mode          : "revert" | "ride" | "band_break"
    band_k        : deviation threshold in VWAP sigmas
    touch_k       : how close to VWAP a pullback must come (ride), in sigmas
    trigger       : revert only -- "turn" (bar closes back toward VWAP) or
                    "reenter" (close comes back inside the band)
    min_bars      : minimum bars elapsed in the session before signalling
    slope_n       : ride only -- require the VWAP to slope the right way over
                    this many bars (0 = off)
    tp_mode       : "atr" (scfg.atr_tp_mult x ATR) or "vwap" (target the VWAP
                    itself, clipped into [tp_min_atr, tp_max_atr] x ATR)
    min_sigma_atr : skip when sigma < this x ATR (degenerate early session)
    max_dev       : skip when |deviation| exceeds this many sigmas (news spike)
    sides         : "both" | "long" | "short"

SL = scfg.atr_sl_mult x ATR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "anchor_h": 0,
    "vol_mode": "range",
    "mode": "revert",
    "band_k": 2.0,
    "touch_k": 0.25,
    "trigger": "turn",
    "min_bars": 12,
    "slope_n": 0,
    "tp_mode": "atr",
    "tp_min_atr": 0.5,
    "tp_max_atr": 6.0,
    "min_sigma_atr": 0.0,
    "max_dev": 99.0,
    "sides": "both",
}

_VOL_PATH = Path(__file__).resolve().parents[2] / "data" / "gold_m5_vol.csv"
_VOL_CACHE: dict[str, pd.Series] = {}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 15)


def _tick_volume(index: pd.DatetimeIndex) -> pd.Series | None:
    """Real M5 tick volume, aligned to `index`. None if unusable."""
    if "m5" not in _VOL_CACHE:
        if not _VOL_PATH.exists():
            _VOL_CACHE["m5"] = pd.Series(dtype=float)
        else:
            raw = pd.read_csv(_VOL_PATH)
            raw.columns = [c.strip().lower() for c in raw.columns]
            raw["time"] = pd.to_datetime(raw["time"])
            _VOL_CACHE["m5"] = raw.set_index("time")["tick_volume"].astype(float)
    src = _VOL_CACHE["m5"]
    if src.empty:
        return None
    aligned = src.reindex(index)
    if aligned.isna().mean() > 0.05:   # timeframe/date mismatch -> refuse
        return None
    return aligned.ffill().fillna(1.0).clip(lower=1.0)


def _weights(out: pd.DataFrame, vol_mode: str) -> pd.Series:
    rng = (out["high"] - out["low"]).astype(float)
    if vol_mode == "equal":
        return pd.Series(1.0, index=out.index)
    if vol_mode == "tick":
        tv = _tick_volume(out.index)
        if tv is not None:
            return tv
        # fall through to the range proxy when tick volume is unavailable
    return rng.clip(lower=1e-6)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)

    anchor_h = int(p["anchor_h"])
    # session id: shift the clock back by anchor_h, then take the calendar day
    sess = pd.Series((out.index - pd.Timedelta(hours=anchor_h)).normalize(),
                     index=out.index)

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    w = _weights(out, str(p["vol_mode"]))

    grp = sess
    cw = w.groupby(grp).cumsum()
    cwx = (w * tp).groupby(grp).cumsum()
    cwx2 = (w * tp * tp).groupby(grp).cumsum()

    vwap = cwx / cw
    var = (cwx2 / cw) - vwap * vwap
    sigma = np.sqrt(var.clip(lower=0.0))

    out["vwap"] = vwap
    out["vwsig"] = sigma
    out["sess_n"] = w.groupby(grp).cumcount() + 1

    n = max(int(p["slope_n"]), 0)
    if n > 0:
        # VWAP slope over n bars, only valid inside the same session
        out["vwap_prev_n"] = vwap.groupby(grp).shift(n)
    else:
        out["vwap_prev_n"] = np.nan
    return out


def warmup(scfg: StrategyCfg) -> int:
    bars_per_day = 1440 // _bar_minutes(scfg)
    return bars_per_day + 5 * scfg.atr_period + 10


def _dev(row) -> float:
    s = float(row["vwsig"])
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    return (float(row["close"]) - float(row["vwap"])) / s


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg) + 2:
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    if float(last["sess_n"]) < float(p["min_bars"]):
        return flat("session too young")

    a = float(last["atr"])
    sig = float(last["vwsig"])
    if not np.isfinite(sig) or sig <= 0:
        return flat("no VWAP sigma")
    if sig < float(p["min_sigma_atr"]) * a:
        return flat("sigma too small")

    d_last = _dev(last)
    d_prev = _dev(prev)
    if not (np.isfinite(d_last) and np.isfinite(d_prev)):
        return flat("no deviation")
    if abs(d_last) > float(p["max_dev"]):
        return flat("deviation off the chart")

    k = float(p["band_k"])
    mode = str(p["mode"])
    vwap = float(last["vwap"])
    c_last = float(last["close"])
    direction = 0
    desc = ""

    if mode == "revert":
        if str(p["trigger"]) == "reenter":
            if d_prev <= -k and d_last > -k:
                direction, desc = LONG, f"re-entered lower band ({d_prev:+.2f}->{d_last:+.2f}s)"
            elif d_prev >= k and d_last < k:
                direction, desc = SHORT, f"re-entered upper band ({d_prev:+.2f}->{d_last:+.2f}s)"
        else:  # "turn"
            if d_prev <= -k and d_prev < d_last < 0:
                direction, desc = LONG, f"turned up at {d_prev:+.2f}s below VWAP"
            elif d_prev >= k and 0 < d_last < d_prev:
                direction, desc = SHORT, f"turned down at {d_prev:+.2f}s above VWAP"

    elif mode == "ride":
        tk = float(p["touch_k"])
        n = int(p["slope_n"])
        slope_ok_up = slope_ok_dn = True
        if n > 0:
            vprev = float(last["vwap_prev_n"])
            if not np.isfinite(vprev):
                return flat("no VWAP slope yet")
            slope_ok_up = vwap > vprev
            slope_ok_dn = vwap < vprev
        low_dev = (float(last["low"]) - vwap) / sig
        high_dev = (float(last["high"]) - vwap) / sig
        bull_bar = c_last > float(last["open"])
        bear_bar = c_last < float(last["open"])
        if d_last > 0 and low_dev <= tk and bull_bar and slope_ok_up:
            direction, desc = LONG, f"pullback to VWAP (low {low_dev:+.2f}s) in up-session"
        elif d_last < 0 and high_dev >= -tk and bear_bar and slope_ok_dn:
            direction, desc = SHORT, f"pullback to VWAP (high {high_dev:+.2f}s) in down-session"

    elif mode == "band_break":
        if d_prev <= k < d_last:
            direction, desc = LONG, f"broke +{k:.1f}s band ({d_last:+.2f}s)"
        elif d_prev >= -k > d_last:
            direction, desc = SHORT, f"broke -{k:.1f}s band ({d_last:+.2f}s)"
    else:
        return flat(f"unknown mode {mode}")

    if direction == 0:
        return flat("no VWAP signal")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    sl_dist = scfg.atr_sl_mult * a
    if str(p["tp_mode"]) == "vwap":
        tp_dist = abs(c_last - vwap)
        tp_dist = min(max(tp_dist, float(p["tp_min_atr"]) * a), float(p["tp_max_atr"]) * a)
    else:
        tp_dist = scfg.atr_tp_mult * a
    if tp_dist <= 0 or sl_dist <= 0:
        return flat("degenerate SL/TP")

    side = "long" if direction == LONG else "short"
    return Signal(direction, c_last, sl_dist, tp_dist, a,
                  f"VWAP {mode} {side}: {desc} (vwap {vwap:.2f} sig {sig:.2f})")
