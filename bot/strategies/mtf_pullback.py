"""Multi-timeframe confluence: HIGHER-TF direction + LOWER-TF timing (mtf_ family).

The harness feeds ONE timeframe, so the higher timeframe is SYNTHESISED inside
add_indicators by resampling the base bars.  The whole causality question is
the mapping back onto the base frame, so it is done explicitly:

    HTF bar labelled T covers [T, T+P) and is COMPLETE at wall-clock T+P.
    Base bar labelled t covers [t, t+b) and is evaluated at wall-clock t+b.
    -> base bar t may only see HTF bars with  T + P <= t + b.

So the HTF indicator frame is re-indexed by its own CLOSE time (label + P) and
then `reindex(base_index + b, method="ffill")`.  A partially-filled trailing
HTF bucket has close time > t+b for every base bar that exists, so it can never
be read.  Verified with the prefix-truncation test.

Params (scfg.params):
    htf         : "1h" | "2h" | "4h" | "8h" | "1d"   higher timeframe
    htf_mode    : "ema"   fast/slow EMA stack on HTF closes
                  "donch" HTF donchian break state (last break, ffilled)
                  "slope" sign of the HTF EMA slope over htf_fast bars
    htf_fast    : EMA fast span ("ema") / slope lookback ("slope")
    htf_slow    : EMA slow span ("ema"/"slope") / donchian lookback ("donch")
    htf_sep_atr : "ema"   -> require |EMAf-EMAs| >= this x HTF ATR
                  "slope" -> require |slope|    >= this x HTF ATR   (0 = off)
    trigger     : lower-timeframe entry trigger, in the HTF direction only
                  "pullback" close crosses back over EMA(ltf_ema) after a
                            pullback of 1..pb_bars consecutive closes
                  "break"   close breaks the prior brk_n-bar donchian extreme
                  "cross"   EMA(ltf_fast) crosses EMA(ltf_slow)
                  "rsi"     RSI recovers back through rsi_lo / rsi_hi
                  "none"    every bar in the HTF direction (control / regime beta)
    ltf_ema, pb_bars, brk_n, ltf_fast, ltf_slow, rsi_lo, rsi_hi : trigger knobs
    ltf_slope   : 1 = also require EMA(ltf_ema) itself rising (longs) / falling
    sides       : "both" | "long" | "short"
    min_atr     : absolute $ ATR floor (skip dead volatility)
    require_htf : 1 = trade only with the HTF (default). 0 = IGNORE the HTF
                  bias entirely -> the control run that says whether the
                  higher timeframe adds anything at all.

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x base-timeframe ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, rsi, valid_atr

HTF_MINUTES = {"1h": 60, "2h": 120, "4h": 240, "8h": 480, "1d": 1440}

DEFAULTS = {
    "htf": "4h",
    "htf_mode": "ema",
    "htf_fast": 12,
    "htf_slow": 50,
    "htf_sep_atr": 0.0,
    "trigger": "pullback",
    "ltf_ema": 20,
    "pb_bars": 6,
    "brk_n": 20,
    "ltf_fast": 9,
    "ltf_slow": 30,
    "rsi_lo": 35.0,
    "rsi_hi": 65.0,
    "ltf_slope": 0,
    "sides": "both",
    "min_atr": 0.0,
    "require_htf": 1,
    "sl_src": "ltf",         # "ltf" | "htf" — which ATR scales the stop
    "tp_src": "ltf",         # "ltf" | "htf" — which ATR scales the target
    "invert": 0,             # 1 = FADE the confluence instead of following it
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 15)


def _htf_minutes(p: dict) -> int:
    key = str(p["htf"]).strip().lower()
    if key not in HTF_MINUTES:
        raise ValueError(f"htf must be one of {list(HTF_MINUTES)}, got {p['htf']!r}")
    return HTF_MINUTES[key]


def _htf_direction(df: pd.DataFrame, scfg: StrategyCfg, p: dict) -> pd.DataFrame:
    """Resample to the higher timeframe and return a frame indexed by the HTF
    bar's CLOSE time with a single column 'dir' in {-1, 0, +1}."""
    hm = _htf_minutes(p)
    h = (
        df.resample(f"{hm}min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    mode = str(p["htf_mode"]).strip().lower()
    fast, slow = int(p["htf_fast"]), int(p["htf_slow"])

    if mode == "donch":
        # state machine: +1 after an up-break of the prior `slow` HTF bars,
        # -1 after a down-break, carried forward until the opposite break.
        hh = h["high"].rolling(slow).max().shift(1)
        ll = h["low"].rolling(slow).min().shift(1)
        d = pd.Series(np.nan, index=h.index, dtype=float)
        d[h["close"] > hh] = 1.0
        d[h["close"] < ll] = -1.0
        d = d.ffill().fillna(0.0)
    elif mode == "slope":
        e = ema(h["close"], slow)
        sl = e - e.shift(fast)
        thr = float(p["htf_sep_atr"]) * atr(h, scfg.atr_period)
        d = pd.Series(0.0, index=h.index)
        d = d.mask(sl > thr, 1.0).mask(sl < -thr, -1.0)
    else:  # "ema"
        f = ema(h["close"], fast)
        s = ema(h["close"], slow)
        d = pd.Series(0.0, index=h.index)
        d = d.mask(f > s, 1.0).mask(f < s, -1.0)
        sep_min = float(p["htf_sep_atr"])
        if sep_min > 0:
            d = d.where((f - s).abs() >= sep_min * atr(h, scfg.atr_period), 0.0)

    out = pd.DataFrame({"dir": d.astype(float), "hatr": atr(h, scfg.atr_period)})
    out.index = h.index + pd.Timedelta(minutes=hm)   # HTF bar CLOSE time
    return out


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    bm = _bar_minutes(scfg)

    out["atr"] = atr(out, scfg.atr_period)
    out["ltf_ema"] = ema(out["close"], int(p["ltf_ema"]))
    out["ltf_ema_prev"] = out["ltf_ema"].shift(int(p["pb_bars"]))
    out["ltf_fast"] = ema(out["close"], int(p["ltf_fast"]))
    out["ltf_slow"] = ema(out["close"], int(p["ltf_slow"]))
    out["rsi"] = rsi(out["close"], scfg.rsi_period)
    out["don_hi"], out["don_lo"] = donchian(out, int(p["brk_n"]))

    # run length of consecutive closes on the wrong side of the LTF EMA
    # (cumsum-within-group: causal, and identical on any prefix of the frame)
    below = out["close"] <= out["ltf_ema"]
    out["run_below"] = below.groupby((~below).cumsum()).cumsum().astype(float)
    above = out["close"] >= out["ltf_ema"]
    out["run_above"] = above.groupby((~above).cumsum()).cumsum().astype(float)

    # ---- higher timeframe, mapped on COMPLETED HTF bars only ----
    htf = _htf_direction(out, scfg, p)
    mapped = htf.reindex(out.index + pd.Timedelta(minutes=bm), method="ffill")
    out["htf_dir"] = mapped["dir"].to_numpy()
    out["htf_atr"] = mapped["hatr"].to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bm = _bar_minutes(scfg)
    ratio = max(1, _htf_minutes(p) // bm)
    htf_bars = 4 * max(int(p["htf_fast"]), int(p["htf_slow"]), scfg.atr_period) + 10
    ltf_bars = max(
        5 * int(p["ltf_ema"]), 5 * int(p["ltf_slow"]), 5 * int(p["ltf_fast"]),
        int(p["brk_n"]) + 2, int(p["pb_bars"]) + 2,
        5 * scfg.atr_period, 5 * scfg.rsi_period,
    ) + 10
    return int(htf_bars * ratio + ltf_bars)


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")

    # ---- higher-timeframe bias ----
    bias = float(last["htf_dir"]) if np.isfinite(last["htf_dir"]) else 0.0
    use_htf = bool(int(p["require_htf"]))
    if use_htf and bias == 0.0:
        return flat("no HTF bias")

    c_last, c_prev = float(last["close"]), float(prev["close"])
    trig = str(p["trigger"]).strip().lower()
    pb = int(p["pb_bars"])

    long_ok = short_ok = False
    if trig == "pullback":
        e_last, e_prev = float(last["ltf_ema"]), float(prev["ltf_ema"])
        rb, ra = float(prev["run_below"]), float(prev["run_above"])
        long_ok = (c_last > e_last) and (1 <= rb <= pb)
        short_ok = (c_last < e_last) and (1 <= ra <= pb)
        del e_prev
    elif trig == "break":
        hi, lo = last["don_hi"], last["don_lo"]
        if not (np.isfinite(hi) and np.isfinite(lo)):
            return flat("donchian warming up")
        long_ok = (c_prev <= float(prev["don_hi"])) and (c_last > float(hi))
        short_ok = (c_prev >= float(prev["don_lo"])) and (c_last < float(lo))
    elif trig == "cross":
        long_ok = prev["ltf_fast"] <= prev["ltf_slow"] and last["ltf_fast"] > last["ltf_slow"]
        short_ok = prev["ltf_fast"] >= prev["ltf_slow"] and last["ltf_fast"] < last["ltf_slow"]
    elif trig == "rsi":
        lo_t, hi_t = float(p["rsi_lo"]), float(p["rsi_hi"])
        long_ok = float(prev["rsi"]) < lo_t <= float(last["rsi"])
        short_ok = float(prev["rsi"]) > hi_t >= float(last["rsi"])
    elif trig == "none":
        long_ok = short_ok = True
    else:
        raise ValueError(f"unknown trigger {p['trigger']!r}")

    if int(p["ltf_slope"]):
        e_now, e_ago = float(last["ltf_ema"]), float(last["ltf_ema_prev"])
        if not np.isfinite(e_ago):
            return flat("LTF slope warming up")
        long_ok = long_ok and e_now > e_ago
        short_ok = short_ok and e_now < e_ago

    # ---- confluence: the LTF trigger must agree with the HTF bias ----
    if use_htf:
        long_ok = long_ok and bias > 0
        short_ok = short_ok and bias < 0
    if long_ok and short_ok:          # "none" trigger without an HTF bias
        return flat("ambiguous trigger")
    if not (long_ok or short_ok):
        return flat("no confluence")

    direction = LONG if long_ok else SHORT
    if int(p["invert"]):
        direction = -direction
    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    # "sniper" geometry: the stop can be sized off the precise LTF entry while
    # the target is sized off the higher timeframe the trade is actually riding.
    ha = float(last["htf_atr"]) if np.isfinite(last["htf_atr"]) else 0.0
    if (str(p["sl_src"]) == "htf" or str(p["tp_src"]) == "htf") and ha <= 0:
        return flat("HTF ATR warming up")
    sl_a = ha if str(p["sl_src"]) == "htf" else a
    tp_a = ha if str(p["tp_src"]) == "htf" else a

    side = "long" if direction == LONG else "short"
    return Signal(
        direction, c_last, scfg.atr_sl_mult * sl_a, scfg.atr_tp_mult * tp_a, a,
        f"MTF {p['htf']}/{p['htf_mode']} bias {bias:+.0f} + {trig} {side} @ {c_last:.2f}",
    )
