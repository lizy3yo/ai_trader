"""Multi-timeframe confluence #2: HTF/daily BIAS + first-hour-of-session entry.

Top-down analysis, mechanised: a higher timeframe (H4 / D1, synthesised by
resampling the base bars) decides the DIRECTION for the day, and the entry is
the break of the session's own OPENING RANGE — the first `or_bars` bars after
`ses_start_h` — but only on the side the higher timeframe allows.

Causality:
  * HTF bars are re-indexed by their CLOSE time (label + period) and ffilled
    onto base-bar close times, so a base bar can only ever read HTF bars that
    have already completed.  (Same construction as mtf_pullback; both pass the
    prefix-truncation test.)
  * The opening range is exposed only on bars whose within-day session rank is
    strictly greater than `or_bars`, i.e. after every range bar has closed.

Params (scfg.params):
    htf, htf_mode, htf_fast, htf_slow, htf_sep_atr : see mtf_pullback
    ses_start_h  : session start hour, server clock (opening range starts here)
    or_bars      : number of base bars forming the opening range
    entry_end_h  : entries stop when the ENTRY moment reaches this hour
    buf_atr      : breakout buffer beyond the range, in base-timeframe ATRs
    min_range_atr / max_range_atr : skip days whose OR is too narrow / too wide
    mode         : "follow" (trade the break) | "fade" (bet it fails)
    sides        : "both" | "long" | "short"
    require_htf  : 1 = only bias-aligned breaks (default).  0 = ignore the HTF
                   bias -> the control run that isolates what the HTF adds.
    min_atr      : absolute $ ATR floor

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x base-timeframe ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

HTF_MINUTES = {"1h": 60, "2h": 120, "4h": 240, "8h": 480, "1d": 1440}

DEFAULTS = {
    "htf": "1d",
    "htf_mode": "ema",
    "htf_fast": 5,
    "htf_slow": 20,
    "htf_sep_atr": 0.0,
    "ses_start_h": 9,
    "or_bars": 4,
    "entry_end_h": 14,
    "buf_atr": 0.15,
    "min_range_atr": 0.0,
    "max_range_atr": 99.0,
    "mode": "follow",
    "sides": "both",
    "require_htf": 1,
    "min_atr": 0.0,
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
    """HTF direction frame, indexed by the HTF bar's CLOSE time."""
    hm = _htf_minutes(p)
    h = (
        df.resample(f"{hm}min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    mode = str(p["htf_mode"]).strip().lower()
    fast, slow = int(p["htf_fast"]), int(p["htf_slow"])

    if mode == "donch":
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

    out = pd.DataFrame({"dir": d.astype(float)})
    out.index = h.index + pd.Timedelta(minutes=hm)
    return out


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    bm = _bar_minutes(scfg)
    out["atr"] = atr(out, scfg.atr_period)

    # ---- session opening range, exposed only after it has fully closed ----
    hours = out.index.hour
    day = out.index.normalize()
    start_h, or_bars = int(p["ses_start_h"]), int(p["or_bars"])
    in_ses = pd.Series(hours >= start_h, index=out.index)
    rank = in_ses.astype(int).groupby(day).cumsum()
    is_or = (in_ses & (rank <= or_bars) & (rank >= 1)).to_numpy()
    or_hi = out["high"].where(is_or).groupby(day).transform("max")
    or_lo = out["low"].where(is_or).groupby(day).transform("min")
    avail = (in_ses & (rank > or_bars)).to_numpy()
    out["or_hi"] = or_hi.where(avail)
    out["or_lo"] = or_lo.where(avail)

    # ---- higher-timeframe bias on COMPLETED HTF bars only ----
    htf = _htf_direction(out, scfg, p)
    mapped = htf.reindex(out.index + pd.Timedelta(minutes=bm), method="ffill")
    out["htf_dir"] = mapped["dir"].to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    bm = _bar_minutes(scfg)
    ratio = max(1, _htf_minutes(p) // bm)
    htf_bars = 4 * max(int(p["htf_fast"]), int(p["htf_slow"]), scfg.atr_period) + 10
    bars_per_day = max(1, 1440 // bm)
    return int(htf_bars * ratio + 2 * bars_per_day + 5 * scfg.atr_period + 10)


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

    # the order is sent at this bar's close == the next bar's open
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=_bar_minutes(scfg))
    if not (int(p["ses_start_h"]) <= entry_ts.hour < int(p["entry_end_h"])):
        return flat("outside entry window")

    hi, lo = last["or_hi"], last["or_lo"]
    if not (np.isfinite(hi) and np.isfinite(lo)):
        return flat("opening range not complete")
    width = float(hi) - float(lo)
    if width < float(p["min_range_atr"]) * a:
        return flat("range too narrow")
    if width > float(p["max_range_atr"]) * a:
        return flat("range too wide")

    buf = float(p["buf_atr"]) * a
    up, dn = float(hi) + buf, float(lo) - buf
    c_last, c_prev = float(last["close"]), float(prev["close"])
    broke_up = c_prev <= up < c_last
    broke_dn = c_prev >= dn > c_last
    if not (broke_up or broke_dn):
        return flat("no range break")

    fade = str(p["mode"]).strip().lower() == "fade"
    if broke_up:
        direction = SHORT if fade else LONG
        desc = f"OR hi {float(hi):.2f}+{buf:.2f}"
    else:
        direction = LONG if fade else SHORT
        desc = f"OR lo {float(lo):.2f}-{buf:.2f}"

    if int(p["require_htf"]):
        bias = float(last["htf_dir"]) if np.isfinite(last["htf_dir"]) else 0.0
        if bias == 0.0:
            return flat("no HTF bias")
        if bias * direction < 0:
            return flat("break against HTF bias")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    side = "long" if direction == LONG else "short"
    return Signal(
        direction, c_last, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
        f"MTF {p['htf']} bias + OR {p['mode']} {side}: {c_last:.2f} broke {desc}",
    )
