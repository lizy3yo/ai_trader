"""Liquidity-sweep / stop-hunt reversal family (prefix lq_).

Mechanizes the ICT / Smart-Money "liquidity sweep" narrative plus the classic
Connors Turtle Soup as ONE causal state machine over a reference level pair:

    level_mode = "pday" : yesterday's high/low. Daily aggregates are computed
                          per completed day and shifted one day, so every bar
                          of today sees only the FINISHED prior day.
    level_mode = "asia" : today's Asia-session range (bars whose OPEN hour is
                          in [range_start_h, range_end_h)), visible only on
                          bars at/after range_end_h (ses_orb double-mask
                          pattern). Faded during London = ICT "Judas swing".
    level_mode = "roll" : rolling roll_n-bar extremes via the shift(1)-safe
                          donchian() helper (classic Turtle Soup).

SHORT sweep signal on the newest closed bar i (LONG is the mirror on lvl_lo):
  * some bar within the last max_beyond bars had high > lvl_hi + sweep_atr*ATR
      (the liquidity grab beyond the level)
  * close[i] < lvl_hi - reclaim_atr*ATR      (price CLOSED back inside)
  * freshness: bar i-1 closed beyond lvl_hi OR bar i itself wicked beyond —
      i.e. this is the FIRST reclaim close, not a stale one
  * at most max_out_bars consecutive CLOSES beyond the level immediately
      before the reclaim (more = genuine breakout, not a stop hunt -> skip)

Entry window is governed by the harness "session.window" key (the backtester
gates both signal evaluation and fills on the session clock).

Params (scfg.params):
    level_mode    : "pday" | "asia" | "roll"
    roll_n        : rolling extreme lookback in bars      (roll mode)
    range_start_h / range_end_h : Asia range hours        (asia mode)
    sweep_atr     : min penetration beyond the level, in ATRs
    reclaim_atr   : close must be back inside by this many ATRs
    max_beyond    : penetration must have happened within this many bars
    max_out_bars  : max consecutive closes beyond the level before reclaim
    mode          : "fade" (ICT claim: reverse after the reclaim) or
                    "follow" (mirror test: sweep+reclaim is a pullback in a
                    continuing move -> trade back IN the sweep direction)
    trend_filter  : if true, longs need close > EMA(ema_trend), shorts below
    sides         : "both" | "long" | "short"
    sl_mode       : "atr" (scfg.atr_sl_mult/atr_tp_mult x ATR) or
                    "wick" (SL beyond the sweep extreme + wick_pad_atr*ATR,
                            TP = rr x SL distance)
    wick_pad_atr  : stop padding beyond the sweep extreme  (wick mode)
    wick_max_atr  : skip if wick-mode SL distance exceeds this many ATRs
    rr            : reward:risk multiple                   (wick mode)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr

DEFAULTS = {
    "level_mode": "pday",
    "roll_n": 24,
    "range_start_h": 1,
    "range_end_h": 9,
    "sweep_atr": 0.0,
    "reclaim_atr": 0.0,
    "max_beyond": 3,
    "max_out_bars": 2,
    "mode": "fade",
    "trend_filter": False,
    "sides": "both",
    "sl_mode": "atr",
    "wick_pad_atr": 0.25,
    "wick_max_atr": 3.0,
    "rr": 1.5,
}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def _consec_true(flag: pd.Series) -> pd.Series:
    """Length of the run of consecutive True values ending at each row
    (0 where the row itself is False). Purely cumulative -> causal."""
    f = flag.fillna(False).astype(bool)
    return f.groupby((~f).cumsum()).cumsum().astype(float)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    mode = str(p["level_mode"])
    if mode == "pday":
        day = out.index.normalize()
        # per-day aggregates shifted one day: today only sees FINISHED days
        dhi = out["high"].groupby(day).max().shift(1)
        dlo = out["low"].groupby(day).min().shift(1)
        out["lvl_hi"] = dhi.reindex(day).to_numpy()
        out["lvl_lo"] = dlo.reindex(day).to_numpy()
    elif mode == "asia":
        hours = out.index.hour
        day = out.index.normalize()
        rs, re_ = int(p["range_start_h"]), int(p["range_end_h"])
        in_range = (hours >= rs) & (hours < re_)
        rng_hi = out["high"].where(in_range).groupby(day).transform("max")
        rng_lo = out["low"].where(in_range).groupby(day).transform("min")
        # causality: the range only becomes visible once every range bar of
        # the day has closed (bars whose OPEN hour is at/after range_end_h)
        complete = hours >= re_
        out["lvl_hi"] = rng_hi.where(complete)
        out["lvl_lo"] = rng_lo.where(complete)
    elif mode == "roll":
        hh, ll = donchian(out, int(p["roll_n"]))   # shift(1)-safe
        out["lvl_hi"] = hh
        out["lvl_lo"] = ll
    else:
        raise ValueError(f"lq_sweep: unknown level_mode {mode!r}")

    margin = float(p["sweep_atr"]) * out["atr"]
    beyond_hi = out["high"] > out["lvl_hi"] + margin   # NaN level -> False
    beyond_lo = out["low"] < out["lvl_lo"] - margin
    mb = max(1, int(p["max_beyond"]))
    out["beyond_hi"] = beyond_hi.astype(float)
    out["beyond_lo"] = beyond_lo.astype(float)
    out["swept_hi"] = beyond_hi.astype(float).rolling(mb, min_periods=1).max()
    out["swept_lo"] = beyond_lo.astype(float).rolling(mb, min_periods=1).max()

    out_hi = out["close"] > out["lvl_hi"]              # closed beyond level
    out_lo = out["close"] < out["lvl_lo"]
    out["out_hi"] = out_hi.astype(float)
    out["out_lo"] = out_lo.astype(float)
    out["consec_out_hi"] = _consec_true(out_hi)
    out["consec_out_lo"] = _consec_true(out_lo)

    # sweep extreme over the lookback (includes current bar) for wick stops
    out["sw_hi_ext"] = out["high"].rolling(mb, min_periods=1).max()
    out["sw_lo_ext"] = out["low"].rolling(mb, min_periods=1).min()
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    bars_per_day = 1440 // _bar_minutes(scfg)
    if str(p["level_mode"]) == "roll":
        need_lvl = int(p["roll_n"]) + 2
    else:
        need_lvl = 2 * bars_per_day
    trend_part = 5 * scfg.ema_trend if p["trend_filter"] else 0
    return max(need_lvl + 5 * scfg.atr_period, trend_part) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    a = float(last["atr"])
    c = float(last["close"])
    reclaim = float(p["reclaim_atr"]) * a
    max_out = int(p["max_out_bars"])

    lvl_hi, lvl_lo = float(last["lvl_hi"]), float(last["lvl_lo"])

    short_sig = (
        np.isfinite(lvl_hi)
        and last["swept_hi"] > 0                      # grab happened recently
        and c < lvl_hi - reclaim                      # closed back inside
        and (prev["out_hi"] > 0 or last["beyond_hi"] > 0)   # fresh reclaim
        and float(prev["consec_out_hi"]) <= max_out   # hunt, not breakout
    )
    long_sig = (
        np.isfinite(lvl_lo)
        and last["swept_lo"] > 0
        and c > lvl_lo + reclaim
        and (prev["out_lo"] > 0 or last["beyond_lo"] > 0)
        and float(prev["consec_out_lo"]) <= max_out
    )
    if short_sig and long_sig:
        return flat("both sides swept — no bias")
    if not (short_sig or long_sig):
        return flat("no sweep reclaim")

    follow = str(p["mode"]) == "follow"
    if short_sig:                       # high-side sweep was reclaimed
        direction = LONG if follow else SHORT
    else:                               # low-side sweep was reclaimed
        direction = SHORT if follow else LONG
    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    if p["trend_filter"]:
        trend = float(last["ema_trend"])
        if direction == LONG and c <= trend:
            return flat("sweep against trend filter")
        if direction == SHORT and c >= trend:
            return flat("sweep against trend filter")

    if str(p["sl_mode"]) == "wick":
        # stop beyond the lookback extreme on the adverse side: the sweep
        # wick itself (fade) or the pullback extreme (follow)
        if direction == SHORT:
            sl_dist = float(last["sw_hi_ext"]) - c + float(p["wick_pad_atr"]) * a
        else:
            sl_dist = c - float(last["sw_lo_ext"]) + float(p["wick_pad_atr"]) * a
        if sl_dist <= 0:
            return flat("degenerate wick stop")
        if sl_dist > float(p["wick_max_atr"]) * a:
            return flat("sweep too wide for wick stop")
        tp_dist = float(p["rr"]) * sl_dist
    else:
        sl_dist = scfg.atr_sl_mult * a
        tp_dist = scfg.atr_tp_mult * a

    lvl_mode = str(p["level_mode"])
    swept_side = "high" if short_sig else "low"
    lvl = lvl_hi if short_sig else lvl_lo
    verb = "follow" if follow else "fade"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"LQ {verb} sweep of {lvl_mode} {swept_side} {lvl:.2f}, close {c:.2f}")
