"""Session-range breakout with daily regime gates (gid_ family, intraday).

Improves the Asia-range / London-open breakout (ses_orb) and adds the same
mechanic for the NY open (break of the London-morning range). Server clock
is EET (UTC+2/+3, EU DST): London 08:00 = server 10:00, NY 08:30 data =
server 15:30, London PM fix 15:00 London = server 17:00.

Range  = bars whose OPEN hour is in [range_start_h, range_end_h) of the same
         server day; visible only on bars at/after range_end_h (causal).
Entry  = first close beyond range +/- buf_atr*ATR while the fill moment
         (signal bar close) is inside [entry_start_hm, entry_end_hm).
         one_shot=1 -> at most one break per day (first one only).
Gates (all from COMPLETED prior days, mapped onto today's bars):
  regime   "none" | "rng_rel": yesterday's range / median(range, 20 prior
           days) >= rr_min  | "atrp": 14-day mean daily range as % of price
           is in the top (1-q_min) of its trailing 250-day distribution
  dtrend   1 = prior close vs SMA(dtrend_n) of daily closes must agree with
           the trade direction
  rng_max  skip if the range is wider than rng_max x daily ATR (exhausted)
  skip_dows weekdays to skip (0=Mon..4=Fri)
  max_cost_atr skip if cost_ref ($ spread) / ATR exceeds this (trade only
           when the spread is a small slice of the bar range)
Stops: sl_mode "atr" = atr_sl_mult x ATR; "range" = beyond the opposite
       range side (floored at 0.5 ATR, capped at atr_sl_mult x ATR);
       "mid" = range midpoint (same floor/cap). TP = atr_tp_mult x ATR.
Exit before rollover with session.flat_at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr

DEFAULTS = {
    "range_start_h": 1, "range_end_h": 9,
    "entry_start_hm": "09:00", "entry_end_hm": "12:00",
    "buf_atr": 0.2, "one_shot": 1, "sides": "both",
    "regime": "none", "rr_min": 1.0, "q_min": 0.5,
    "dtrend": 0, "dtrend_n": 20, "rng_max": 99.0,
    "sl_mode": "atr", "skip_dows": [],
    "max_cost_atr": 99.0, "cost_ref": 0.55,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def _hm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    hours = out.index.hour
    day = out.index.normalize()

    in_rng = (hours >= int(p["range_start_h"])) & (hours < int(p["range_end_h"]))
    hi = out["high"].where(in_rng).groupby(day).cummax()
    lo = out["low"].where(in_rng).groupby(day).cummin()
    hi = hi.groupby(day).ffill()
    lo = lo.groupby(day).ffill()
    n = pd.Series(in_rng.astype(float), index=out.index).groupby(day).cumsum()
    done = hours >= int(p["range_end_h"])
    out["rng_hi"] = hi.where(done)
    out["rng_lo"] = lo.where(done)
    out["rng_n"] = n.where(done)
    # earlier breaks today (for one_shot): close beyond range +/- buf*ATR on a
    # PREVIOUS bar of the same day after the range closed (causal cumsum)
    buf = float(p["buf_atr"]) * out["atr"]
    brk = ((out["close"] > out["rng_hi"] + buf) | (out["close"] < out["rng_lo"] - buf)).astype(float)
    out["prior_brk"] = brk.groupby(day).cumsum() - brk

    # daily stats from COMPLETED days only (shift(1) at the daily level)
    g = out.groupby(day)
    daily = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min(), "c": g["close"].last()})
    daily = daily[daily.index.dayofweek < 5]
    rng = daily["hi"] - daily["lo"]
    d_atr = rng.rolling(14).mean()
    feats = pd.DataFrame(index=daily.index)
    feats["rng_rel"] = rng / rng.rolling(20).median()
    atrp = d_atr / daily["c"]
    feats["atrp_q"] = atrp.rolling(250, min_periods=120).rank(pct=True)
    feats["dtr"] = np.sign(daily["c"] - daily["c"].rolling(int(p["dtrend_n"])).mean())
    feats["d_atr"] = d_atr
    feats = feats.shift(1)                       # yesterday's values for today
    mapped = feats.reindex(day, method="ffill")
    for col in feats.columns:
        out["d_" + col] = mapped[col].to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 2 * 1440 // _bar_minutes(scfg) + 5 * scfg.atr_period + 10


def cost_gate_fails(a: float, p: dict) -> bool:
    return float(p["cost_ref"]) / a > float(p["max_cost_atr"])


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last, prev = ind_window.iloc[-1], ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    bm = _bar_minutes(scfg)
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=bm)
    mins = entry_ts.hour * 60 + entry_ts.minute
    if not (_hm(p["entry_start_hm"]) <= mins < _hm(p["entry_end_hm"])):
        return flat("outside entry window")
    if entry_ts.weekday() in p["skip_dows"]:
        return flat("skipped weekday")
    hi, lo, n = last["rng_hi"], last["rng_lo"], last["rng_n"]
    if not (np.isfinite(hi) and np.isfinite(lo)) or n < 2:
        return flat("no range")
    hi, lo = float(hi), float(lo)
    a = float(last["atr"])
    buf = float(p["buf_atr"]) * a
    up, dn = hi + buf, lo - buf
    c, cp = float(last["close"]), float(prev["close"])

    if int(p["one_shot"]):
        if float(last["prior_brk"]) > 0:
            return flat("not the first break")

    if cp <= up < c:
        d = LONG
    elif cp >= dn > c:
        d = SHORT
    else:
        return flat("no break")
    sides = str(p["sides"])
    if (d == LONG and sides == "short") or (d == SHORT and sides == "long"):
        return flat("side disabled")

    # cost gate: skip when the live spread (cost_ref $) is a big slice of ATR
    if cost_gate_fails(a, p):
        return flat("spread too large vs ATR")
    datr = last["d_d_atr"]
    if float(p["rng_max"]) < 99 and np.isfinite(datr) and (hi - lo) > float(p["rng_max"]) * float(datr):
        return flat("range too wide")
    reg = str(p["regime"])
    if reg == "rng_rel":
        v = last["d_rng_rel"]
        if not np.isfinite(v) or v < float(p["rr_min"]):
            return flat("quiet prior day")
    elif reg == "atrp":
        v = last["d_atrp_q"]
        if not np.isfinite(v) or v < float(p["q_min"]):
            return flat("low vol regime")
    if int(p["dtrend"]):
        v = last["d_dtr"]
        if not np.isfinite(v) or v * d <= 0:
            return flat("against daily trend")

    cap = scfg.atr_sl_mult * a
    mode = str(p["sl_mode"])
    if mode == "range":
        sl = (c - lo) if d == LONG else (hi - c)
        sl = min(max(sl, 0.5 * a), cap)
    elif mode == "mid":
        mid = 0.5 * (hi + lo)
        sl = abs(c - mid)
        sl = min(max(sl, 0.5 * a), cap)
    else:
        sl = cap
    return Signal(d, c, sl, scfg.atr_tp_mult * a, a,
                  f"range break {'up' if d == LONG else 'down'} {lo:.2f}-{hi:.2f}")
