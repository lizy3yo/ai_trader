"""User-supplied strategy: 8/13/21 EMA pullback with a 200 EMA trend filter.

Rules as given by the user (2026-11-08), with the vague parts fixed BEFORE testing:
  Trend (buy): close > EMA200; EMA8 > EMA13 > EMA21; each of the three is higher
    than `slope_n` bars ago ("trending up"); gaps EMA8-EMA13 and EMA13-EMA21 are
    both >= `gap_atr` x ATR ("space in between").  Sell = mirror.
  Trigger (buy): while the trend holds, a DOWN candle (close < open) closes
    between EMA21 and EMA13. A newer qualifying candle replaces the trigger.
  Invalid: any close below EMA21 before entry cancels the setup; a setup also
    expires `expiry` bars after the first trigger (not specified by the user).
  Entry (buy), `entry` param:
    "ema8"  = a later candle closes back above EMA8          (user's way 1)
    "break5" = a later candle closes above the highest high of the 5 candles
               ending at the trigger                          (user's way 3; the
               backtester has no stop orders, so the break must be a CLOSE)
    Way 2 ("community indicator") is not specified and not tested.
    Fill = next bar's open (live: the order goes in when the bar closes).
  Stop: trigger candle low - `buf_atr` x ATR ("2 pips" has no common meaning
    across gold / BTC / indices).  Target: `cfg.atr_tp_mult` is NOT used; the
    target is `rr` x the stop distance. Scale-out (half at 1R, stop to
    break-even, rest to 3R) is done by the backtester via cfg.partial_at_r.
  stop_mode (added on request): "trigger" (user's rule), "atr" (stop_atr x ATR),
    "max" (trigger stop, but never tighter than stop_atr x ATR).
  Not modelled: news filter, Friday 16:00 close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat

DEFAULTS = {"entry": "ema8", "side": "both", "slope_n": 3, "gap_atr": 0.0,
            "buf_atr": 0.1, "expiry": 10, "rr": 3.0,
            "stop_mode": "trigger", "stop_atr": 2.0}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    e8, e13, e21, e200 = (ema(out["close"], n).to_numpy() for n in (8, 13, 21, 200))
    a = atr(out, scfg.atr_period).to_numpy()
    k, g = int(p["slope_n"]), float(p["gap_atr"])
    n = len(out)
    sig = np.zeros(n, dtype=np.int8)
    slp = np.full(n, np.nan)
    sides = {"long": (1,), "short": (-1,), "both": (1, -1)}[p["side"]]
    hi5 = pd.Series(h).rolling(5).max().to_numpy()
    lo5 = pd.Series(l).rolling(5).min().to_numpy()
    # one state machine per side; all decisions use bars <= i only
    state = {d: None for d in sides}   # d -> dict(trig_i, first_i, stop, level)
    for i in range(max(k, 200), n):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        for d in sides:
            if d == 1:
                trend = (c[i] > e200[i] and e8[i] > e13[i] > e21[i]
                         and e8[i] > e8[i - k] and e13[i] > e13[i - k] and e21[i] > e21[i - k]
                         and e8[i] - e13[i] >= g * a[i] and e13[i] - e21[i] >= g * a[i])
                trig = trend and c[i] < o[i] and e21[i] < c[i] < e13[i]
                broke = c[i] < e21[i]
            else:
                trend = (c[i] < e200[i] and e8[i] < e13[i] < e21[i]
                         and e8[i] < e8[i - k] and e13[i] < e13[i - k] and e21[i] < e21[i - k]
                         and e13[i] - e8[i] >= g * a[i] and e21[i] - e13[i] >= g * a[i])
                trig = trend and c[i] > o[i] and e13[i] < c[i] < e21[i]
                broke = c[i] > e21[i]
            st = state[d]
            if st is not None:
                if broke or i - st["first_i"] > int(p["expiry"]):
                    state[d] = st = None
                elif i > st["trig_i"]:
                    fire = ((c[i] > e8[i] if d == 1 else c[i] < e8[i]) if p["entry"] == "ema8"
                            else (c[i] > st["level"] if d == 1 else c[i] < st["level"]))
                    if fire and sig[i] == 0:
                        sig[i], slp[i] = d, st["stop"]
                        state[d] = None
                        continue
            if trig:
                stop = l[i] - float(p["buf_atr"]) * a[i] if d == 1 else h[i] + float(p["buf_atr"]) * a[i]
                level = hi5[i] if d == 1 else lo5[i]
                first = st["first_i"] if st is not None else i
                state[d] = {"trig_i": i, "first_i": first, "stop": stop, "level": level}
    out["atr"] = a
    out["us_sig"] = sig
    out["us_stop"] = slp
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 210 + int(_p(scfg)["slope_n"])


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    d = int(last["us_sig"])
    if d == 0:
        return flat("no setup")
    close, stop = float(last["close"]), float(last["us_stop"])
    sl = (close - stop) if d == 1 else (stop - close)
    atr_sl = float(p["stop_atr"]) * float(last["atr"])
    # stop_mode: "trigger" = user's rule (beyond the trigger candle);
    # "atr" = fixed stop_atr x ATR; "max" = trigger stop but never tighter than stop_atr x ATR
    if p["stop_mode"] == "atr":
        sl = atr_sl
    elif p["stop_mode"] == "max":
        sl = max(sl, atr_sl) if np.isfinite(sl) else atr_sl
    if not np.isfinite(sl) or sl <= 0:
        return flat("stop on the wrong side")
    return Signal(LONG if d == 1 else SHORT, close, sl, float(p["rr"]) * sl,
                  float(last["atr"]), f"EMA 8/13/21 pullback ({p['entry']})")
