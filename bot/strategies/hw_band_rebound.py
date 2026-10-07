"""hw_band_rebound — Bollinger OUTER-band touch-and-hold (high win-rate family).

The midline-touch variant (hw_bb_mid) showed win rates BELOW the random-walk
touch probability of its TP/SL geometry — shallow pullbacks kept drifting
against the entry. This module buys the DEEPER dip: price tags the lower
Bollinger band and closes back above it (the extreme got rejected), where
short-horizon snap-back is strongest. TP stays smaller than SL
(cfg.atr_tp_mult < cfg.atr_sl_mult) for the many-small-wins profile.

Long entry (short is the mirror on the upper band):
  - regime "trend": close > EMA(trend)  (buy dips only in an up regime)
    regime "any"  : no regime filter    (pure fade of the band extreme)
  - prev bar did NOT touch the band (fresh touch, not a grind-through)
  - this bar low <= lower_band + tol*ATR
  - this bar close > lower_band          (the poke was rejected)

Params (scfg.params):
  bb_period : int   Bollinger period (default 20)
  bb_dev    : float band width in std-devs (default 2.0)
  regime    : "trend" | "any"
  side      : "both" | "long" | "short"
  tol_atr   : float ATR fraction the extreme may stop short of the band
Shared knobs used from scfg: ema_trend, atr_period, atr_sl_mult, atr_tp_mult.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "bb_period": 20,
    "bb_dev": 2.0,
    "regime": "trend",
    "side": "both",
    "tol_atr": 0.0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    period = int(p["bb_period"])
    dev = float(p["bb_dev"])
    out = df.copy()
    mid = sma(out["close"], period)
    sd = rolling_std(out["close"], period)
    out["bb_up"] = mid + dev * sd
    out["bb_dn"] = mid - dev * sd
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    return max(5 * scfg.ema_trend, int(p["bb_period"]) + 2, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = {**DEFAULTS, **scfg.params}
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (pd.notna(last["bb_dn"]) and pd.notna(prev["bb_dn"])):
        return flat("bands warming up")

    a = float(last["atr"])
    tol = float(p["tol_atr"]) * a
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    regime = str(p["regime"])
    side = str(p["side"])

    long_regime = last["close"] > last["ema_trend"] if regime == "trend" else True
    short_regime = last["close"] < last["ema_trend"] if regime == "trend" else True

    if (
        side in ("both", "long")
        and long_regime
        and prev["low"] > prev["bb_dn"]
        and last["low"] <= last["bb_dn"] + tol
        and last["close"] > last["bb_dn"]
    ):
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"lower-band rejection ({regime} regime)",
        )
    if (
        side in ("both", "short")
        and short_regime
        and prev["high"] < prev["bb_up"]
        and last["high"] >= last["bb_up"] - tol
        and last["close"] < last["bb_up"]
    ):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"upper-band rejection ({regime} regime)",
        )
    return flat("no band rejection")
