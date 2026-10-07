"""hw_bb_mid — Bollinger midline touch in trend direction (high win-rate family).

Idea: the Bollinger middle band (SMA of bb_period) acts as dynamic value in a
trend. When price, trading above the midline in an uptrend, pulls back so its
LOW touches the midline while the CLOSE holds above it, buy the touch with a
near TP (cfg.atr_tp_mult < cfg.atr_sl_mult). Mirror for shorts.

Long setup (short is the mirror):
  - close > EMA(trend)                       (regime up)
  - midline rising: mid > mid `slope_bars` ago
  - prev close > prev mid                    (was above value before the dip)
  - this bar low <= mid + tol*ATR            (touched the midline)
  - this bar close > mid                     (touch held)
  - optional band-momentum confirm: some high of the previous `band_lookback`
    bars reached the UPPER band (there was real thrust before this pullback);
    computed with .shift(1) so the current bar never confirms itself.

Params (scfg.params):
  bb_period     : int, Bollinger period for the midline/bands (default 20)
  bb_dev        : float, band width in std-devs (only used by the confirm)
  mid_atr_tol   : float, ATR fraction the low may stop short of the midline
  slope_bars    : int, how many bars back to measure midline slope (0 = off)
  band_lookback : int, bars to look back for an outer-band touch (0 = off)
Shared knobs used from scfg: ema_trend, atr_period, atr_sl_mult, atr_tp_mult.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "bb_period": 20,
    "bb_dev": 2.0,
    "mid_atr_tol": 0.0,
    "slope_bars": 3,
    "band_lookback": 0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    period = int(p["bb_period"])
    dev = float(p["bb_dev"])
    lb = int(p["band_lookback"])
    out = df.copy()
    out["mid"] = sma(out["close"], period)
    sd = rolling_std(out["close"], period)
    out["upper"] = out["mid"] + dev * sd
    out["lower"] = out["mid"] - dev * sd
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    if lb > 0:
        # did any of the PRIOR lb bars tag the outer band? (.shift(1): the
        # current bar can never confirm its own entry)
        out["hit_up"] = (out["high"] >= out["upper"]).rolling(lb).max().shift(1)
        out["hit_dn"] = (out["low"] <= out["lower"]).rolling(lb).max().shift(1)
    else:
        out["hit_up"] = 1.0
        out["hit_dn"] = 1.0
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    return max(
        5 * scfg.ema_trend,
        int(p["bb_period"]) + int(p["band_lookback"]) + int(p["slope_bars"]),
        scfg.atr_period,
    ) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = {**DEFAULTS, **scfg.params}
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")
    if not pd.notna(last["mid"]) or not pd.notna(prev["mid"]):
        return flat("bands warming up")

    slope_bars = int(p["slope_bars"])
    if slope_bars > 0:
        ref = ind_window.iloc[-1 - slope_bars]
        if not pd.notna(ref["mid"]):
            return flat("slope ref warming up")
        slope_up = last["mid"] > ref["mid"]
        slope_dn = last["mid"] < ref["mid"]
    else:
        slope_up = slope_dn = True

    a = float(last["atr"])
    tol = float(p["mid_atr_tol"]) * a
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a

    if (
        last["close"] > last["ema_trend"]
        and slope_up
        and prev["close"] > prev["mid"]
        and last["low"] <= last["mid"] + tol
        and last["close"] > last["mid"]
        and float(last["hit_up"]) > 0
    ):
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"BB-mid buy: low touched midline({int(p['bb_period'])}) in uptrend",
        )
    if (
        last["close"] < last["ema_trend"]
        and slope_dn
        and prev["close"] < prev["mid"]
        and last["high"] >= last["mid"] - tol
        and last["close"] < last["mid"]
        and float(last["hit_dn"]) > 0
    ):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"BB-mid sell: high touched midline({int(p['bb_period'])}) in downtrend",
        )
    return flat("no midline touch")
