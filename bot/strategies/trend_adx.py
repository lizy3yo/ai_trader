"""ADX-gated EMA crossover (advanced trend-following family).

Classic EMA(fast)x EMA(slow) crossover with EMA(trend) direction filter, but
the trade is only taken when Wilder's ADX says trend strength is real
(ADX >= adx_min) and the DI lines agree with the trade direction
(+DI > -DI for longs) when di_confirm=1. Filters out the chop where naked
crossovers bleed.

All smoothing is Wilder ewm (alpha=1/period) on past data only — causal.

Shared knobs used from StrategyCfg: ema_fast, ema_slow, ema_trend,
atr_period, atr_sl_mult, atr_tp_mult.
params: adx_period, adx_min, di_confirm (0/1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "adx_period": 14,
    "adx_min": 25.0,
    "di_confirm": 1,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _dmi(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder +DI, -DI, ADX. Uses only diffs and ewm of past bars (causal)."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["ema_fast"] = ema(out["close"], scfg.ema_fast)
    out["ema_slow"] = ema(out["close"], scfg.ema_slow)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    plus_di, minus_di, adx = _dmi(out, int(p["adx_period"]))
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = adx
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    span = max(scfg.ema_trend, scfg.ema_slow, int(p["adx_period"]), scfg.atr_period)
    return 5 * span + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a

    if float(last["adx"]) < float(p["adx_min"]):
        return flat(f"ADX {last['adx']:.0f} < {p['adx_min']}")

    di_confirm = bool(int(p["di_confirm"]))
    crossed_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    crossed_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    if (
        crossed_up
        and last["close"] > last["ema_trend"]
        and (not di_confirm or last["plus_di"] > last["minus_di"])
    ):
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"EMA{scfg.ema_fast}x{scfg.ema_slow} up, ADX {last['adx']:.0f}",
        )
    if (
        crossed_dn
        and last["close"] < last["ema_trend"]
        and (not di_confirm or last["minus_di"] > last["plus_di"])
    ):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"EMA{scfg.ema_fast}x{scfg.ema_slow} down, ADX {last['adx']:.0f}",
        )
    return flat("no gated crossover")
