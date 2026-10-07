"""EMA-pullback continuation (advanced trend-following family).

Idea: in an established uptrend (EMA(fast) > EMA(slow), close > EMA(trend)),
wait for price to pull back to/below the fast EMA (previous bar CLOSED at or
below EMA(fast)), then enter LONG on the first bar that closes back above it.
Entries happen at better prices than crossovers, which tends to raise win
rate. Short is the exact mirror.

Optional gate: trend_gap_atr requires |EMA(fast)-EMA(slow)| >= gap * ATR so
we only buy pullbacks inside a trend with real separation (an ADX-lite).

Optional drift filters (all default OFF, all causal):
  long_only  : 1 = never short (gold's secular drift is up)
  mom_bars   : N > 0 requires close > close[N bars ago] for longs (mirror
               for shorts) — multi-bar time-series momentum confirmation
  min_atr    : absolute $ ATR floor — skip dead-volatility regimes where
               the spread eats the target
  regime_n   : N > 0 additionally requires a FRESH BREAKOUT regime: some
               close within the last regime_k bars exceeded the prior
               N-bar high (mirror: below the prior N-bar low for shorts).
               Buys the first pullback after real strength, instead of
               any pullback in a drifting EMA stack.
  regime_k   : how many bars a breakout keeps the regime "fresh" (default 24)

Shared knobs used from StrategyCfg: ema_fast, ema_slow, ema_trend,
atr_period, atr_sl_mult, atr_tp_mult.
params: trend_gap_atr, long_only, mom_bars, min_atr.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr

DEFAULTS = {
    "trend_gap_atr": 0.0,   # min (ema_fast-ema_slow)/ATR separation to trade
    "long_only": 0,
    "mom_bars": 0,
    "min_atr": 0.0,
    "regime_n": 0,
    "regime_k": 24,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["ema_fast"] = ema(out["close"], scfg.ema_fast)
    out["ema_slow"] = ema(out["close"], scfg.ema_slow)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    n = int(p["regime_n"])
    if n > 0:
        k = int(p["regime_k"])
        hh, ll = donchian(out, n)          # prior-N-bar extremes (shifted)
        new_hi = (out["close"] > hh).astype(float)
        new_lo = (out["close"] < ll).astype(float)
        # regime is ON if any of the last k bars (incl. this closed bar)
        # closed beyond the prior N-bar extreme — rolling over past bars only
        out["reg_up"] = new_hi.rolling(k, min_periods=1).max()
        out["reg_dn"] = new_lo.rolling(k, min_periods=1).max()
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    return max(5 * scfg.ema_trend, 5 * scfg.ema_slow, scfg.atr_period,
               int(p["mom_bars"]) + 2,
               int(p["regime_n"]) + int(p["regime_k"]) + 2) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = {**DEFAULTS, **scfg.params}
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat(f"ATR {a:.2f} below floor {p['min_atr']}")
    gap_need = float(p["trend_gap_atr"]) * a
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a

    mom_bars = int(p["mom_bars"])
    mom_up = mom_down = True
    if mom_bars > 0:
        if len(ind_window) < mom_bars + 1:
            return flat("not enough bars for momentum filter")
        ref = float(ind_window["close"].iloc[-1 - mom_bars])
        mom_up = float(last["close"]) > ref
        mom_down = float(last["close"]) < ref

    up_trend = (
        last["ema_fast"] > last["ema_slow"]
        and last["close"] > last["ema_trend"]
        and (last["ema_fast"] - last["ema_slow"]) >= gap_need
    )
    dn_trend = (
        last["ema_fast"] < last["ema_slow"]
        and last["close"] < last["ema_trend"]
        and (last["ema_slow"] - last["ema_fast"]) >= gap_need
    )

    # pullback bar closed across the fast EMA, current bar closed back through
    long_trigger = prev["close"] <= prev["ema_fast"] and last["close"] > last["ema_fast"]
    short_trigger = prev["close"] >= prev["ema_fast"] and last["close"] < last["ema_fast"]

    if int(p["regime_n"]) > 0:
        long_trigger = long_trigger and float(last["reg_up"]) > 0
        short_trigger = short_trigger and float(last["reg_dn"]) > 0

    if up_trend and long_trigger and mom_up:
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"pullback long: reclaim EMA{scfg.ema_fast} in EMA{scfg.ema_slow}"
            f"/{scfg.ema_trend} uptrend",
        )
    if dn_trend and short_trigger and mom_down and not int(p["long_only"]):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"pullback short: lose EMA{scfg.ema_fast} in EMA{scfg.ema_slow}"
            f"/{scfg.ema_trend} downtrend",
        )
    return flat("no pullback signal")
