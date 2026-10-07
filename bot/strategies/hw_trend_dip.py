"""hw_trend_dip — trend-dip continuation aimed at HIGH WIN RATE geometry.

Idea: in an established trend (EMA(fast) vs EMA(slow) aligned AND price on the
right side of EMA(trend)), wait for a pullback whose LOW touches the fast EMA
and whose CLOSE reclaims it (the dip got bought). Enter in trend direction with
a TP smaller than the SL (cfg.atr_tp_mult < cfg.atr_sl_mult), so most trades
reach the near target: many small wins, rare bigger losses.

Long setup (short is the mirror):
  - ema_fast > ema_slow                     (short-term trend up)
  - close > ema_trend                       (regime up)
  - prev close  > prev ema_fast             (we were riding above the EMA)
  - this bar low <= ema_fast + tol*ATR      (pullback reached the EMA)
  - this bar close > ema_fast               (dip rejected / reclaimed)
  - RSI < rsi_overbought (optional filter)  (not chasing a stretched move)

Params (scfg.params):
  dip_atr_tol : float, how close (in ATRs) the low must get to the fast EMA.
                0.0 = must actually touch/pierce it; 0.25 = within a quarter ATR.
  use_rsi     : 1/0, apply the RSI stretch filter (levels from scfg).
Shared knobs used from scfg: ema_fast (pullback EMA), ema_slow, ema_trend,
rsi_period, rsi_overbought, rsi_oversold, atr_period, atr_sl_mult, atr_tp_mult.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rsi, valid_atr

DEFAULTS = {
    "dip_atr_tol": 0.0,
    "use_rsi": 1,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], scfg.ema_fast)
    out["ema_slow"] = ema(out["close"], scfg.ema_slow)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["rsi"] = rsi(out["close"], scfg.rsi_period)
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    return max(5 * scfg.ema_trend, scfg.ema_slow, scfg.rsi_period, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = {**DEFAULTS, **scfg.params}
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    tol = float(p["dip_atr_tol"]) * a
    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    use_rsi = bool(int(p["use_rsi"]))

    up = last["ema_fast"] > last["ema_slow"] and last["close"] > last["ema_trend"]
    dn = last["ema_fast"] < last["ema_slow"] and last["close"] < last["ema_trend"]

    if (
        up
        and prev["close"] > prev["ema_fast"]
        and last["low"] <= last["ema_fast"] + tol
        and last["close"] > last["ema_fast"]
        and (not use_rsi or last["rsi"] < scfg.rsi_overbought)
    ):
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, a,
            f"dip-buy: low touched EMA{scfg.ema_fast} in uptrend, RSI {last['rsi']:.0f}",
        )
    if (
        dn
        and prev["close"] < prev["ema_fast"]
        and last["high"] >= last["ema_fast"] - tol
        and last["close"] < last["ema_fast"]
        and (not use_rsi or last["rsi"] > scfg.rsi_oversold)
    ):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, a,
            f"dip-sell: high touched EMA{scfg.ema_fast} in downtrend, RSI {last['rsi']:.0f}",
        )
    return flat("no trend-dip")
