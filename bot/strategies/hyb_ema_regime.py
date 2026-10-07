"""EMA-crossover entry engine wrapped in REGIME GATES (hyb_ family).

Entry = exactly the ema_cross logic (fast x slow crossover, close on the
right side of the trend EMA, RSI cap), but a signal is only allowed through
when the regime gates agree:

scfg.params knobs (all optional):
  adx_period  -- ADX lookback (default 14)
  adx_min     -- only trade when ADX(adx_period) > adx_min. 0 = gate off.
  atr_pct_win -- rolling window (bars) for the ATR percentile gate (default 200)
  atr_pct_min -- only trade when current ATR >= this quantile (0..1) of the
                 PRIOR atr_pct_win ATR values (shift-safe). 0 = gate off.
  mode        -- "both" | "long" | "short" (directional lock)

SL/TP come from the shared scfg.atr_sl_mult / atr_tp_mult fields.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rsi, valid_atr
from .hyb_common import adx, atr_pct_ok

DEFAULTS = {
    "adx_period": 14,
    "adx_min": 0.0,
    "atr_pct_win": 200,
    "atr_pct_min": 0.0,
    "mode": "both",
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["ema_fast"] = ema(out["close"], scfg.ema_fast)
    out["ema_slow"] = ema(out["close"], scfg.ema_slow)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["rsi"] = rsi(out["close"], scfg.rsi_period)
    out["atr"] = atr(out, scfg.atr_period)
    out["adx"] = adx(out, int(p["adx_period"]))
    out["vol_ok"] = atr_pct_ok(out["atr"], int(p["atr_pct_win"]), float(p["atr_pct_min"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need_vol = int(p["atr_pct_win"]) + 1 if float(p["atr_pct_min"]) > 0 else 0
    return max(
        5 * scfg.ema_trend, scfg.ema_slow, scfg.rsi_period, scfg.atr_period,
        4 * int(p["adx_period"]), need_vol,
    ) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    adx_min = float(p["adx_min"])
    if adx_min > 0 and not float(last["adx"]) > adx_min:
        return flat(f"ADX {last['adx']:.0f} <= {adx_min:.0f}")
    if not bool(last["vol_ok"]):
        return flat("ATR below vol percentile")

    mode = str(p["mode"]).lower()
    sl_dist = scfg.atr_sl_mult * float(last["atr"])
    tp_dist = scfg.atr_tp_mult * float(last["atr"])

    crossed_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    crossed_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    if (
        crossed_up and mode != "short"
        and last["close"] > last["ema_trend"] and last["rsi"] < scfg.rsi_overbought
    ):
        return Signal(
            LONG, float(last["close"]), sl_dist, tp_dist, float(last["atr"]),
            f"EMA{scfg.ema_fast}x{scfg.ema_slow} up, ADX {last['adx']:.0f}",
        )
    if (
        crossed_dn and mode != "long"
        and last["close"] < last["ema_trend"] and last["rsi"] > scfg.rsi_oversold
    ):
        return Signal(
            SHORT, float(last["close"]), sl_dist, tp_dist, float(last["atr"]),
            f"EMA{scfg.ema_fast}x{scfg.ema_slow} down, ADX {last['adx']:.0f}",
        )
    return flat("no crossover")
