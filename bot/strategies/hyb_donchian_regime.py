"""Donchian breakout entry engine wrapped in REGIME GATES (hyb_ family).

Entry: close breaks above the highest high of the PRIOR `lookback` bars ->
LONG; below the lowest low -> SHORT (donchian() from bot/strategy.py is
already shift(1)-ed, so a bar never breaks its own extreme). The condition
is level-based (Turtle-style), so after an exit the strategy can re-enter
while the run continues; cooldown_bars throttles churn.

scfg.params knobs (all optional):
  lookback    -- Donchian channel length in bars (default 20)
  adx_period  -- ADX lookback (default 14)
  adx_min     -- only trade when ADX(adx_period) > adx_min. 0 = gate off.
  atr_pct_win -- rolling window (bars) for the ATR percentile gate (default 200)
  atr_pct_min -- only trade when current ATR >= this quantile (0..1) of the
                 PRIOR atr_pct_win ATR values (shift-safe). 0 = gate off.
  mode        -- "both" | "long" | "short" (directional lock)
  trend_ema   -- if > 0, longs need close > EMA(trend_ema), shorts the mirror

SL/TP come from the shared scfg.atr_sl_mult / atr_tp_mult fields.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr
from .hyb_common import adx, atr_pct_ok

DEFAULTS = {
    "lookback": 20,
    "adx_period": 14,
    "adx_min": 0.0,
    "atr_pct_win": 200,
    "atr_pct_min": 0.0,
    "mode": "both",
    "trend_ema": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    hh, ll = donchian(out, int(p["lookback"]))
    out["don_hh"] = hh
    out["don_ll"] = ll
    out["atr"] = atr(out, scfg.atr_period)
    out["adx"] = adx(out, int(p["adx_period"]))
    out["vol_ok"] = atr_pct_ok(out["atr"], int(p["atr_pct_win"]), float(p["atr_pct_min"]))
    trend_ema = int(p["trend_ema"])
    out["ema_trend"] = ema(out["close"], trend_ema) if trend_ema > 0 else 0.0
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    need_vol = int(p["atr_pct_win"]) + 1 if float(p["atr_pct_min"]) > 0 else 0
    trend_ema = int(p["trend_ema"])
    return max(
        int(p["lookback"]) + 1, scfg.atr_period, 4 * int(p["adx_period"]),
        need_vol, 5 * trend_ema if trend_ema > 0 else 0,
    ) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    if not (pd.notna(last["don_hh"]) and pd.notna(last["don_ll"])):
        return flat("no channel yet")

    adx_min = float(p["adx_min"])
    if adx_min > 0 and not float(last["adx"]) > adx_min:
        return flat(f"ADX {last['adx']:.0f} <= {adx_min:.0f}")
    if not bool(last["vol_ok"]):
        return flat("ATR below vol percentile")

    mode = str(p["mode"]).lower()
    trend_ema = int(p["trend_ema"])
    close = float(last["close"])
    sl_dist = scfg.atr_sl_mult * float(last["atr"])
    tp_dist = scfg.atr_tp_mult * float(last["atr"])
    lb = int(p["lookback"])

    long_ok = close > float(last["don_hh"]) and mode != "short"
    short_ok = close < float(last["don_ll"]) and mode != "long"
    if trend_ema > 0:
        long_ok = long_ok and close > float(last["ema_trend"])
        short_ok = short_ok and close < float(last["ema_trend"])

    if long_ok:
        return Signal(
            LONG, close, sl_dist, tp_dist, float(last["atr"]),
            f"Donchian{lb} breakout up, ADX {last['adx']:.0f}",
        )
    if short_ok:
        return Signal(
            SHORT, close, sl_dist, tp_dist, float(last["atr"]),
            f"Donchian{lb} breakout down, ADX {last['adx']:.0f}",
        )
    return flat("inside channel")
