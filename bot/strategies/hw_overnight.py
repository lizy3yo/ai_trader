"""hw_overnight — late-NY long drift harvester with high-win-rate geometry.

Research finding behind this module (train split only): gold shows a
persistent positive LONG drift in the late New York hours (server clock
~19:00-23:00), stable across both halves of the train split and strong
enough (~$0.4-1.0 per 6-12 M30 bars) to clear the ~$0.55 spread — but only
when the SL is wide enough that the spread deficit (spread/SL) stops eating
the edge. Classic touch/pullback entries (EMA dip, Bollinger mid/outer band)
tested at eff < 1 everywhere and are NOT used here.

Mechanism: during the session window (set session.window to the late-NY
hours — the window IS the entry signal), go LONG on any eligible bar with
TP < SL in ATR terms (e.g. TP 2.0 / SL 5.0 -> ~74% train win rate). The
single-position rule + cooldown_bars throttle re-entries.

Long entry:
  - regime_filter=1: close > EMA(trend)   (only harvest drift in an up regime)
  - min_atr: skip bars with ATR below this many $ (dead tape, spread too big
    relative to the barriers)

Params (scfg.params):
  regime_filter : 1/0 require close > EMA(ema_trend)
  min_atr       : float, minimum ATR in $ to take a trade (0 = off)
Shared knobs used from scfg: ema_trend, atr_period, atr_sl_mult, atr_tp_mult.
The trading WINDOW comes from session config (session.window grid key).
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "regime_filter": 1,
    "min_atr": 0.0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = df.copy()
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    return out


def warmup(scfg: StrategyCfg) -> int:
    return max(5 * scfg.ema_trend, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = {**DEFAULTS, **scfg.params}
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    if float(p["min_atr"]) > 0 and a < float(p["min_atr"]):
        return flat("ATR below floor")
    if int(p["regime_filter"]) and not last["close"] > last["ema_trend"]:
        return flat("regime down")

    return Signal(
        LONG, float(last["close"]),
        scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
        f"late-session long drift (regime {'on' if int(p['regime_filter']) else 'off'})",
    )
