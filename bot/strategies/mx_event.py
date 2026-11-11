"""Macro shock EVENT (mx_ family): a sharp move in a FEATURE market (never
traded) triggers a fixed-direction trade in the traded market.

Economic stories pre-registered for this module (round 6, mx_):
  usdx-dec26 down (dollar falls k sigma)   -> gold long        (gold = anti-dollar)
  usdjpy down (yen bid, risk-off)          -> S&P/BTC fade (long) or follow (short)
  oilcash up (oil momentum = inflation)    -> gold long

Signal, decided on the traded bar's close:
  z = feature's `days`-day log move / (its 1-bar vol over `vol_days` * sqrt(n))
  (computed on the feature's own bars: feat_tf "D1" or "same" = traded TF)
  trigger "down": z crosses below -k on a feature bar that closed since the
  previous traded bar (cross = previous feature bar was not beyond -k);
  "up" mirror. Causal alignment by close time (mx_common.align).
Exit: cfg.max_hold_bars (time) or the ATR SL/TP (cfg.atr_sl_mult/atr_tp_mult).
params: feat, feat_tf, days, k, trigger, side ("long"/"short"), vol_days
LIVE: needs the feature symbol's closed bars from MT5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat
from . import mx_common as M

DEFAULTS = {"feat": "usdx-dec26", "feat_tf": "D1", "days": 1, "k": 2.0, "trigger": "down",
            "side": "long", "vol_days": 60}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    tf = str(scfg.timeframe).upper()
    ftf = tf if p["feat_tf"] == "same" else str(p["feat_tf"]).upper()
    bpd = M.BPD[ftf]
    z = M.zmove(p["feat"], ftf, int(p["days"]) * bpd, int(p["vol_days"]) * bpd)
    k = float(p["k"])
    hit = (z <= -k) if p["trigger"] == "down" else (z >= k)
    ev = (hit & ~hit.shift(1, fill_value=False)).astype(float)
    cum = ev.cumsum()
    seen, _new = M.align(out.index, tf, cum, ftf, M.stale_for(ftf))
    fired = seen.diff() > 0                        # >= 1 event bar arrived since the last traded bar
    out["mx_sig"] = fired.fillna(False).astype(bool)
    out["mx_z"] = M.align(out.index, tf, z, ftf, M.stale_for(ftf))[0]
    return out


def warmup(scfg: StrategyCfg) -> int:
    return scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if not bool(ind_window["mx_sig"].iat[-1]):
        return flat("no macro event")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    p = _p(scfg)
    d = LONG if p["side"] == "long" else SHORT
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"{p['feat']} {p['trigger']} z={float(ind_window['mx_z'].iat[-1]):.2f}")
