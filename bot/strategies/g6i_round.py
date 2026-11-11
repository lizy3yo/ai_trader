"""Round-number handles: first-touch rejection vs break (g6i_).

Handles are multiples of `step` in PRICE units ($10 / $50 on gold), so this
module is market-specific by construction (gold only; there is no generic
scaling of psychological levels). Up side: H = next handle above the
previous close; a bar whose high reaches H while no bar of the prior `look`
bars touched H (first touch) is a touch.
  mode "reject": the touch bar closes back below H - buf x ATR -> short
  mode "break" : the touch bar closes above H + buf x ATR -> long
Mirror on the down side. Exits: ATR stop/target, hold_min cap
(exit_signal), session.flat_at.
params: step, look, mode, buf, hold_min, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import g6i_common as G
from . import ls_common as C

DEFAULTS = {"step": 10.0, "look": 48, "mode": "reject", "buf": 0.1, "hold_min": 120,
            "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    st, look, buf = float(p["step"]), int(p["look"]), float(p["buf"])
    h, l, c = out["high"], out["low"], out["close"]
    pc = c.shift(1)
    hu = np.ceil(pc / st) * st                      # handle above prev close
    hd = np.floor(pc / st) * st                     # handle below prev close
    ph = h.shift(1).rolling(look).max()
    pl = l.shift(1).rolling(look).min()
    tu = (h >= hu) & (ph < hu) & (pc < hu)
    td = (l <= hd) & (pl > hd) & (pc > hd)
    a = out["atr"]
    if p["mode"] == "reject":
        s_up = tu & (c < hu - buf * a)      # short
        s_dn = td & (c > hd + buf * a)      # long
        sig = np.where(s_dn & ~s_up, 1.0, np.where(s_up & ~s_dn, -1.0, 0.0))
    else:
        b_up = tu & (c > hu + buf * a)
        b_dn = td & (c < hd - buf * a)
        sig = np.where(b_up & ~b_dn, 1.0, np.where(b_dn & ~b_up, -1.0, 0.0))
    out["ls_sig"] = sig
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = C.params(scfg, DEFAULTS)
    return int(p["look"]) + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"round {p['mode']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held,
                            G.hold_bars(p["hold_min"], C.tf_minutes(scfg)))
