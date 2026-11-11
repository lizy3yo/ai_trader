"""Elliott Wave "wave 5" entry (prefix ew_), rules in results/ew_rules.md.

Causal ATR ZigZag pivots L0-H1-L2-H3-L4 (long; short = mirror) obeying the
three hard rules (L2 > L0; W3 >= W1 so wave 3 cannot end up the shortest
with an equality target; L4 > H1 no overlap), wave 4 retracing r_lo..r_hi of
wave 3 -> enter when the ZigZag confirms L4 (price turned back by k x ATR).
Stop beyond wave 4's extreme. Target "struct" = wave 5 equal to wave 1 from
L4, or "rr" fixed R (= cfg.atr_tp_mult / cfg.atr_sl_mult).
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal
from . import ew_common as E

DEFAULTS = {"k": 2.0, "band": "def", "target": "struct", "rr": 0.0, "side": "both"}
BANDS = {"def": (0.236, 0.382), "wide": (0.146, 0.618)}


def _p(scfg: StrategyCfg) -> dict:
    p = {**DEFAULTS, **scfg.params}
    p["r_lo"], p["r_hi"] = BANDS[p["band"]]
    return p


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    return E.add(df, scfg, "w5", _p(scfg))


def warmup(scfg: StrategyCfg) -> int:
    return 200


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    return E.signal(ind_window, scfg, _p(scfg), "EW wave-5 after wave-4 end")
