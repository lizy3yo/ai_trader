"""Elliott Wave "wave 3" entry (prefix ew_), rules in results/ew_rules.md.

Causal ATR ZigZag pivots L0-H1-L2 (long; short = mirror): wave 2 retraces
r_lo..r_hi of wave 1 without breaking wave 1's origin -> enter at the first
close beyond wave 1's extreme (fill next open). Stop beyond wave 2's extreme.
Target "struct" = 1.618 x wave 1 projected from wave 2's end, or "rr" fixed R
(= cfg.atr_tp_mult / cfg.atr_sl_mult).
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal
from . import ew_common as E

DEFAULTS = {"k": 2.0, "band": "def", "target": "struct", "rr": 0.0, "side": "both"}
BANDS = {"def": (0.5, 0.618), "wide": (0.382, 0.786)}


def _p(scfg: StrategyCfg) -> dict:
    p = {**DEFAULTS, **scfg.params}
    p["r_lo"], p["r_hi"] = BANDS[p["band"]]
    return p


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    return E.add(df, scfg, "w3", _p(scfg))


def warmup(scfg: StrategyCfg) -> int:
    return 200


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    return E.signal(ind_window, scfg, _p(scfg), "EW wave-3 break of wave-1 extreme")
