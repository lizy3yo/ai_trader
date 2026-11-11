"""Elliott Wave A-B-C zigzag correction entry (prefix ew_), rules in results/ew_rules.md.

Causal ATR ZigZag pivots Lpp-Hp-Lpre-H0-La-Hb-Lc (long; short = mirror):
established uptrend (H0 > Hp, Lpre > Lpp), A = H0-La, B retraces 38.2-78.6%
of A, C ends beyond A's end but above Lpp, C/A in r_lo..r_hi -> once C is
confirmed, enter at the first close beyond B. Stop beyond C. Target fixed R
(= cfg.atr_tp_mult / cfg.atr_sl_mult; 2R default, 3R alternative).
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal
from . import ew_common as E

DEFAULTS = {"k": 2.0, "band": "def", "target": "rr", "rr": 0.0, "side": "both"}
BANDS = {"def": (0.85, 1.15), "wide": (0.618, 1.618)}


def _p(scfg: StrategyCfg) -> dict:
    p = {**DEFAULTS, **scfg.params}
    p["r_lo"], p["r_hi"] = BANDS[p["band"]]
    p["b_lo"], p["b_hi"] = 0.382, 0.786
    return p


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    return E.add(df, scfg, "abc", _p(scfg))


def warmup(scfg: StrategyCfg) -> int:
    return 200


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    return E.signal(ind_window, scfg, _p(scfg), "EW A-B-C end, break of B")
