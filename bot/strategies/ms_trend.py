"""Classic market-structure trend trading (prefix ms_), rules in results/ms_rules.md.

Swing points = LuxAlgo leg pivots of size `length` (a pivot low is known
`length` bars after it, once every later low is higher).
LONG setup (short = mirror):
  - structure is bullish: the latest break was a bullish BOS/CHoCH (close
    above the last pivot high), and that break happened AFTER the previous
    pivot low  (=> HL, break, pullback sequence)
  - a new pivot low is confirmed at this bar and it is a HIGHER LOW
    (above the previous pivot low)
  - entry at the confirmation bar's close (fill next open)
  - stop = higher low - sl_pad_atr x ATR
  - exit: tp_mode "rr" -> rr x risk; tp_mode "trail" -> close below the latest
    confirmed pivot low (structure trail via exit_signal; TP parked far away)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat
from . import ms_common as M

DEFAULTS = {"length": 5, "tp_mode": "rr", "rr": 0.0, "sl_pad_atr": 0.1,
            "min_sl_atr": 0.3, "side": "both"}


def _rr(p: dict, scfg: StrategyCfg) -> float:
    """Reward:risk. rr <= 0 -> cfg.atr_tp_mult / cfg.atr_sl_mult, so the grid sets
    RR through the shared ATR knobs and the random baseline (which uses those
    ATR knobs for its own stops) gets the same stop/target geometry."""
    r = float(p["rr"])
    return r if r > 0 else float(scfg.atr_tp_mult) / float(scfg.atr_sl_mult)


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    a200 = atr(out, 200).to_numpy(float)
    o, h, l, c = M.arrays(out)
    s = M.structure(o, h, l, c, int(p["length"]), a200)
    with np.errstate(invalid="ignore"):
        hl = (s["new_lo"] == 1) & (s["last_lo"] > s["prev_lo"])
        lh = (s["new_hi"] == 1) & (s["last_hi"] < s["prev_hi"])
    # the break must be newer than the PREVIOUS pivot low (HL -> BOS -> pullback -> new HL)
    out["ms_long"] = (hl & (s["trend"] == 1) & (s["brk_up_bar"] >= 0) &
                      (s["brk_up_bar"] > _prev_bar(s["lo_bar"], s["new_lo"]))).astype(float)
    out["ms_short"] = (lh & (s["trend"] == -1) & (s["brk_dn_bar"] >= 0) &
                       (s["brk_dn_bar"] > _prev_bar(s["hi_bar"], s["new_hi"]))).astype(float)
    out["ms_stop_l"] = s["last_lo"]
    out["ms_stop_s"] = s["last_hi"]
    out["ms_trail_l"] = s["last_lo"]     # latest confirmed pivot low
    out["ms_trail_s"] = s["last_hi"]
    if p["side"] == "long":
        out["ms_short"] = 0.0
    elif p["side"] == "short":
        out["ms_long"] = 0.0
    return out


def _prev_bar(piv_bar: np.ndarray, new_flag: np.ndarray) -> np.ndarray:
    """Bar index of the pivot BEFORE the current latest one (causal)."""
    n = len(piv_bar)
    out = np.full(n, -1, dtype=np.int64)
    prev = -1
    cur = -1
    for t in range(n):
        if new_flag[t]:
            prev, cur = cur, piv_bar[t]
        out[t] = prev
    return out


def warmup(scfg: StrategyCfg) -> int:
    return max(600, 20 * int(_p(scfg)["length"]))


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    go_l, go_s = last["ms_long"] > 0, last["ms_short"] > 0
    if not (go_l or go_s) or (go_l and go_s):
        return flat("no structure setup")
    a = float(last["atr"])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    c = float(last["close"])
    pad = float(p["sl_pad_atr"]) * a
    d = LONG if go_l else SHORT
    sl = (c - (float(last["ms_stop_l"]) - pad)) if d == LONG else ((float(last["ms_stop_s"]) + pad) - c)
    sl = max(sl, float(p["min_sl_atr"]) * a)
    tp = 50.0 * sl if p["tp_mode"] == "trail" else _rr(p, scfg) * sl
    if not (np.isfinite(sl) and sl > 0):
        return flat("bad SL")
    return Signal(d, c, sl, tp, a, f"structure {'HL' if d == LONG else 'LH'} entry")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    if _p(scfg)["tp_mode"] != "trail":
        return False
    last = ind_window.iloc[-1]
    c = float(last["close"])
    if direction == LONG:
        v = float(last["ms_trail_l"])
        return bool(np.isfinite(v) and c < v)
    v = float(last["ms_trail_s"])
    return bool(np.isfinite(v) and c > v)
