"""Intraday mean reversion after an extreme 1-2 hour move, volume-filtered (g6i_).

move = close - close `win_min` minutes ago; extreme when |move| >= m x the
hourly-equivalent ATR (ATR(TF) x sqrt(60 / TF minutes)) -> fade it.
Only the first bar of a run of extreme bars signals. Volume filter on the
mean rvol (time-of-day-normalised tick volume) of the bars in the move:
  vf "none" | "low" (mean rvol <= 1.0: a move with no participation =
  liquidity-driven, should revert) | "high" (>= 1.5: climax).
Exits: ATR stop/target, hold_min cap (exit_signal), session.flat_at.
params: win_min, m, vf, vdays, hold_min, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import g6i_common as G
from . import ls_common as C

DEFAULTS = {"win_min": 60, "m": 3.0, "vf": "none", "vdays": 20, "hold_min": 120, "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = C.tf_minutes(scfg)
    k = max(1, int(round(int(p["win_min"]) / step)))
    c = out["close"]
    mv = (c - c.shift(k)).to_numpy()
    ah = out["atr"].to_numpy() * np.sqrt(60.0 / step)
    with np.errstate(invalid="ignore"):
        ext = np.abs(mv) >= float(p["m"]) * ah
    first = ext & ~np.r_[False, ext[:-1]]
    rv = pd.Series(G.rvol(out, int(p["vdays"])), index=out.index)
    mrv = rv.rolling(k, min_periods=k).mean().to_numpy()
    vf = p["vf"]
    with np.errstate(invalid="ignore"):
        if vf == "low":
            first &= mrv <= 1.0
        elif vf == "high":
            first &= mrv >= 1.5
    out["ls_sig"] = np.nan_to_num(np.where(first, -np.sign(mv), 0.0))
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 5 * scfg.atr_period + 30


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), "xmr fade")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held,
                            G.hold_bars(p["hold_min"], C.tf_minutes(scfg)))
