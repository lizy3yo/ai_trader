"""Reopen / weekend gap: fade vs gap-and-go (g6i_).

Reopen bar = first bar after a pause > 50 min (gold: daily 23:59-01:00 break;
the weekend). gap = reopen OPEN - last close before the pause. The first
minutes after the reopen are untradeable (spread), so the decision is taken
at the close of the bar ending `wait_min` after the reopen and filled then.
Conditions: |gap| >= gmin x daily ATR (14 completed days' mean high-low) and
the gap still open at the decision close (price on the gap side of the old
close). mode "fade" trades toward the old close, "follow" away from it.
dows "mon" = weekend gaps only (reopen on Monday after > 24 h), "all".
Exit: time exit at x_hm (exit_signal; "exit_from_strategy": true) and
session.flat_at, plus ATR stop/target.
params: mode, gmin, wait_min, dows, x_hm, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import g6i_common as G
from . import ls_common as C

DEFAULTS = {"mode": "fade", "gmin": 0.1, "wait_min": 30, "dows": "mon", "x_hm": "10:00",
            "side": "both"}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = C.tf_minutes(scfg)
    idx = out.index
    dt = idx.to_series().diff().dt.total_seconds().div(60).to_numpy()
    reopen = dt > 50
    if p["dows"] == "mon":
        reopen &= (dt > 24 * 60) & (idx.weekday == 0)
    c = out["close"].to_numpy()
    o = out["open"].to_numpy()
    prev_c = np.r_[np.nan, c[:-1]]
    gap = np.where(reopen, o - prev_c, np.nan)
    old_c = np.where(reopen, prev_c, np.nan)
    t_re = np.where(reopen, idx.asi8, np.iinfo(np.int64).min)
    s = pd.DataFrame({"gap": gap, "old": old_c}).ffill()   # carry the last reopen forward
    t_last = np.maximum.accumulate(t_re)
    close_t = (idx + pd.Timedelta(minutes=step)).asi8
    at_dec = (close_t - t_last) == int(p["wait_min"]) * 60_000_000_000
    # daily ATR from completed days
    did = C.day_id(idx, 0)
    g = pd.DataFrame({"d": did, "h": out["high"].to_numpy(), "l": out["low"].to_numpy()})
    daily = g.groupby("d").agg(h=("h", "max"), l=("l", "min"))
    datr = (daily["h"] - daily["l"]).rolling(14).mean().shift(1).reindex(did).to_numpy()
    gp, old = s["gap"].to_numpy(), s["old"].to_numpy()
    with np.errstate(invalid="ignore"):
        ok = at_dec & (np.abs(gp) >= float(p["gmin"]) * datr) & (gp != 0)
        ok &= (c - old) * np.sign(gp) > 0
    d = np.sign(gp) * (1 if p["mode"] == "follow" else -1)
    out["ls_sig"] = np.nan_to_num(np.where(ok, d, 0.0))
    G.time_exit_cols(out, step, str(p["x_hm"]))
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 15 * 1440 // C.tf_minutes(scfg) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"gap {p['mode']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_from_cols(ind_window, direction, bars_held, 0)
