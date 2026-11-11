"""Time-of-day entries (ls_ library, session family): one module for
overnight-vs-daytime holds, session drift windows, gap fade / gap-and-go and
Gao-style intraday momentum.

The signal fires on the bar CLOSING at entry_hm (server time), so the fill
is at entry_hm. Direction by mode:
  "long" / "short" : unconditional (drift / overnight / daytime holds)
  "follow" / "fade": sign of the move from the close at t0 (most recent bar
                     closing at t0, e.g. the prior cash close 23:00) to the
                     close at t1 (most recent bar closing at t1, e.g. 17:00 =
                     first 30 min of the US cash session, or 16:30 = the
                     overnight gap into the cash open); follow = same sign,
                     fade = opposite. |move| must be >= min_move_rng x the
                     previous completed day's high-low range.
Exits: wide ATR stop/target plus either session.flat_at in the study, or
`win` = "HH:MM-HH:MM" (entry time - exit time; the exit is an exit_signal at
the first bar close at/after the exit time, so set "exit_from_strategy": true
and the random baseline gets the same time exit). Entry times must be
multiples of the timeframe (else no trades).

params: mode, entry_hm, t0, t1, min_move_rng, win, side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import ls_common as C

DEFAULTS = {"mode": "long", "entry_hm": "16:30", "t0": "23:00", "t1": "17:00",
            "min_move_rng": 0.0, "win": "", "side": "both"}


def _times(p: dict) -> tuple[str, str]:
    if p["win"]:
        e, x = (t.strip() for t in str(p["win"]).split("-"))
        return e, x
    return str(p["entry_hm"]), ""


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = C.tf_minutes(scfg)
    cm = C.close_mod(out.index, step)
    entry_hm, exit_hm = _times(p)
    at_e = cm == C.hm(entry_hm)
    mode = str(p["mode"])
    n = len(out)
    if mode in ("long", "short"):
        sig = np.where(at_e, 1 if mode == "long" else -1, 0)
    else:
        ct = (out.index + pd.Timedelta(minutes=step)).asi8.astype(float)
        c = out["close"]
        at0 = cm == C.hm(p["t0"])
        at1 = cm == C.hm(p["t1"])
        p0 = c.where(at0).ffill().to_numpy()
        p1 = c.where(at1).ffill().to_numpy()
        T0 = pd.Series(np.where(at0, ct, np.nan)).ffill().to_numpy()
        T1 = pd.Series(np.where(at1, ct, np.nan)).ffill().to_numpy()
        day = 86_400e9
        did = C.day_id(out.index, 0)
        rng = C.prev_day_stats(out, did)["pd_rng"].to_numpy()
        with np.errstate(invalid="ignore"):
            ok = at_e & (T1 <= ct) & (ct - T1 < 0.8 * day) & (T0 < T1) & (T1 - T0 <= 4 * day)
            mv = p1 - p0
            ok &= np.abs(mv) >= float(p["min_move_rng"]) * rng
            ok &= mv != 0
        d = np.sign(mv) * (1 if mode == "follow" else -1)
        sig = np.where(ok, d, 0)
    out["ls_sig"] = np.nan_to_num(sig.astype(float)) if n else sig
    if exit_hm:
        # crossed: the latest exit time at/before this bar's close lies after the previous close
        close_t = out.index + pd.Timedelta(minutes=step)
        tx = close_t.normalize() + pd.Timedelta(minutes=C.hm(exit_hm))
        tx = tx.where(tx <= close_t, tx - pd.Timedelta(days=1))
        prev_close = pd.Series(close_t).shift(1).to_numpy()
        crossed = (tx.to_numpy() > prev_close)
        out["ls_xl"] = crossed.astype(float)
        out["ls_xs"] = out["ls_xl"]
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"TOD {p['mode']} {_times(p)}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return C.exit_from_cols(ind_window, direction, bars_held, 0)
