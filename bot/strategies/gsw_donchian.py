"""GOLD swing: Donchian breakout with a daily long-term regime gate and
optional channel / regime exits (research round 4, prefix gsw_).

Entry (long; short is the mirror, only when enabled and the regime is bearish):
  * one lookback      -> first close above the prior `lookbacks`-bar high
                         (identical to trend_donchian)
  * several lookbacks -> ensemble: each lookback keeps a breakout state
                         (+1 after a close above its high, -1 after a close
                         below its low); score = mean of states; enter when
                         the score crosses up through `ens_thr`.
Gates:
  * use_ema=1    close above EMA(cfg.ema_trend) (the old filter)
  * regime_mode  0 none | 1 prior daily close > SMA(ma_days daily closes)
                 2 12-month time-series momentum (prior daily close >
                   close `mom_days` trading days earlier) | 3 both
    The regime is computed on DAILY closes (server date), shifted one day,
    so it is causal and independent of the bar frequency. Until it has
    formed (first ~year of data) it passes if nan_pass=1.
longs=0 + shorts=1 -> short-only research mode.
Exits (via exit_signal; set a wide cfg.atr_tp_mult to make them the exit):
  exit_mode 0 fixed TP only | 1 close beyond the prior `exit_lb`-bar
  low (trailing channel) | 2 close beyond the midpoint of the `exit_lb`-bar
  channel | 3 ensemble score <= ens_exit.  regime_exit=1 also closes when the
  daily regime flips against the position. Time stop: cfg.max_hold_bars.
`start_after` (research only): no entries before this timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "lookbacks": "360",
    "ens_thr": 1.0,
    "ens_exit": 0.0,
    "use_ema": 1,
    "regime_mode": 0,
    "ma_days": 210,
    "mom_days": 252,
    "nan_pass": 1,
    "exit_mode": 0,
    "exit_lb": 180,
    "regime_exit": 0,
    "shorts": 0,
    "longs": 1,
    "start_after": "",
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _lbs(p: dict) -> list[int]:
    return [int(x) for x in str(p["lookbacks"]).replace(";", ",").split(",") if str(x).strip()]


def daily_regime(df: pd.DataFrame, ma_days: int, mom_days: int) -> tuple[np.ndarray, np.ndarray]:
    """(+1/-1/nan) per bar for the SMA rule and the TSM rule, from the
    PREVIOUS server day's close (known before any bar of today closes)."""
    c = df["close"]
    day = c.index.normalize()
    dc = c.groupby(day).last()
    sma_sig = np.sign(dc - dc.rolling(ma_days).mean()).where(dc.rolling(ma_days).count() == ma_days)
    mom_sig = np.sign(dc / dc.shift(mom_days) - 1.0)
    sma_sig, mom_sig = sma_sig.shift(1), mom_sig.shift(1)
    return sma_sig.reindex(day).to_numpy(), mom_sig.reindex(day).to_numpy()


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    lbs = _lbs(p)
    states = []
    for n in lbs:
        hh = out["high"].rolling(n).max().shift(1)
        ll = out["low"].rolling(n).min().shift(1)
        if n == lbs[0]:
            out["dc_hi"], out["dc_lo"] = hh, ll
        st = pd.Series(np.where(c > hh, 1.0, np.where(c < ll, -1.0, np.nan)), index=out.index)
        states.append(st.ffill().fillna(0.0))
    out["score"] = sum(states) / len(states)
    m = int(p["exit_lb"])
    out["ex_hi"] = out["high"].rolling(m).max().shift(1)
    out["ex_lo"] = out["low"].rolling(m).min().shift(1)
    out["ex_mid"] = (out["ex_hi"] + out["ex_lo"]) / 2.0
    out["ema_trend"] = ema(c, scfg.ema_trend)
    out["atr"] = atr(out, scfg.atr_period)
    sma_sig, mom_sig = daily_regime(out, int(p["ma_days"]), int(p["mom_days"]))
    out["reg_sma"], out["reg_mom"] = sma_sig, mom_sig
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(5 * scfg.ema_trend if int(p["use_ema"]) else 0, max(_lbs(p)) + 2,
               int(p["exit_lb"]) + 2, scfg.atr_period) + 10


def _regime_ok(row, p: dict, d: int) -> bool:
    mode = int(p["regime_mode"])
    if mode == 0:
        return True
    vals = []
    if mode in (1, 3):
        vals.append(row["reg_sma"])
    if mode in (2, 3):
        vals.append(row["reg_mom"])
    for v in vals:
        if pd.isna(v):
            if not int(p["nan_pass"]):
                return False
            continue
        if v * d <= 0:
            return False
    return True


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if p["start_after"] and ind_window.index[-1] < pd.Timestamp(p["start_after"]):
        return flat("before start")
    if not valid_atr(last) or pd.isna(last["dc_hi"]) or pd.isna(prev["dc_hi"]):
        return flat("not formed")
    a = float(last["atr"])
    sl, tp = scfg.atr_sl_mult * a, scfg.atr_tp_mult * a
    lbs = _lbs(p)
    if len(lbs) == 1:
        up = prev["close"] <= prev["dc_hi"] and last["close"] > last["dc_hi"]
        dn = prev["close"] >= prev["dc_lo"] and last["close"] < last["dc_lo"]
    else:
        thr = float(p["ens_thr"])
        up = prev["score"] < thr <= last["score"]
        dn = prev["score"] > -thr >= last["score"]
    ema_on = int(p["use_ema"])
    if up and int(p["longs"]) and (not ema_on or last["close"] > last["ema_trend"]) and _regime_ok(last, p, LONG):
        return Signal(LONG, float(last["close"]), sl, tp, a, f"breakout up {p['lookbacks']}")
    if (dn and int(p["shorts"]) and (not ema_on or last["close"] < last["ema_trend"])
            and _regime_ok(last, p, SHORT)
            and (int(p["regime_mode"]) == 0 or _regime_strict(last, p, SHORT))):
        return Signal(SHORT, float(last["close"]), sl, tp, a, f"breakout down {p['lookbacks']}")
    return flat("no breakout")


def _regime_strict(row, p: dict, d: int) -> bool:
    """Shorts need a FORMED bearish regime (no nan pass-through)."""
    q = dict(p, nan_pass=0)
    return _regime_ok(row, q, d)


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = _p(scfg)
    last = ind_window.iloc[-1]
    mode = int(p["exit_mode"])
    c = float(last["close"])
    if mode == 1:
        lvl = last["ex_lo"] if direction == LONG else last["ex_hi"]
        if not pd.isna(lvl) and (c - float(lvl)) * direction < 0:
            return True
    elif mode == 2:
        lvl = last["ex_mid"]
        if not pd.isna(lvl) and (c - float(lvl)) * direction < 0:
            return True
    elif mode == 3:
        if float(last["score"]) * direction <= float(p["ens_exit"]):
            return True
    if int(p["regime_exit"]) and int(p["regime_mode"]):
        q = dict(p, nan_pass=1)
        if not _regime_ok(last, q, direction):
            return True
    return False
