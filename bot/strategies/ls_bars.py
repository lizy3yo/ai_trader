"""Mechanical bar patterns (ls_ library, price-pattern / volatility family).

mode "inside": bar j is an inside bar (high <= mother high, low >= mother low,
               mother = bar j-1). For the next `win` bars the first close
               beyond the mother's high / low trades the breakout.
mode "rev"   : reversal candle at an N-bar extreme. Long when the bar or the
               bar before it made a new low vs the prior n bars and the bar is
                 pattern "engulf": bullish engulfing (bear bar then a bull bar
                                   whose body covers it)
                 pattern "hammer": lower wick >= 60% of range, close in the
                                   top third
                 pattern "either"
               Short is the mirror at an N-bar high.
mode "burst" : k consecutive closes in one direction whose total move is
               >= m_atr x ATR; fires once when the run reaches k bars.
               fade=0 follows the burst, fade=1 fades it.
Exits: ATR stop/target from StrategyCfg, optional cfg.max_hold_bars.

params: mode, win, n, pattern, k, m_atr, fade, side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import ls_common as C

DEFAULTS = {"mode": "inside", "win": 3, "n": 20, "pattern": "either", "k": 3, "m_atr": 1.5,
            "fade": 0, "side": "both"}


def _prev(a: np.ndarray, k: int = 1) -> np.ndarray:
    return np.r_[np.full(k, np.nan), a[:-k]]


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    o, h = out["open"].to_numpy(), out["high"].to_numpy()
    l, c = out["low"].to_numpy(), out["close"].to_numpy()
    a = out["atr"].to_numpy()
    n_bars = len(out)
    idx = np.arange(n_bars, dtype=float)
    mode = str(p["mode"])
    with np.errstate(invalid="ignore"):
        if mode == "inside":
            ins = (h <= _prev(h)) & (l >= _prev(l))
            mh = pd.Series(np.where(ins, _prev(h), np.nan)).ffill().to_numpy()
            ml = pd.Series(np.where(ins, _prev(l), np.nan)).ffill().to_numpy()
            j = pd.Series(np.where(ins, idx, np.nan)).ffill().to_numpy()
            age = idx - j
            armed = (age >= 1) & (age <= int(p["win"]))
            pc = _prev(c)
            # the level pair must be the same one the previous close was judged against
            up = armed & (c > mh) & ~(pc > mh)
            dn = armed & (c < ml) & ~(pc < ml)
        elif mode == "rev":
            n = int(p["n"])
            lo_ref = pd.Series(l).rolling(n).min().shift(2).to_numpy()
            hi_ref = pd.Series(h).rolling(n).max().shift(2).to_numpy()
            new_lo = np.minimum(l, _prev(l)) < lo_ref
            new_hi = np.maximum(h, _prev(h)) > hi_ref
            po, pcl = _prev(o), _prev(c)
            rng = h - l
            bull_eng = (pcl < po) & (c > o) & (c >= po) & (o <= pcl)
            bear_eng = (pcl > po) & (c < o) & (c <= po) & (o >= pcl)
            hammer = ((np.minimum(o, c) - l) >= 0.6 * rng) & (c >= h - rng / 3) & (rng > 0)
            star = ((h - np.maximum(o, c)) >= 0.6 * rng) & (c <= l + rng / 3) & (rng > 0)
            pat = str(p["pattern"])
            bull = bull_eng if pat == "engulf" else hammer if pat == "hammer" else (bull_eng | hammer)
            bear = bear_eng if pat == "engulf" else star if pat == "hammer" else (bear_eng | star)
            up = new_lo & bull
            dn = new_hi & bear
        else:  # burst
            k = int(p["k"])
            s = pd.Series(np.sign(c - _prev(c)))
            grp = (s != s.shift()).cumsum()
            run = s.groupby(grp).cumcount().to_numpy() + 1
            mv = c - _prev(c, k)
            big = np.abs(mv) >= float(p["m_atr"]) * a
            hit = (run == k) & big
            up = hit & (s.to_numpy() > 0)
            dn = hit & (s.to_numpy() < 0)
            if int(p["fade"]):
                up, dn = dn, up
    up = np.nan_to_num(up).astype(bool)
    dn = np.nan_to_num(dn).astype(bool)
    out["ls_sig"] = np.where(up & ~dn, 1.0, np.where(dn & ~up, -1.0, 0.0))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = C.params(scfg, DEFAULTS)
    return max(5 * scfg.atr_period, int(p["n"]) + 3, int(p["k"]) + 2) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"BARS {p['mode']}")
