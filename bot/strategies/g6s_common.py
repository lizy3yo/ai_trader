"""Shared helpers for the g6s_ gold swing library (round 6). Not a strategy.

All features are causal (row i uses rows <= i only):
  rel_volume      tick volume / median tick volume of the SAME bar-of-day over
                  the previous `days` days (removes the time-of-day profile and
                  the slow drift of the broker's tick count)
  atr_pct_floor   True when ATR% (ATR/close) is above its trailing `q` quantile
                  (computed on prior bars) -> skips the dead-volatility regime
  event_day_mask  bar's server date is a scheduled FOMC decision or NFP day
  daily_downtrend completed-day close below a FALLING SMA(n) (slope over `k` days)
Position-relative exits (breakeven / chandelier trail / stale trade) read the
fill bar from `bars_held` exactly like the backtester calls exit_signal:
fill bar = window.iloc[-bars_held], signal bar = window.iloc[-bars_held - 1]
(its ATR set the stop: R = cfg.atr_sl_mult * ATR_signal). They decide on a
bar's CLOSE and fill at the next open, so live can run the identical rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import LONG
from .lm_calendar_data import FOMC_DATES, NFP_DATES


def rel_volume(df: pd.DataFrame, days: int = 20) -> pd.Series:
    v = df["tick_volume"].astype(float) if "tick_volume" in df else pd.Series(np.nan, index=df.index)
    slot = df.index.hour * 60 + df.index.minute
    med = v.groupby(slot).transform(lambda s: s.shift(1).rolling(days, min_periods=max(5, days // 2)).median())
    return v / med.replace(0.0, np.nan)


def atr_pct_floor(df: pd.DataFrame, atr_s: pd.Series, q: float, window: int) -> pd.Series:
    if q <= 0:
        return pd.Series(True, index=df.index)
    ap = atr_s / df["close"]
    thr = ap.shift(1).rolling(window, min_periods=window // 2).quantile(q)
    return (ap > thr).fillna(False)


def event_day_mask(index: pd.DatetimeIndex, which: str = "fomc,nfp") -> np.ndarray:
    days = []
    if "fomc" in which:
        days += list(FOMC_DATES)
    if "nfp" in which:
        days += list(NFP_DATES)
    ev = pd.to_datetime(days).values.astype("datetime64[D]")
    return np.isin(index.normalize().values.astype("datetime64[D]"), ev)


def daily_downtrend(df: pd.DataFrame, n: int = 252, k: int = 21) -> pd.Series:
    """+1 = prior completed day closed below a falling SMA(n); else 0 (NaN-safe)."""
    d = df["close"].resample("1D").last().dropna()
    s = d.rolling(n).mean()
    dn = ((d < s) & (s < s.shift(k))).astype(float).where(s.shift(k).notna(), 0.0)
    dn = dn.shift(1)
    day = df.index.normalize()
    return pd.Series(dn.reindex(day).to_numpy(), index=df.index).fillna(0.0)


def pos_state(w: pd.DataFrame, scfg, direction: int, bars_held: int):
    """(entry_ref, R, mfe, last_close, atr_now) or None if the window is short."""
    if bars_held < 1 or bars_held + 1 > len(w):
        return None
    fill = w.iloc[-bars_held]
    a_sig = float(w["atr"].iat[-bars_held - 1])
    if not np.isfinite(a_sig) or a_sig <= 0:
        return None
    entry = float(fill["open"])
    R = scfg.atr_sl_mult * a_sig
    seg = w.iloc[-bars_held:]
    if direction == LONG:
        mfe = float(seg["high"].max()) - entry
    else:
        mfe = entry - float(seg["low"].min())
    return entry, R, mfe, float(w["close"].iat[-1]), float(w["atr"].iat[-1])


def managed_exit(w: pd.DataFrame, scfg, p: dict, direction: int, bars_held: int) -> bool:
    """Breakeven (be_r), chandelier trail (trail_atr after trail_r), stale exit."""
    be_r, tr_atr, stale = float(p.get("be_r", 0)), float(p.get("trail_atr", 0)), int(p.get("stale_bars", 0))
    if be_r <= 0 and tr_atr <= 0 and stale <= 0:
        return False
    st = pos_state(w, scfg, direction, bars_held)
    if st is None:
        return False
    entry, R, mfe, c, a_now = st
    gain = (c - entry) * direction
    if be_r > 0 and mfe >= be_r * R and gain <= 0:
        return True
    if tr_atr > 0 and mfe >= float(p.get("trail_r", 2.0)) * R:
        seg = w.iloc[-bars_held:]
        ext = float(seg["high"].max()) if direction == LONG else float(seg["low"].min())
        if (c - (ext - direction * tr_atr * a_now)) * direction < 0:
            return True
    if stale > 0 and bars_held >= stale and gain < float(p.get("stale_r", 0.5)) * R:
        return True
    return False
