"""Shared helpers for the hyb_* (regime filters & hybrids) strategy family.

Not a strategy module itself -- just causal indicator math imported by
hyb_ema_regime / hyb_donchian_regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ADX(period). Causal: only bar-to-bar diffs + ewm smoothing,
    so row i uses rows <= i exclusively."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)


def atr_pct_ok(atr_series: pd.Series, window: int, q: float) -> pd.Series:
    """True where the current ATR >= the q-quantile of the PRIOR `window`
    ATR values. shift(1) keeps the current bar's ATR out of its own
    reference distribution (same convention as donchian() in bot/strategy.py).
    q <= 0 disables the gate (all True). NaN threshold rows compare False,
    which just extends the effective warmup."""
    if q <= 0.0:
        return pd.Series(True, index=atr_series.index)
    thresh = atr_series.shift(1).rolling(window).quantile(q)
    return atr_series >= thresh
