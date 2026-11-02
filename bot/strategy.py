"""Shared strategy toolbox: Signal contract and common indicators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LONG, SHORT, FLAT = 1, -1, 0


@dataclass
class Signal:
    direction: int          # LONG / SHORT / FLAT
    entry_hint: float       # last closed bar close (live fills use tick price)
    sl_distance: float      # price distance to stop-loss (always > 0)
    tp_distance: float      # price distance to take-profit (always > 0)
    atr: float
    reason: str


def flat(reason: str = "no signal") -> Signal:
    return Signal(FLAT, 0.0, 0.0, 0.0, 0.0, reason)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in window = RSI 100 (max overbought, not neutral)
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std(ddof=0)


def donchian(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """(highest high, lowest low) of the PRIOR `period` bars (shifted by 1,
    so the current bar's own extreme never triggers its own breakout)."""
    hh = df["high"].rolling(period).max().shift(1)
    ll = df["low"].rolling(period).min().shift(1)
    return hh, ll


def valid_atr(last_row) -> bool:
    return bool(np.isfinite(last_row["atr"])) and float(last_row["atr"]) > 0
