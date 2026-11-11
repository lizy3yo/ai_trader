"""Shared helpers for the lx_ family (helper module, not a strategy).

Cross-market data access with STRICT causal alignment.

All data files share the broker clock and index bars by OPEN time, so a
partner bar stamped t on the same timeframe closes at the same instant as the
traded bar stamped t. Alignment is therefore:

    p = partner.reindex(own.index)          # exact-stamp match only
    p = p.ffill(limit=ffill_limit)          # carry a PAST bar forward, never back

A forward-fill can only move information from earlier stamps to later ones,
so the partner value seen on own bar i is always from a partner bar that had
closed by the close of bar i. `fresh` marks bars with an exact partner stamp
(no carried value); signals are only allowed on fresh bars.

Self-identification: a module traded by the factory only sees an OHLC frame,
not the symbol. identify() matches the frame's closes against the candidate
markets' CSVs (exact equality on a few stamps) to learn which market it is
trading, so a "pair" parameter can resolve the OTHER leg automatically.

LIVE NOTE: a live version needs the partner symbol's closed MT5 bars on the
same timeframe (the backtest reads them from data/<partner>_<tf>.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict = {}
_ID_CACHE: dict = {}

PAIRS = {
    "metals": ("gold", "silver"),
    "indices": ("us500cash", "us100cash"),
    "crypto": ("btcusd", "ethusd"),
}
ALL = ("gold", "silver", "us500cash", "us100cash", "btcusd", "ethusd", "eurusd")


def load_market(name: str, tf: str) -> pd.DataFrame | None:
    """data/<name>_<tf>.csv as an OHLC frame (cached per path+mtime)."""
    path = _ROOT / "data" / f"{name.lower()}_{str(tf).strip().lower()}.csv"
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    _CACHE[key] = df
    return df


def identify(df: pd.DataFrame, tf: str, candidates=ALL) -> str | None:
    """Which market's CSV this frame came from (exact close match)."""
    k = (len(df), df.index[0], df.index[-1], float(df["close"].iloc[-1]), str(tf), tuple(candidates))
    if k in _ID_CACHE:
        return _ID_CACHE[k]
    probe = df.index[np.linspace(0, len(df) - 1, min(15, len(df))).astype(int)]
    found = None
    for name in candidates:
        m = load_market(name, tf)
        if m is None:
            continue
        common = probe.intersection(m.index)
        if len(common) < max(3, len(probe) // 2):
            continue
        if np.allclose(m.loc[common, "close"].to_numpy(), df.loc[common, "close"].to_numpy(),
                       rtol=0, atol=1e-9):
            found = name
            break
    _ID_CACHE[k] = found
    return found


def resolve_partner(df: pd.DataFrame, tf: str, partner: str) -> str | None:
    """partner = a market name, or a PAIRS key (then: the other leg)."""
    partner = str(partner).lower()
    if partner in PAIRS:
        a, b = PAIRS[partner]
        me = identify(df, tf, (a, b))
        if me is None:
            return None
        return b if me == a else a
    me = identify(df, tf)
    return None if me == partner else partner


def aligned_close(df: pd.DataFrame, name: str, tf: str, ffill_limit: int):
    """(partner close aligned causally on df.index, fresh-stamp mask)."""
    m = load_market(name, tf)
    if m is None:
        nan = pd.Series(np.nan, index=df.index)
        return nan, pd.Series(False, index=df.index)
    exact = m["close"].reindex(df.index)
    fresh = exact.notna()
    return exact.ffill(limit=int(ffill_limit)), fresh
