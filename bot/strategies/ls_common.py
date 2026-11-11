"""Shared helpers for the ls_ strategy library (session / volatility / pattern
families). Not a strategy module itself (no add_indicators/evaluate).

Convention used by every ls_ module: add_indicators() computes a causal
signal column "ls_sig" (+1 long, -1 short, 0 none) decided on each bar's
CLOSE, plus "atr". evaluate() only reads the last row, applies the `side`
filter and returns ATR stops (scfg.atr_sl_mult / atr_tp_mult x ATR), so the
factory's random baseline gets identical stop geometry.

Time logic is in SERVER time. A bar's index is its OPEN time; it closes at
index + timeframe. "A bar closing at HH:MM" means the next bar opens (and the
backtester fills) at HH:MM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, flat


def params(scfg: StrategyCfg, defaults: dict) -> dict:
    return {**defaults, **scfg.params}


def tf_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 60)


def hm(s: str) -> int:
    h, m = (int(x) for x in str(s).split(":"))
    return h * 60 + m


def close_mod(index: pd.DatetimeIndex, step: int) -> np.ndarray:
    """Minute-of-day of each bar's CLOSE time (0..1439)."""
    mins = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    return (mins + step) % 1440


def open_mod(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() * 60 + index.minute.to_numpy()


def day_id(index: pd.DatetimeIndex, anchor_min: int = 0) -> np.ndarray:
    """Integer trading-day id; a day starts at server time `anchor_min`."""
    t = index - pd.Timedelta(minutes=anchor_min)
    return (t.normalize().asi8 // 86_400_000_000_000).astype(np.int64)


def prev_day_stats(df: pd.DataFrame, did: np.ndarray) -> pd.DataFrame:
    """Per bar: COMPLETED previous day's high/low/open/close/range plus the
    previous days' range history (for NR-N), and today's first open.
    Only finished days are used (shift by one day group)."""
    g = pd.DataFrame({"d": did, "o": df["open"].to_numpy(), "h": df["high"].to_numpy(),
                      "l": df["low"].to_numpy(), "c": df["close"].to_numpy()})
    daily = g.groupby("d").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
                               n=("o", "size"))
    daily["rng"] = daily["h"] - daily["l"]
    prev = daily.shift(1)
    out = pd.DataFrame(index=df.index)
    m = prev.reindex(did)
    out["pd_h"] = m["h"].to_numpy()
    out["pd_l"] = m["l"].to_numpy()
    out["pd_o"] = m["o"].to_numpy()
    out["pd_c"] = m["c"].to_numpy()
    out["pd_rng"] = m["rng"].to_numpy()
    out["pd_n"] = m["n"].to_numpy()
    # NR-N: previous day's range is the smallest of the last N completed days
    for n in (4, 7):
        mn = daily["rng"].rolling(n).min()
        is_nr = (daily["rng"] <= mn + 1e-12).shift(1)
        out[f"pd_nr{n}"] = is_nr.reindex(did).to_numpy().astype(float)
    # inside day: previous day inside the day before it
    ins = ((daily["h"] <= daily["h"].shift(1)) & (daily["l"] >= daily["l"].shift(1))).shift(1)
    out["pd_inside"] = ins.reindex(did).to_numpy().astype(float)
    out["td_open"] = daily["o"].reindex(did).to_numpy()   # today's first open (known at its open)
    # median bars per day (to skip half-days / partial data)
    full = (daily["n"] >= 0.8 * daily["n"].rolling(20, min_periods=1).median()).shift(1)
    out["pd_full"] = full.reindex(did).to_numpy().astype(float)
    return out


def first_per_day(flag: np.ndarray, did: np.ndarray) -> np.ndarray:
    """Keep only the first True of each day."""
    s = pd.Series(flag.astype(int))
    cum = s.groupby(did).cumsum().to_numpy()
    return flag & (cum == 1)


def side_ok(direction: int, side: str) -> bool:
    side = str(side)
    return not ((direction == LONG and side == "short") or (direction == SHORT and side == "long"))


def signal_from_col(ind_window: pd.DataFrame, scfg: StrategyCfg, p: dict, min_bars: int,
                    tag: str) -> Signal:
    if len(ind_window) < min_bars:
        return flat("not enough bars")
    d = ind_window["ls_sig"].iat[-1]
    if not np.isfinite(d) or d == 0:
        return flat("no signal")
    direction = LONG if d > 0 else SHORT
    if not side_ok(direction, p.get("side", "both")):
        return flat("side disabled")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    c = float(ind_window["close"].iat[-1])
    return Signal(direction, c, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"{tag} {'long' if direction == LONG else 'short'}")


def exit_from_cols(ind_window: pd.DataFrame, direction: int, bars_held: int, hold: int) -> bool:
    """Generic exit: optional bar cap `hold` and per-side exit columns."""
    if hold > 0 and bars_held >= hold:
        return True
    col = "ls_xl" if direction == LONG else "ls_xs"
    if col in ind_window:
        v = ind_window[col].iat[-1]
        return bool(np.isfinite(v) and v > 0)
    return False
