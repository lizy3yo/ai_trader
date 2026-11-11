"""Shared helpers for the mx_ (macro / cross-asset) family (helper, not a strategy).

FEATURE markets (signals only, never traded): data/<name>_<tf>.csv
  usdx-dec26  US Dollar Index futures, back-adjusted, D1 from 2012-11
              (H1/H4 are daily filler bars before 2016-07: use D1 or 2016-07+)
  vix-nov26   VIX futures from 2020-12; NOT back-adjusted before 2023-02:
              the roll to the next contract shows up as a +3..+21% open gap on
              the expiry Wednesday (contango). Returns are roll-adjusted here
              (open gap dropped on expiry day and the day after); LEVELS stay
              raw (= the real front-month price at the time).
  usdjpy      D1 back to 1971, H1 real from 2010
  oilcash     WTI cash, from 2010

CAUSAL ALIGNMENT (by bar CLOSE time, not stamp): every file is stamped by bar
OPEN time on the same broker clock. A feature bar stamped s on timeframe F is
known at s + dur(F); the traded bar stamped t on timeframe T is decided at
t + dur(T). The traded bar sees the LAST feature bar with s + dur(F) <=
t + dur(T) (searchsorted, i.e. forward-fill only). A D1 feature therefore
reaches an intraday bar only once that server day has fully closed.
`max_age` drops values older than that (e.g. a feature market closed for a
holiday) instead of carrying them forever.
`new` marks traded bars on which a NEWER feature bar arrived than on the
previous traded bar (event signals fire once per feature bar).

LIVE NOTE: a live version needs the feature symbol's closed MT5 bars.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict = {}
DUR = {"M15": pd.Timedelta(minutes=15), "M30": pd.Timedelta(minutes=30), "H1": pd.Timedelta(hours=1),
       "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}
BPD = {"H1": 24, "H4": 6, "D1": 1}


def load(name: str, tf: str) -> pd.DataFrame:
    """Feature OHLC frame (cached per path + mtime)."""
    path = _ROOT / "data" / f"{name.lower()}_{tf.lower()}.csv"
    key = (str(path), path.stat().st_mtime_ns)
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


def vix_expiries(start: int = 2020, end: int = 2027) -> pd.DatetimeIndex:
    """VIX futures final settlement: the Wednesday 30 days before the third
    Friday of the following month (holiday shifts ignored: the adjustment also
    covers the next day)."""
    out = []
    for y in range(start, end + 1):
        for m in range(1, 13):
            ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
            first = pd.Timestamp(ny, nm, 1)
            third_fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
            out.append(third_fri - pd.Timedelta(days=30))
    return pd.DatetimeIndex(out)


VIX_ROLL_END = pd.Timestamp("2023-03-01")   # data check: no roll gaps on expiry days after 2023-02


def vix_roll_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Expiry day + next day, only while the series still showed roll gaps."""
    ex = vix_expiries()
    n = index.normalize()
    return (n.isin(ex) | n.isin(ex + pd.Timedelta(days=1))) & (index < VIX_ROLL_END)


def log_ret(name: str, tf: str) -> pd.Series:
    """One-bar log return of a feature; VIX roll gaps removed (D1 only)."""
    key = ("ret", name, tf)
    if key in _CACHE:
        return _CACHE[key]
    d = load(name, tf)
    r = np.log(d["close"] / d["close"].shift(1))
    if name.lower().startswith("vix"):
        if tf.upper() != "D1":
            raise ValueError("mx: VIX features are built from D1 only (roll handling)")
        r = r.where(~vix_roll_days(d.index), np.log(d["close"] / d["open"]))
    _CACHE[key] = r
    return r


def adj_close(name: str, tf: str) -> pd.Series:
    """Price rebuilt from (roll-adjusted) returns, ending at the raw last close."""
    r = log_ret(name, tf).fillna(0.0)
    c = np.exp(r.cumsum())
    return c * (load(name, tf)["close"].iloc[-1] / c.iloc[-1])


def zmove(name: str, tf: str, n_bars: int, vol_bars: int) -> pd.Series:
    """n-bar log move in units of its own 1-bar vol (prior vol_bars) * sqrt(n)."""
    r = log_ret(name, tf)
    mv = r.rolling(n_bars).sum()
    sd = r.rolling(vol_bars, min_periods=vol_bars // 2).std()
    return mv / (sd * np.sqrt(n_bars))


def pct_rank(s: pd.Series, win: int) -> pd.Series:
    """Percentile (0..100) of the current value within the last `win` values."""
    return s.rolling(win, min_periods=win // 2).apply(lambda a: (a[:-1] < a[-1]).mean() * 100.0
                                                      if len(a) > 1 else np.nan, raw=True)


def align(own_index: pd.DatetimeIndex, own_tf: str, feat: pd.Series, feat_tf: str,
          max_age: pd.Timedelta | None = None) -> tuple[pd.Series, pd.Series]:
    """(feature value seen by each traded bar, `new` mask). Causal by close time."""
    avail = feat.index + DUR[feat_tf.upper()]
    dec = own_index + DUR[own_tf.upper()]
    pos = avail.searchsorted(dec, side="right") - 1
    ok = pos >= 0
    vals = np.full(len(own_index), np.nan)
    vals[ok] = feat.to_numpy()[pos[ok]]
    if max_age is not None:
        age = np.full(len(own_index), np.inf)
        age[ok] = (dec[ok] - avail[pos[ok]]).total_seconds()
        vals[age > max_age.total_seconds()] = np.nan
    p = pd.Series(np.where(ok, pos, -1), index=own_index)
    new = (p != p.shift(1)) & ok
    return pd.Series(vals, index=own_index), new


def stale_for(feat_tf: str) -> pd.Timedelta:
    """Default max age: 4 days for D1 (long weekends), 3 days intraday."""
    return pd.Timedelta(days=4) if feat_tf.upper() == "D1" else pd.Timedelta(days=3)
