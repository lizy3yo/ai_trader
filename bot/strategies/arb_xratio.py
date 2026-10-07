"""Cross-asset signal: XAUEUR as an EXOGENOUS input for trading GOLD (XAUUSD).

Family prefix: arb_.  Single-instrument only — we trade GOLD, we never place a
second leg.  XAUEUR is read from disk purely as a SIGNAL source.

WHERE THE PARTNER DATA COMES FROM
---------------------------------
add_indicators() loads  data/xaueur_<timeframe>.csv  from the project root
(the same folder layout tools/research.py uses for gold_*.csv).  The file is
looked up relative to this module (parents[2] == project root), cached in a
module-level dict keyed by (path, mtime) so a grid sweep pays the read once.
If the file is MISSING or unreadable, every partner-derived column is filled
with NaN — evaluate() then returns a FLAT signal instead of crashing.

ALIGNMENT / CAUSALITY (the whole point of this module)
------------------------------------------------------
Both files are indexed by bar OPEN time on the same broker clock and the same
timeframe, so the partner bar stamped t closes at exactly the same instant as
the gold bar stamped t.  We therefore reindex the partner frame onto the gold
index:

    p = partner.reindex(gold.index)            # exact-stamp match, NaN if absent
    p = p.ffill(limit=ffill_limit)             # tolerate a few missing stamps

reindex+ffill can only ever carry a bar from the PAST forward, never a future
bar backwards, so the partner bar used for gold bar i is always a partner bar
that closed at or before i's close.  Every derived statistic is a rolling or
shift() operation — no full-frame quantile/mean/max/zscore anywhere.  A bar
whose partner history is not clean (NaN anywhere in the rolling window, or no
exact-stamp partner bar on the signal bar itself) produces NaN and is skipped.

FEATURES
--------
    ratio = gold_close / xaueur_close          ~ EURUSD (the ONLY orthogonal
                                                 information the pair carries)
    gm = (close - close.shift(n)) / atr        gold's n-bar move, in gold ATRs
    xm = (px    - px.shift(n))    / patr       XAUEUR's n-bar move, in its ATRs
    u  = n-bar ratio return / (rolling 1-bar ratio-return sd * sqrt(n))
                                               the USD move, in sigmas
    rz = (ratio - sma(ratio, z_win)) / rolling_std(ratio, z_win)
                                               USD level vs its recent range

PARAMS (scfg.params)
--------------------
    feature   : "conf" both legs agree and both moved >= thr / x_thr
                        -> raw direction = sign(gm)   (a REAL gold move, not a
                           currency artifact)
                "div"  the legs DISAGREE in sign, |gm| >= thr, |xm| >= x_thr
                        -> raw direction = sign(gm)   (gold moved on USD alone)
                "usd"  |u| >= thr  -> raw direction = sign(u)  (dollar momentum:
                           EURUSD up = USD weak = gold bid)
                "z"    |rz| >= z_thr -> raw direction = sign(rz)
    polarity  : "follow" (trade the raw direction) or "fade" (trade against it)
    trigger   : "cross" (condition true now, false on the previous bar) or
                "state" (any bar the condition holds)
    n         : lookback bars for gm/xm/u
    thr       : gold / usd threshold (ATRs or sigmas depending on feature)
    x_thr     : XAUEUR-leg threshold in partner ATRs (conf/div only)
    z_win     : window for rz and for the ratio-return sd used by u
    z_thr     : threshold for feature "z"
    sides     : "both" / "long" / "short"
    trend_filter : 0/1 — require close on the right side of EMA(cfg.ema_trend)
    ffill_limit  : max bars a stale partner bar may be carried forward
    min_fresh    : min fraction of exact-stamp partner bars in the lookback

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x gold ATR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, rolling_std, sma, valid_atr

DEFAULTS = {
    "partner": "xaueur",
    "feature": "conf",
    "polarity": "follow",
    "trigger": "cross",
    "n": 12,
    "thr": 1.0,
    "x_thr": 1.0,
    "z_win": 96,
    "z_thr": 2.0,
    "sides": "both",
    "trend_filter": 0,
    "ffill_limit": 2,
    "min_fresh": 0.9,
}

_ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict = {}


def _load_partner(name: str, tf: str) -> pd.DataFrame | None:
    """Read data/<name>_<tf>.csv once per (path, mtime). None if unavailable."""
    path = _ROOT / "data" / f"{name.lower()}_{tf.strip().lower()}.csv"
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    if key in _CACHE:
        return _CACHE[key]
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not {"time", "open", "high", "low", "close"}.issubset(df.columns):
            return None
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")[["open", "high", "low", "close"]].astype(float)
        df = df[~df.index.duplicated(keep="first")].sort_index()
    except Exception:
        return None
    _CACHE[key] = df
    return df


def _nan_cols(out: pd.DataFrame) -> pd.DataFrame:
    for c in ("gm", "xm", "u", "rz", "fresh_frac"):
        out[c] = np.nan
    return out


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    n = max(1, int(p["n"]))
    z_win = max(5, int(p["z_win"]))

    partner = _load_partner(str(p["partner"]), scfg.timeframe)
    if partner is None or len(partner) == 0:
        # No partner data on disk -> no cross-asset signal, no crash.
        return _nan_cols(out)

    # ---- align: exact stamp only, then carry a PAST bar forward at most
    # ffill_limit bars. Never pulls a future partner bar backwards. ----
    aligned = partner.reindex(out.index)
    fresh = aligned["close"].notna().astype(float)      # exact-stamp hit?
    lim = max(0, int(p["ffill_limit"]))
    aligned = aligned.ffill(limit=lim) if lim > 0 else aligned

    px = aligned["close"]
    patr = atr(aligned, scfg.atr_period)

    # fraction of exact-stamp partner bars over the lookback (rolling, causal)
    look = max(n, z_win)
    out["fresh_frac"] = fresh.rolling(look).mean()
    # the signal bar itself must have a same-stamp partner bar
    out["fresh_frac"] = out["fresh_frac"].where(fresh > 0)

    # n-bar moves, each normalised by its OWN instrument's ATR
    out["gm"] = (out["close"] - out["close"].shift(n)) / out["atr"]
    out["xm"] = (px - px.shift(n)) / patr

    ratio = out["close"] / px                                    # ~ EURUSD
    r1 = ratio.pct_change(fill_method=None)
    sd1 = rolling_std(r1, z_win)
    out["u"] = (ratio / ratio.shift(n) - 1.0) / (sd1 * np.sqrt(n))
    out["rz"] = (ratio - sma(ratio, z_win)) / rolling_std(ratio, z_win)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    need = max(int(p["n"]), int(p["z_win"])) * 2 + 5 * scfg.atr_period
    if int(p.get("trend_filter", 0)):
        need = max(need, 5 * scfg.ema_trend)
    return need + 10


def _raw_dir(row, p) -> int:
    """Direction implied by the chosen feature on one row, or 0."""
    feat = str(p["feature"])
    thr = float(p["thr"])
    if feat == "conf":
        gm, xm, x_thr = row["gm"], row["xm"], float(p["x_thr"])
        if not (np.isfinite(gm) and np.isfinite(xm)):
            return 0
        if gm >= thr and xm >= x_thr:
            return LONG
        if gm <= -thr and xm <= -x_thr:
            return SHORT
        return 0
    if feat == "div":
        gm, xm, x_thr = row["gm"], row["xm"], float(p["x_thr"])
        if not (np.isfinite(gm) and np.isfinite(xm)):
            return 0
        if gm >= thr and xm <= -x_thr:
            return LONG
        if gm <= -thr and xm >= x_thr:
            return SHORT
        return 0
    if feat == "usd":
        u = row["u"]
        if not np.isfinite(u):
            return 0
        if u >= thr:
            return LONG
        if u <= -thr:
            return SHORT
        return 0
    if feat == "z":
        rz, z_thr = row["rz"], float(p["z_thr"])
        if not np.isfinite(rz):
            return 0
        if rz >= z_thr:
            return LONG
        if rz <= -z_thr:
            return SHORT
        return 0
    return 0


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    ff = last["fresh_frac"]
    if not np.isfinite(ff):
        return flat("no partner bar for this stamp")
    if float(ff) < float(p["min_fresh"]):
        return flat("partner history too gappy")

    d = _raw_dir(last, p)
    if d == 0:
        return flat("no cross-asset trigger")

    if str(p["trigger"]) == "cross":
        prev = ind_window.iloc[-2]
        if _raw_dir(prev, p) == d:
            return flat("condition already active on previous bar")

    if str(p["polarity"]) == "fade":
        d = -d

    sides = str(p["sides"])
    if (d == LONG and sides == "short") or (d == SHORT and sides == "long"):
        return flat("side disabled")

    c = float(last["close"])
    if int(p.get("trend_filter", 0)):
        trend = float(last["ema_trend"])
        if not np.isfinite(trend):
            return flat("no trend value")
        if d == LONG and c <= trend:
            return flat("against trend filter")
        if d == SHORT and c >= trend:
            return flat("against trend filter")

    a = float(last["atr"])
    side = "long" if d == LONG else "short"
    reason = (f"arb {p['feature']}/{p['polarity']} {side}: "
              f"gm {last['gm']:.2f} xm {last['xm']:.2f} "
              f"u {last['u']:.2f} rz {last['rz']:.2f}")
    return Signal(d, c, scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a, reason)
