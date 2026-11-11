"""Shared, causal market-structure engine for the ms_ family (not a strategy).

Port of the structure logic of LuxAlgo "Smart Money Concepts" (TradingView,
CC BY-NC-SA 4.0), see results/ms_rules.md for the sources:

  leg(size): at bar t, if high[t-size] > highest(high, size) over bars
  t-size+1..t, a BEARISH leg starts and bar t-size becomes a pivot HIGH;
  if low[t-size] < lowest(low, size) a BULLISH leg starts and bar t-size is a
  pivot LOW. A pivot is therefore KNOWN only at the close of bar t = p+size.

  Structure break: close crosses above the current (uncrossed) pivot high ->
  bullish BOS, or CHoCH if the trend was bearish. Mirror for lows. Each pivot
  can be broken once. Internal layer (size 5) optionally ignores a break of a
  level equal to the swing layer's level (LuxAlgo `extraCondition`).

  Order block on a bullish break at bar t (LuxAlgo storeOrderBlock): among the
  bars [pivot bar, t-1] pick the one with the lowest "parsed low" (bars with
  range >= 2 x ATR(200) are high-volatility bars whose high/low are swapped,
  which effectively removes them from the search); the OB is that bar's
  high/low. Mirror for bearish breaks.

  Trailing extremes (premium/discount): top resets to each new pivot high and
  then tracks max(high); bottom resets to each pivot low and tracks min(low).
  Equilibrium = (top + bottom) / 2.

Everything is a single forward pass: values at row t use rows <= t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import atr

TF_MIN = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
# higher timeframe used for "bias" / odds enhancers (pre-registered pairing)
HTF_OF = {"M5": "H1", "M15": "H4", "M30": "H4", "H1": "D1", "H4": "D1", "D1": "W1"}


def structure(o, h, l, c, size: int, atr_long, other_hi=None, other_lo=None):
    """Run the LuxAlgo leg/structure machine. Returns a dict of numpy arrays:
    trend        +1 / -1 / 0 after bar t
    evt          +1 bull BOS, +2 bull CHoCH, -1 bear BOS, -2 bear CHoCH (at t)
    ob_lo/ob_hi  order block of the break at t (NaN otherwise)
    leg_ext      extreme of the broken leg: min low over [pivot, t] for bull
                 breaks, max high for bear breaks (the "swing" to stop beyond)
    sweep        1 if that leg extreme took out the previous opposite pivot
                 (the last pivot low before the broken pivot high, for bulls)
    piv_hi/piv_lo  current pivot levels after bar t (crossed or not)
    last_hi/last_lo  most recent confirmed pivot levels (for HH/HL checks)
    prev_lo/prev_hi  the confirmed pivot before last_lo / last_hi
    new_lo/new_hi  1 at the bar where a pivot low/high gets confirmed
    top/bot      trailing extremes (dealing range)
    brk_up_bar / brk_dn_bar  bar index of the latest bullish / bearish break
    brk_from     bar index of the pivot broken at t (-1 otherwise)
    lo_bar / hi_bar  bar index of last_lo / last_hi pivots
    """
    n = len(c)
    size = int(size)
    hmax = pd.Series(h).rolling(size).max().to_numpy()   # max h[t-size+1..t]
    lmin = pd.Series(l).rolling(size).min().to_numpy()
    hv = (h - l) >= 2.0 * atr_long
    hv = np.where(np.isfinite(atr_long), hv, False)
    p_hi = np.where(hv, l, h)        # parsed highs / lows (LuxAlgo)
    p_lo = np.where(hv, h, l)

    f = lambda: np.full(n, np.nan)   # noqa: E731
    trend = np.zeros(n, dtype=np.int8)
    evt = np.zeros(n, dtype=np.int8)
    ob_lo, ob_hi, leg_ext, sweep = f(), f(), f(), np.zeros(n, dtype=np.int8)
    piv_hi, piv_lo, last_hi, last_lo, prev_hi, prev_lo = f(), f(), f(), f(), f(), f()
    top, bot = f(), f()
    new_lo = np.zeros(n, dtype=np.int8)
    new_hi = np.zeros(n, dtype=np.int8)
    brk_up_bar = np.full(n, -1, dtype=np.int64)
    brk_dn_bar = np.full(n, -1, dtype=np.int64)
    lo_bar = np.full(n, -1, dtype=np.int64)
    hi_bar = np.full(n, -1, dtype=np.int64)
    brk_from = np.full(n, -1, dtype=np.int64)

    leg = 0                       # LuxAlgo: var leg = 0 (BEARISH_LEG)
    ph_lvl = pl_lvl = np.nan
    ph_bar = pl_bar = -1
    ph_x = pl_x = True            # crossed flags
    lows = []                     # recent confirmed pivot lows (bar, level)
    highs = []
    tr = 0
    t_top = t_bot = np.nan
    b_up = b_dn = -1

    for t in range(n):
        # ---- leg / pivots (getCurrentStructure) ----
        if t >= size:
            p = t - size
            nl = leg
            if h[p] > hmax[t]:
                nl = 0
            elif l[p] < lmin[t]:
                nl = 1
            if nl != leg:
                if nl == 1:                       # bullish leg -> pivot LOW at p
                    pl_lvl, pl_bar, pl_x = l[p], p, False
                    lows.append((p, l[p]))
                    if len(lows) > 8:
                        lows.pop(0)
                    new_lo[t] = 1
                    t_bot = l[p]
                else:                             # bearish leg -> pivot HIGH at p
                    ph_lvl, ph_bar, ph_x = h[p], p, False
                    highs.append((p, h[p]))
                    if len(highs) > 8:
                        highs.pop(0)
                    new_hi[t] = 1
                    t_top = h[p]
                leg = nl
        # trailing extremes every bar
        if np.isfinite(t_top):
            t_top = max(t_top, h[t])
        if np.isfinite(t_bot):
            t_bot = min(t_bot, l[t])

        # ---- structure breaks (displayStructure) ----
        if not ph_x and np.isfinite(ph_lvl) and c[t] > ph_lvl and (
                other_hi is None or not (other_hi[t] == ph_lvl)):
            evt[t] = 2 if tr == -1 else 1
            ph_x = True
            tr = 1
            b_up = t
            s, e = ph_bar, t                      # OB search window [pivot, t-1]
            brk_from[t] = s
            if e > s:
                j = s + int(np.argmin(p_lo[s:e]))
                ob_hi[t], ob_lo[t] = p_hi[j], p_lo[j]
            ext = float(np.min(l[s:t + 1]))
            leg_ext[t] = ext
            prior = [lv for (b, lv) in lows if b < s]
            if prior and ext < prior[-1]:
                sweep[t] = 1
        if not pl_x and np.isfinite(pl_lvl) and c[t] < pl_lvl and (
                other_lo is None or not (other_lo[t] == pl_lvl)):
            evt[t] = -2 if tr == 1 else -1
            pl_x = True
            tr = -1
            b_dn = t
            s, e = pl_bar, t
            brk_from[t] = s
            if e > s:
                j = s + int(np.argmax(p_hi[s:e]))
                ob_hi[t], ob_lo[t] = p_hi[j], p_lo[j]
            ext = float(np.max(h[s:t + 1]))
            leg_ext[t] = ext
            prior = [lv for (b, lv) in highs if b < s]
            if prior and ext > prior[-1]:
                sweep[t] = 1

        trend[t] = tr
        piv_hi[t], piv_lo[t] = ph_lvl, pl_lvl
        if lows:
            last_lo[t], lo_bar[t] = lows[-1][1], lows[-1][0]
            if len(lows) > 1:
                prev_lo[t] = lows[-2][1]
        if highs:
            last_hi[t], hi_bar[t] = highs[-1][1], highs[-1][0]
            if len(highs) > 1:
                prev_hi[t] = highs[-2][1]
        top[t], bot[t] = t_top, t_bot
        brk_up_bar[t], brk_dn_bar[t] = b_up, b_dn

    return dict(trend=trend, evt=evt, ob_lo=ob_lo, ob_hi=ob_hi, leg_ext=leg_ext,
                sweep=sweep, piv_hi=piv_hi, piv_lo=piv_lo, last_hi=last_hi,
                last_lo=last_lo, prev_hi=prev_hi, prev_lo=prev_lo, new_lo=new_lo,
                new_hi=new_hi, top=top, bot=bot, brk_up_bar=brk_up_bar,
                brk_dn_bar=brk_dn_bar, lo_bar=lo_bar, hi_bar=hi_bar,
                brk_from=brk_from)


def arrays(df: pd.DataFrame):
    return (df["open"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float), df["close"].to_numpy(float))


def htf_key(index: pd.DatetimeIndex, htf: str) -> pd.DatetimeIndex:
    """Start time of the higher-timeframe period each bar belongs to."""
    if htf == "W1":
        d = index.normalize()
        return d - pd.to_timedelta(d.dayofweek, unit="D")
    if htf == "D1":
        return index.normalize()
    return index.floor(f"{TF_MIN[htf]}min")


def htf_bars(df: pd.DataFrame, htf: str):
    """(htf OHLC frame indexed by period start, key per base bar)."""
    key = htf_key(df.index, htf)
    g = df.groupby(key)
    hb = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                       "low": g["low"].min(), "close": g["close"].last()})
    return hb, key


def map_prior(hb_feat: pd.DataFrame, key: pd.DatetimeIndex, index) -> pd.DataFrame:
    """Give each base bar the features of the last COMPLETED htf period
    (shift(1) on the htf series, then broadcast by period key)."""
    return hb_feat.shift(1).reindex(key).set_axis(index)


def htf_structure(df: pd.DataFrame, htf: str, size: int) -> pd.DataFrame:
    """HTF trend and dealing range from completed HTF bars only."""
    hb, key = htf_bars(df, htf)
    a = atr(hb, 200).to_numpy(float) if len(hb) else np.array([])
    o, h, l, c = arrays(hb)
    s = structure(o, h, l, c, size, a)
    feat = pd.DataFrame({"h_trend": s["trend"].astype(float), "h_top": s["top"],
                         "h_bot": s["bot"]}, index=hb.index)
    return map_prior(feat, key, df.index)


def fvg_flags(o, h, l, c):
    """LuxAlgo FVG with the default auto threshold: bullish at bar t if
    low[t] > high[t-2] and close[t-1] > high[t-2] and the middle bar's
    percent body change exceeds 2 x its cumulative mean (causal)."""
    n = len(c)
    dp = np.zeros(n)
    dp[1:] = (c[:-1] - o[:-1]) / (o[:-1] * 100.0)      # bar t-1 delta, seen at t
    thr = np.cumsum(np.abs(dp)) / np.maximum(np.arange(n), 1) * 2.0
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull[2:] = (l[2:] > h[:-2]) & (c[1:-1] > h[:-2]) & (dp[2:] > thr[2:])
    bear[2:] = (h[2:] < l[:-2]) & (c[1:-1] < l[:-2]) & (-dp[2:] > thr[2:])
    return bull, bear
