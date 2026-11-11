"""Gold swing Donchian (round 6, g6s_): trend_donchian + pre-registered fixes.

Baseline (all extras off) == trend_donchian: first close beyond the prior
`n`-bar high (low for shorts), close on the trade's side of EMA(cfg.ema_trend)
when trend=1, fixed ATR SL/TP.
Extras (each default OFF, all causal):
  exit_n      also exit when the close breaks the opposite `exit_n`-bar channel
              (loss cutter; the fixed TP stays)
  be_r        breakeven: after +be_r R of favourable excursion, exit on a close
              back at/below the entry price
  trail_atr   chandelier: after +trail_r R, exit on a close more than
              trail_atr x ATR below the best high since entry
  stale_bars  exit once held >= stale_bars with less than +stale_r R gained
              (swap drag on going-nowhere trades)
  vol_k       breakout bar's relative tick volume (vs same bar-of-day median of
              the prior 20 days) must be >= vol_k
  atr_q       skip entries while ATR% sits below its trailing atr_q quantile
              (one year of bars)
  ev_skip     no entries on scheduled FOMC / NFP days (news-spike breakouts)
  regime      none | down (prior day below a FALLING daily SMA252; for shorts)
              | up (mirror; for longs)
params: n, side, trend, exit_n, be_r, trail_atr, trail_r, stale_bars, stale_r,
        vol_k, atr_q, ev_skip, regime
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat
from . import g6s_common as G
from .lt_common import bars_per_day

DEFAULTS = {"n": 360, "side": "long", "trend": 1, "exit_n": 0,
            "be_r": 0.0, "trail_atr": 0.0, "trail_r": 2.0, "stale_bars": 0, "stale_r": 0.5,
            "vol_k": 0.0, "atr_q": 0.0, "ev_skip": 0, "regime": "none"}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    out["atr"] = atr(out, scfg.atr_period)
    hh, ll = donchian(out, int(p["n"]))
    up = (c.shift(1) <= hh.shift(1)) & (c > hh)
    dn = (c.shift(1) >= ll.shift(1)) & (c < ll)
    if int(p["trend"]):
        e = ema(c, scfg.ema_trend)
        up, dn = up & (c > e), dn & (c < e)
    ok = pd.Series(True, index=out.index)
    if float(p["vol_k"]) > 0:
        ok &= (G.rel_volume(out) >= float(p["vol_k"])).fillna(False)
    if float(p["atr_q"]) > 0:
        ok &= G.atr_pct_floor(out, out["atr"], float(p["atr_q"]), 250 * bars_per_day(scfg.timeframe))
    if int(p["ev_skip"]):
        ok &= ~pd.Series(G.event_day_mask(out.index), index=out.index)
    reg = str(p["regime"])
    if reg == "down":
        dt = G.daily_downtrend(out) > 0
        up, dn = up & False, dn & dt
    elif reg == "up":
        neg = pd.DataFrame({"close": -c}, index=out.index)   # mirror: above a rising SMA
        ut = G.daily_downtrend(neg) > 0
        up, dn = up & ut, dn & False
    up, dn = up & ok, dn & ok
    side = p["side"]
    sig = np.where(up & (side != "short"), 1, np.where(dn & (side != "long"), -1, 0))
    out["g_sig"] = sig.astype("int64")
    m = int(p["exit_n"])
    if m > 0:
        xh, xl = donchian(out, m)
        out["g_xl"], out["g_xs"] = c < xl, c > xh
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    w = max(int(p["n"]) + 2, int(p["exit_n"]) + 2, scfg.atr_period) + 10
    return max(w, 5 * scfg.ema_trend + 10) if int(p["trend"]) else w


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    s = int(ind_window["g_sig"].iat[-1])
    if s == 0:
        return flat("no breakout")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = LONG if s > 0 else SHORT
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a, f"{_p(scfg)['n']}-bar breakout")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = _p(scfg)
    col = "g_xl" if direction == LONG else "g_xs"
    if col in ind_window.columns and bool(ind_window[col].iat[-1]):
        return True
    return G.managed_exit(ind_window, scfg, p, direction, bars_held)
