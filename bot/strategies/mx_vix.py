"""Buy the S&P after a VIX SPIKE (mx_ family, idea 2a).

Story (documented in the literature): fear spikes overshoot; equity returns
after sharp VIX rises / very high VIX are above average as volatility
mean-reverts. VIX futures data start 2020-12, so studies use max_bars that
put the split start after it.
Trigger (VIX D1, completed days, causal by close time; first day only):
  j1_10 / j1_20  VIX up > 10% / 20% in 1 day     (roll-adjusted log return)
  j3_15 / j3_30  VIX up > 15% / 30% over 3 days
  p80 / p90      raw VIX level crosses above its 126-day 80th / 90th percentile
                 (crosses on pre-2023-03 roll days are dropped)
Exit (next open): vix_exit "norm" = VIX back below its 10-day SMA (roll-adj),
"time" = only the cap; always max_hold traded bars; ATR disaster stop.
params: trig, vix_exit, max_hold, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, sma
from . import mx_common as M

VIX = "vix-nov26"
DEFAULTS = {"trig": "j1_10", "vix_exit": "norm", "max_hold": 10, "side": "long"}


def vix_frame() -> pd.DataFrame:
    """VIX D1 features on the VIX index (cached)."""
    key = ("vixframe",)
    if key in M._CACHE:
        return M._CACHE[key]
    raw = M.load(VIX, "D1")["close"]
    r = M.log_ret(VIX, "D1")
    adj = M.adj_close(VIX, "D1")
    f = pd.DataFrame(index=raw.index)
    f["r1"] = r
    f["r3"] = r.rolling(3).sum()
    f["pct"] = M.pct_rank(raw, 126)
    f["above10"] = (adj > sma(adj, 10)).astype(float)
    f["above20"] = (adj > sma(adj, 20)).astype(float)
    M._CACHE[key] = f
    return f


def trigger(trig: str) -> pd.Series:
    f = vix_frame()
    kind, val = trig[:1], trig[1:]
    if kind == "j":
        k, x = val.split("_")
        hit = f[f"r{k}"] > np.log(1 + float(x) / 100.0)
    elif kind == "p":
        hit = f["pct"] > float(val)
    else:
        raise ValueError(f"mx_vix: unknown trig {trig!r}")
    cross = hit & ~hit.shift(1, fill_value=False)
    if kind == "p":   # raw-level crosses caused by a contract roll are not fear spikes
        cross &= ~M.vix_roll_days(cross.index)
    return cross


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    tf = str(scfg.timeframe)
    out["atr"] = atr(out, scfg.atr_period)
    cum = trigger(p["trig"]).astype(float).cumsum()
    seen, _ = M.align(out.index, tf, cum, "D1", M.stale_for("D1"))
    out["mx_sig"] = (seen.diff() > 0).fillna(False)
    ab, _ = M.align(out.index, tf, vix_frame()["above10"], "D1", M.stale_for("D1"))
    out["mx_norm"] = (ab < 0.5).fillna(False)
    return out


def warmup(scfg: StrategyCfg) -> int:
    return scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if not bool(ind_window["mx_sig"].iat[-1]):
        return flat("no VIX spike")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = LONG if _p(scfg)["side"] == "long" else SHORT
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"VIX {_p(scfg)['trig']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = _p(scfg)
    if bars_held >= int(p["max_hold"]):
        return True
    return p["vix_exit"] == "norm" and bool(ind_window["mx_norm"].iat[-1])
