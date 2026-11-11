"""trend_donchian gated by the US DOLLAR's trend (mx_ family, idea 1a).

Story: gold is priced in dollars and tends to move opposite the dollar, so
gold breakouts should follow through better while the dollar is in a
downtrend. Base rule = trend_donchian / g6s_donch baseline: first close above
the prior `n`-bar high, close above EMA(cfg.ema_trend), fixed ATR SL/TP,
long only (side "short" mirrors; gates then flip to "dollar up").
gate (USDX D1, COMPLETED server days only, causal by close time):
  none        baseline (identical to trend_donchian long_only)
  sma50/100   last USDX daily close below its 50/100-day SMA
  roc20       USDX down over the last 20 days
  *_inv       diagnostic: the opposite regime (dollar strong) — should be worse
params: n, side, gate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, sma
from . import mx_common as M

DEFAULTS = {"n": 360, "side": "long", "gate": "none", "feat": "usdx-dec26"}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _weak_dollar(p: dict) -> pd.Series:
    """True on USDX D1 bars where the dollar is in the 'weak' regime."""
    g = str(p["gate"]).replace("_inv", "")
    c = M.load(p["feat"], "D1")["close"]
    if g.startswith("sma"):
        return c < sma(c, int(g[3:]))
    if g.startswith("roc"):
        return c < c.shift(int(g[3:]))
    raise ValueError(f"mx_dxgate: unknown gate {p['gate']!r}")


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    c = out["close"]
    out["atr"] = atr(out, scfg.atr_period)
    hh, ll = donchian(out, int(p["n"]))
    e = ema(c, scfg.ema_trend)
    up = (c.shift(1) <= hh.shift(1)) & (c > hh) & (c > e)
    dn = (c.shift(1) >= ll.shift(1)) & (c < ll) & (c < e)
    if p["gate"] != "none":
        weak = _weak_dollar(p).astype(float)
        seen, _ = M.align(out.index, str(scfg.timeframe), weak, "D1", M.stale_for("D1"))
        if str(p["gate"]).endswith("_inv"):
            seen = 1.0 - seen
        ok_l = seen.fillna(0.0) > 0.5                 # weak dollar  -> gold longs
        ok_s = (1.0 - seen).fillna(0.0) > 0.5         # strong dollar -> gold shorts
        up, dn = up & ok_l, dn & ok_s
    out["mx_long"] = up.fillna(False) & (p["side"] in ("long", "both"))
    out["mx_short"] = dn.fillna(False) & (p["side"] in ("short", "both"))
    return out


def warmup(scfg: StrategyCfg) -> int:
    return max(5 * scfg.ema_trend, int(_p(scfg)["n"]) + 2, scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    lg, sh = bool(ind_window["mx_long"].iat[-1]), bool(ind_window["mx_short"].iat[-1])
    if not (lg or sh):
        return flat("no gated breakout")
    a = float(ind_window["atr"].iat[-1])
    if not np.isfinite(a) or a <= 0:
        return flat("bad ATR")
    d = LONG if lg else SHORT
    return Signal(d, float(ind_window["close"].iat[-1]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"donchian {_p(scfg)['n']} gate {_p(scfg)['gate']}")
