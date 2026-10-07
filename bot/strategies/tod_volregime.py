"""tod_volregime -- one strategy that flips personality with the vol regime.

Classify the tape by where the current ATR sits inside the distribution of
the PRIOR `vol_win` ATR values, then run a different engine in each regime:

  HIGH vol  (atr >= q_hi quantile of the prior window)  -> TREND: Donchian
            breakout of the prior `bo_lb` bars.
  LOW  vol  (atr <= q_lo quantile of the prior window)  -> MEAN REVERSION:
            fade a z-score extreme against SMA(mr_lb).
  MIDDLE                                                -> stand aside.

`mapping="inverted"` swaps the two engines (MR in high vol, trend in low
vol) and exists purely as a control: if the inverted map scores as well as
the normal one, the regime split is doing nothing.

Causality: the quantile thresholds are built from atr.shift(1).rolling(win)
so the current bar's own ATR is never inside its own reference distribution
(same convention as donchian() and hyb_common.atr_pct_ok).  The Donchian
channel is the shift-safe one from bot.strategy.  SMA/std are trailing
rolling windows.  No full-frame statistic anywhere.

Params (scfg.params)
  vol_win   : window of prior bars for the ATR quantiles
  q_hi/q_lo : quantiles that define the high / low regimes (0..1)
  bo_lb     : Donchian lookback for the trend engine
  mr_lb     : SMA/std lookback for the mean-reversion engine
  mr_z      : |z| needed to fade
  mapping   : "normal" | "inverted"
  enable    : "both" | "trend" | "mr"   (isolate one engine)
  sides     : "both" | "long" | "short"
  mr_sl_mult / mr_tp_mult : ATR multiples for the MR leg (0 = use scfg)

SL/TP for the trend leg: scfg.atr_sl_mult / atr_tp_mult x ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import (LONG, SHORT, Signal, atr, donchian, flat, rolling_std,
                        sma, valid_atr)

DEFAULTS = {
    "vol_win": 300,
    "q_hi": 0.7,
    "q_lo": 0.3,
    "bo_lb": 40,
    "mr_lb": 30,
    "mr_z": 2.0,
    "mapping": "normal",
    "enable": "both",
    "sides": "both",
    "mr_sl_mult": 0.0,
    "mr_tp_mult": 0.0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)

    win = int(p["vol_win"])
    prior = out["atr"].shift(1)                      # exclude the current bar
    out["q_hi"] = prior.rolling(win).quantile(float(p["q_hi"]))
    out["q_lo"] = prior.rolling(win).quantile(float(p["q_lo"]))

    hh, ll = donchian(out, int(p["bo_lb"]))          # already shift(1)-ed
    out["dc_hi"], out["dc_lo"] = hh, ll

    m = sma(out["close"], int(p["mr_lb"]))
    s = rolling_std(out["close"], int(p["mr_lb"]))
    out["z"] = (out["close"] - m) / s.replace(0.0, np.nan)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return max(int(p["vol_win"]) + scfg.atr_period, int(p["bo_lb"]),
               int(p["mr_lb"]), 5 * scfg.atr_period) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    qh, ql = float(last["q_hi"]), float(last["q_lo"])
    if not (np.isfinite(qh) and np.isfinite(ql)):
        return flat("vol regime not established")
    if a >= qh:
        regime = "high"
    elif a <= ql:
        regime = "low"
    else:
        return flat("middle vol regime -- stand aside")

    inverted = str(p["mapping"]).lower() == "inverted"
    engine = "trend" if (regime == "high") != inverted else "mr"
    want = str(p["enable"]).lower()
    if want != "both" and engine != want:
        return flat(f"{engine} engine disabled")

    c, cp = float(last["close"]), float(prev["close"])
    if engine == "trend":
        hi, lo = float(last["dc_hi"]), float(last["dc_lo"])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            return flat("no donchian channel")
        sl_dist, tp_dist = scfg.atr_sl_mult * a, scfg.atr_tp_mult * a
        if cp <= hi < c:
            direction, why = LONG, f"breakout > {hi:.2f}"
        elif cp >= lo > c:
            direction, why = SHORT, f"breakdown < {lo:.2f}"
        else:
            return flat("no breakout")
    else:
        z = float(last["z"])
        if not np.isfinite(z):
            return flat("no z-score")
        thr = float(p["mr_z"])
        slm = float(p["mr_sl_mult"]) or scfg.atr_sl_mult
        tpm = float(p["mr_tp_mult"]) or scfg.atr_tp_mult
        sl_dist, tp_dist = slm * a, tpm * a
        if z <= -thr:
            direction, why = LONG, f"z {z:+.2f} oversold"
        elif z >= thr:
            direction, why = SHORT, f"z {z:+.2f} overbought"
        else:
            return flat("no z extreme")

    sides = str(p["sides"]).lower()
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    side = "long" if direction == LONG else "short"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"{regime}-vol {engine} {side}: {why}")
