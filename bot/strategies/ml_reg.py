"""Walk-forward ML regressor: predict the SIZE of the next-N-bar move.

Same walk-forward machinery as ml_clf (refit every `refit_every` bars, purge
gap of `horizon` bars, deterministic models), but the target is the continuous
ATR-normalised forward return instead of its sign. The threshold is therefore
expressed in ATRs of expected move: trade only when the model expects the next
N bars to travel at least `thr_atr` ATRs in one direction. That directly
encodes "only trade when the expected move is large enough to pay the spread".

params:
    model        "hgb" | "rf" | "lr" (ridge)
    featset      "small" | "mid" | "full"
    horizon      forecast horizon N, in bars
    refit_every / min_train / max_train / seed   walk-forward controls
    thr_atr      LONG if prediction >= +thr_atr, SHORT if <= -thr_atr
    sides        "both" | "long" | "short"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr
from . import ml_common as mc

DEFAULTS = {
    "model": "hgb",
    "featset": "mid",
    "horizon": 16,
    "refit_every": 2000,
    "min_train": 6000,
    "max_train": 0,
    "thr_atr": 0.15,
    "sides": "both",
    "seed": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    a = atr(out, scfg.atr_period)
    out["atr"] = a
    horizon = int(p["horizon"])
    featset = str(p["featset"])

    recipe = {
        "fam": "ml_reg", "model": str(p["model"]), "featset": featset,
        "horizon": horizon, "refit": int(p["refit_every"]),
        "min_train": int(p["min_train"]), "max_train": int(p["max_train"]),
        "seed": int(p["seed"]), "atr_period": int(scfg.atr_period),
    }

    def compute():
        X = mc.build_features(out, scfg.atr_period, featset).to_numpy(float)
        _y, fwd = mc.label_sign(out, a, horizon, 0.0)   # fwd = ATR-normalised move
        return mc.walk_forward(
            X, np.asarray(fwd, float), horizon=horizon,
            refit_every=int(p["refit_every"]), min_train=int(p["min_train"]),
            max_train=int(p["max_train"]), kind=str(p["model"]), task="reg",
            seed=int(p["seed"]))

    out["ml_yhat"] = mc.cached_predictions(out, recipe, compute)
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = _p(scfg)
    return int(p["min_train"]) + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = _p(scfg)
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    yhat = float(last["ml_yhat"])
    if not np.isfinite(yhat):
        return flat("no model prediction yet")

    thr = float(p["thr_atr"])
    if yhat >= thr:
        direction = LONG
    elif yhat <= -thr:
        direction = SHORT
    else:
        return flat(f"expected move {yhat:+.3f} ATR too small")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    a = float(last["atr"])
    side = "long" if direction == LONG else "short"
    return Signal(direction, float(last["close"]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a,
                  f"ML-reg {side}: E[{int(p['horizon'])}-bar move] = {yhat:+.3f} ATR")
