"""Walk-forward ML classifier: predict the sign of the next-N-bar move.

This is the "trading AI" test. A small sklearn classifier is refitted every
`refit_every` bars on all history up to that point (or a trailing window),
with a purge gap of `horizon` bars between the last training label and the
first predicted bar, and emits P(up) for the block of bars that follows.
evaluate() turns that probability into LONG / SHORT / FLAT via a threshold.

No model is ever fitted on a bar it later predicts — see ml_common for the
walk-forward layout and the truncation-safety argument.

params:
    model        "hgb" | "gb" | "rf" | "lr"   (all small: depth<=4)
    featset      "small" (10) | "mid" (20) | "full" (27)
    horizon      label horizon N, in bars
    label        "sign"  -> y = sign of the N-bar ATR-normalised move
                 "tb"    -> triple barrier: does +tp_atr come before -sl_atr?
    deadband     ("sign" only) drop training rows whose |move| < deadband*ATR
    lbl_sl/lbl_tp ("tb" only) barrier distances in ATR (default = cfg SL/TP)
    refit_every  bars between refits (prediction block size)
    min_train    rows before the first fit (= first predictable bar)
    max_train    0 = expanding window, else trailing window length
    thr          LONG if P(up) >= thr, SHORT if P(up) <= 1-thr
    sides        "both" | "long" | "short"
    seed         model random_state

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR.
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
    "horizon": 8,
    "label": "sign",
    "deadband": 0.0,
    "lbl_sl": 0.0,          # 0 -> use scfg.atr_sl_mult
    "lbl_tp": 0.0,          # 0 -> use scfg.atr_tp_mult
    "refit_every": 2000,
    "min_train": 6000,
    "max_train": 0,
    "thr": 0.55,
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
    label = str(p["label"])
    sl_atr = float(p["lbl_sl"]) or float(scfg.atr_sl_mult)
    tp_atr = float(p["lbl_tp"]) or float(scfg.atr_tp_mult)

    recipe = {
        "fam": "ml_clf", "model": str(p["model"]), "featset": featset,
        "horizon": horizon, "label": label,
        "deadband": float(p["deadband"]),
        "sl": sl_atr if label == "tb" else 0.0,
        "tp": tp_atr if label == "tb" else 0.0,
        "refit": int(p["refit_every"]), "min_train": int(p["min_train"]),
        "max_train": int(p["max_train"]), "seed": int(p["seed"]),
        "atr_period": int(scfg.atr_period),
    }

    def compute():
        X = mc.build_features(out, scfg.atr_period, featset).to_numpy(float)
        if label == "tb":
            y = mc.label_triple_barrier(out, a, horizon, sl_atr, tp_atr)
        else:
            y, _ = mc.label_sign(out, a, horizon, float(p["deadband"]))
        return mc.walk_forward(
            X, y, horizon=horizon, refit_every=int(p["refit_every"]),
            min_train=int(p["min_train"]), max_train=int(p["max_train"]),
            kind=str(p["model"]), task="clf", seed=int(p["seed"]))

    out["ml_p"] = mc.cached_predictions(out, recipe, compute)
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
    prob = float(last["ml_p"])
    if not np.isfinite(prob):
        return flat("no model prediction yet")

    thr = float(p["thr"])
    if prob >= thr:
        direction = LONG
    elif prob <= 1.0 - thr:
        direction = SHORT
    else:
        return flat(f"P(up) {prob:.3f} inside the neutral band")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    a = float(last["atr"])
    side = "long" if direction == LONG else "short"
    return Signal(direction, float(last["close"]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a,
                  f"ML {p['model']} {side}: P(up)={prob:.3f} thr={thr:.2f}")
