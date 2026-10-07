"""Meta-labelling: a walk-forward ML gate on top of a Donchian breakout.

The base rule is the family that already works here (trend_donchian): go with
the first close that breaks the prior `lookback`-bar high/low. The ML model
never picks the direction — it only answers "would a trade taken on THIS
breakout have hit its take-profit before its stop?", learned from the outcome
of every past breakout (triple-barrier label, mirrored for shorts). Trades are
taken only when P(win) >= thr.

That is the classic López de Prado meta-labelling setup and it is the most
defensible use of ML here: the model filters an edge that exists, instead of
being asked to conjure direction out of noise.

Training rows are restricted to bars where the base signal actually fired, so
fits are small; min_fit is correspondingly low. Everything else (purge gap,
refit anchors, determinism) is the ml_common walk-forward.

params:
    lookback, use_trend, long_only   base breakout knobs
    model, featset, refit_every, min_train, max_train, min_fit, seed
    horizon      barrier horizon in bars for the win/loss label
    lbl_sl/lbl_tp barrier distances in ATR (0 -> use cfg SL/TP multipliers)
    thr          minimum P(win) to take the breakout
    sides        "both" | "long" | "short"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, donchian, ema, flat, valid_atr
from . import ml_common as mc

DEFAULTS = {
    "lookback": 48,
    "use_trend": 0,
    "long_only": 0,
    "model": "hgb",
    "featset": "mid",
    "horizon": 24,
    "lbl_sl": 0.0,
    "lbl_tp": 0.0,
    "refit_every": 2000,
    "min_train": 6000,
    "max_train": 0,
    "min_fit": 120,
    "thr": 0.5,
    "sides": "both",
    "seed": 0,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _base_direction(out: pd.DataFrame, p: dict, scfg: StrategyCfg) -> np.ndarray:
    """+1 / -1 / 0 per bar: first close outside the prior-`lookback` channel."""
    hh, ll = donchian(out, int(p["lookback"]))
    c = out["close"]
    up = (c.shift(1) <= hh.shift(1)) & (c > hh)
    dn = (c.shift(1) >= ll.shift(1)) & (c < ll)
    if int(p["use_trend"]):
        tr = ema(c, scfg.ema_trend)
        up &= c > tr
        dn &= c < tr
    if int(p["long_only"]):
        dn &= False
    d = np.zeros(len(out))
    d[up.fillna(False).to_numpy()] = 1.0
    d[dn.fillna(False).to_numpy()] = -1.0
    return d


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    a = atr(out, scfg.atr_period)
    out["atr"] = a

    d = _base_direction(out, p, scfg)
    out["base_dir"] = d

    horizon = int(p["horizon"])
    featset = str(p["featset"])
    sl_atr = float(p["lbl_sl"]) or float(scfg.atr_sl_mult)
    tp_atr = float(p["lbl_tp"]) or float(scfg.atr_tp_mult)

    recipe = {
        "fam": "ml_meta", "model": str(p["model"]), "featset": featset,
        "horizon": horizon, "sl": sl_atr, "tp": tp_atr,
        "lookback": int(p["lookback"]), "use_trend": int(p["use_trend"]),
        "long_only": int(p["long_only"]), "ema_trend": int(scfg.ema_trend),
        "refit": int(p["refit_every"]), "min_train": int(p["min_train"]),
        "max_train": int(p["max_train"]), "min_fit": int(p["min_fit"]),
        "seed": int(p["seed"]), "atr_period": int(scfg.atr_period),
    }

    def compute():
        feats = mc.build_features(out, scfg.atr_period, featset)
        X = np.column_stack([feats.to_numpy(float), d])   # direction is a feature
        y = mc.label_triple_barrier(out, a, horizon, sl_atr, tp_atr, direction=d)
        return mc.walk_forward(
            X, y, horizon=horizon, refit_every=int(p["refit_every"]),
            min_train=int(p["min_train"]), max_train=int(p["max_train"]),
            kind=str(p["model"]), task="clf", seed=int(p["seed"]),
            min_fit=int(p["min_fit"]), sample_mask=(d != 0))

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
    d = float(last["base_dir"])
    if d == 0.0:
        return flat("no channel breakout")
    prob = float(last["ml_p"])
    gate = float(p["thr"])
    if gate > 0.0:                      # thr <= 0 = gate OFF: the raw base rule,
        if not np.isfinite(prob):       # so the sweep measures the ML filter's
            return flat("no model prediction yet")   # value against its own baseline
        if prob < gate:
            return flat(f"ML gate rejected breakout: P(win)={prob:.3f}")

    direction = LONG if d > 0 else SHORT
    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    a = float(last["atr"])
    side = "long" if direction == LONG else "short"
    return Signal(direction, float(last["close"]), scfg.atr_sl_mult * a,
                  scfg.atr_tp_mult * a, a,
                  f"meta-ML {side}: {int(p['lookback'])}-bar breakout, "
                  f"P(win)={prob:.3f} >= {float(p['thr']):.2f}")
