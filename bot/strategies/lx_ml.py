"""Walk-forward ML direction model (family: ml) — round-5 recheck on new markets.

Predicts the sign of the next-N-bar close-to-close move with a small sklearn
model on simple causal features, and trades only high-confidence predictions.

Features (row i uses rows <= i only):
    r_k    = (close_i - close_{i-k}) / (ATR_i * sqrt(k))   k in 1,3,6,12,24,48,96
    atrp   = ATR / close                                   (volatility level)
    volr   = sd(1-bar ret, 20) / sd(1-bar ret, 100)        (vol regime)
    dist   = (close - EMA100) / ATR                        (trend stretch)
    hour, day-of-week as sin/cos
Label (training rows only): 1 if close_{i+N} > close_i else 0.

Walk-forward, strictly causal: the model for block [a, a+R) is fitted on rows
[0, a-N-1] (expanding window; labels of the last training row end before a),
then predicts the block; refit every R bars. Anchors are fixed offsets from
the frame start, so truncating the frame never changes earlier predictions.
In the factory's validation run the frame is (last 10k train bars + val), so
validation predictions come from models fitted on earlier bars only.

Horizon N = cfg.max_hold_bars (the trade is closed after N bars, so the
random baseline gets the same holding rule through the shared cfg key).

Confidence is measured against the base rate of the model's own training
labels (markets with drift have P(up) > 0.5 almost everywhere):
    edge = P(up) - base_rate(training labels of that block)
LONG if edge >= thr, SHORT if edge <= -thr.

params: model ("hgb"|"lr"), thr (edge margin), side, min_train, refit_every
SL/TP: cfg.atr_sl_mult / cfg.atr_tp_mult x ATR.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "model": "hgb",
    "thr": 0.05,
    "side": "both",
    "min_train": 4000,
    "refit_every": 2000,
}
_MEM: dict = {}
_DIR = Path(tempfile.gettempdir()) / "lx_ml_cache"


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def _horizon(scfg: StrategyCfg) -> int:
    return int(scfg.max_hold_bars) if scfg.max_hold_bars > 0 else 8


def features(df: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    c = df["close"]
    a = atr(df, atr_period).replace(0.0, np.nan)
    f = {}
    for k in (1, 3, 6, 12, 24, 48, 96):
        f[f"r{k}"] = (c - c.shift(k)) / (a * np.sqrt(k))
    f["atrp"] = 1000.0 * a / c
    r1 = np.log(c).diff()
    f["volr"] = r1.rolling(20).std(ddof=0) / r1.rolling(100).std(ddof=0).replace(0.0, np.nan)
    f["dist"] = (c - ema(c, 100)) / a
    h = df.index.hour + df.index.minute / 60.0
    f["hs"] = np.sin(2 * np.pi * h / 24.0)
    f["hc"] = np.cos(2 * np.pi * h / 24.0)
    d = df.index.dayofweek
    f["ds"] = np.sin(2 * np.pi * d / 7.0)
    f["dc"] = np.cos(2 * np.pi * d / 7.0)
    return pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)


def _model(kind: str):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def walk_forward(X: np.ndarray, y: np.ndarray, h: int, min_train: int, refit: int,
                 kind: str) -> np.ndarray:
    n = len(X)
    out = np.full(n, np.nan)
    okx = np.isfinite(X).all(axis=1)
    oky = np.isfinite(y)
    a = int(min_train)
    while a < n:
        b = min(a + refit, n)
        end = a - h - 1                      # last training row (inclusive)
        if end >= 500:
            sel = np.zeros(n, bool)
            sel[: end + 1] = True
            sel &= okx & oky
            if sel.sum() >= 500 and len(np.unique(y[sel])) == 2:
                m = _model(kind)
                m.fit(X[sel], y[sel])
                idx = np.arange(a, b)[okx[a:b]]
                if len(idx):
                    out[idx] = m.predict_proba(X[idx])[:, 1] - float(y[sel].mean())
        a = b
    return out


def _cached(df: pd.DataFrame, recipe: dict, compute) -> np.ndarray:
    fid = f"{len(df)}|{df.index[0]}|{df.index[-1]}|{df['close'].iloc[0]:.6f}|{df['close'].iloc[-1]:.6f}|{df['close'].iloc[len(df)//2]:.6f}"
    key = hashlib.sha1((fid + json.dumps(recipe, sort_keys=True)).encode()).hexdigest()
    if key in _MEM:
        return _MEM[key]
    path = _DIR / f"{key}.npy"
    if path.exists():
        try:
            arr = np.load(path)
            if len(arr) == len(df):
                _MEM[key] = arr
                return arr
        except Exception:
            pass
    arr = compute()
    _MEM[key] = arr
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    except Exception:
        pass
    return arr


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    h = _horizon(scfg)
    recipe = {"v": 2, "model": p["model"], "h": h, "min_train": int(p["min_train"]),
              "refit": int(p["refit_every"]), "atr_p": int(scfg.atr_period)}

    def compute():
        X = features(out, scfg.atr_period).to_numpy(float)
        fwd = out["close"].shift(-h) - out["close"]
        y = np.where(fwd.isna(), np.nan, (fwd > 0).astype(float))
        return walk_forward(X, y, h, int(p["min_train"]), int(p["refit_every"]), str(p["model"]))

    out["ml_edge"] = _cached(out, recipe, compute)
    return out


def warmup(scfg: StrategyCfg) -> int:
    return int(_p(scfg)["min_train"]) + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")
    pr = float(last["ml_edge"])
    if not np.isfinite(pr):
        return flat("no prediction")
    t = float(p["thr"])
    d = LONG if pr >= t else SHORT if pr <= -t else 0
    if d == 0:
        return flat("low confidence")
    if (d == LONG and p["side"] == "short") or (d == SHORT and p["side"] == "long"):
        return flat("side off")
    a = float(last["atr"])
    return Signal(d, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"ML {p['model']} edge={pr:+.3f}")
