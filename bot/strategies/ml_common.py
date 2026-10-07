"""Shared machine-learning toolbox for the ml_ family (helper, not a strategy).

Everything here is built around ONE non-negotiable rule: a model may only ever
be fitted on bars that closed strictly BEFORE the bar it predicts, with a purge
gap of at least the label horizon in between.

    walk-forward layout (anchors are absolute row positions in the frame)

        rows 0 .............. a-h-1 | purge (h bars) | a ....... a+R-1
        ^-- training data           ^--- gap ---^    ^-- predicted block
        (labels of the last training row end at a-1, before the first
         predicted bar a, so no label can overlap a predicted bar)

    the next block refits at a+R on rows 0 .. a+R-h-1, and so on.

Because every anchor is a fixed offset from the START of the frame, truncating
the frame at any row i leaves every prediction at rows <= i bit-identical: the
same anchors, the same training rows, the same deterministic model. That is
what makes the truncation test pass (see tools-free check in ml_selftest).

Features are strictly rolling/expanding (no full-frame quantiles, means or
extremes), and forward-looking LABELS are only ever materialised for rows that
sit inside a training window.

Fitting the same walk-forward repeatedly for every SL/TP/threshold combo would
be wasted work, so the prediction vector is memoised on (frame identity, model
recipe) in RAM and on disk under the OS temp dir — the cache never influences
the numbers, it only skips recomputation.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..strategy import atr as _atr
from ..strategy import ema as _ema
from ..strategy import rsi as _rsi

# ---------------------------------------------------------------- features ---

_EPS = 1e-9

FEATURE_SETS = {
    # 10 features: momentum, one oscillator, volatility state, channel pos, clock
    "small": [
        "r1", "r3", "r12", "rsi14", "emad50", "atrp", "volr", "dcpos50",
        "hsin", "hcos",
    ],
    # 20 features: multi-lookback momentum + bar geometry + trend distances
    "mid": [
        "r1", "r2", "r3", "r6", "r12", "r24", "r48",
        "rsi14", "rsi5", "emad20", "emad50", "emad200",
        "atrp", "volr", "rng", "body", "dcpos50", "dcpos200",
        "hsin", "hcos",
    ],
    # 27 features: everything, incl. wicks, raw clock ints and day of week
    "full": [
        "r1", "r2", "r3", "r6", "r12", "r24", "r48", "r96",
        "rsi14", "rsi5", "emad20", "emad50", "emad200", "ema_slope",
        "atrp", "volr", "vol100", "rng", "body", "uw", "lw",
        "dcpos50", "dcpos200", "hsin", "hcos", "hour", "dow",
    ],
}


def build_features(df: pd.DataFrame, atr_period: int, featset: str) -> pd.DataFrame:
    """Causal feature frame. Every column uses only rows <= its own row."""
    cols = FEATURE_SETS[featset]
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    a = _atr(df, atr_period)
    a_safe = a.replace(0.0, np.nan)
    rng_raw = (high - low).replace(0.0, np.nan)
    r1_raw = close.diff()

    f = {}

    def need(name):
        return name in cols

    for k in (1, 2, 3, 6, 12, 24, 48, 96):
        n = f"r{k}"
        if need(n):
            f[n] = (close - close.shift(k)) / (a_safe * np.sqrt(k))
    if need("rsi14"):
        f["rsi14"] = (_rsi(close, 14) - 50.0) / 50.0
    if need("rsi5"):
        f["rsi5"] = (_rsi(close, 5) - 50.0) / 50.0
    for span in (20, 50, 200):
        n = f"emad{span}"
        if need(n):
            f[n] = (close - _ema(close, span)) / a_safe
    if need("ema_slope"):
        e50 = _ema(close, 50)
        f["ema_slope"] = (e50 - e50.shift(10)) / a_safe
    if need("atrp"):
        f["atrp"] = 1000.0 * a / close
    if need("volr"):
        s20 = r1_raw.rolling(20).std(ddof=0)
        s100 = r1_raw.rolling(100).std(ddof=0).replace(0.0, np.nan)
        f["volr"] = s20 / s100
    if need("vol100"):
        f["vol100"] = r1_raw.rolling(100).std(ddof=0) / a_safe
    if need("rng"):
        f["rng"] = (high - low) / a_safe
    if need("body"):
        f["body"] = (close - open_) / rng_raw
    if need("uw"):
        f["uw"] = (high - np.maximum(open_, close)) / rng_raw
    if need("lw"):
        f["lw"] = (np.minimum(open_, close) - low) / rng_raw
    for lb in (50, 200):
        n = f"dcpos{lb}"
        if need(n):
            hh = high.rolling(lb).max().shift(1)
            ll = low.rolling(lb).min().shift(1)
            f[n] = (close - ll) / (hh - ll).replace(0.0, np.nan)
    hours = df.index.hour + df.index.minute / 60.0
    if need("hsin"):
        f["hsin"] = pd.Series(np.sin(2 * np.pi * hours / 24.0), index=df.index)
    if need("hcos"):
        f["hcos"] = pd.Series(np.cos(2 * np.pi * hours / 24.0), index=df.index)
    if need("hour"):
        f["hour"] = pd.Series(df.index.hour.astype(float), index=df.index)
    if need("dow"):
        f["dow"] = pd.Series(df.index.dayofweek.astype(float), index=df.index)

    out = pd.DataFrame({c: f[c] for c in cols}, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ labels ---

def label_sign(df: pd.DataFrame, atr_s: pd.Series, horizon: int, deadband: float):
    """y=1 if the h-bar forward move exceeds +deadband*ATR, y=0 if below
    -deadband*ATR, NaN (dropped from training) inside the band.

    Rows near the frame's tail get NaN because the future is unknown — the
    walk-forward never trains on those rows, so the value there is irrelevant.
    """
    fwd = (df["close"].shift(-horizon) - df["close"]) / atr_s.replace(0.0, np.nan)
    y = np.full(len(df), np.nan)
    v = fwd.to_numpy()
    y[v > deadband] = 1.0
    y[v < -deadband] = 0.0
    return y, v


def label_triple_barrier(df: pd.DataFrame, atr_s: pd.Series, horizon: int,
                         sl_atr: float, tp_atr: float, direction=None):
    """y=1 if the trade's take-profit barrier is touched before its stop within
    the next `horizon` bars, y=0 if the stop comes first, NaN if neither is
    touched in time (or if `direction` is 0 on that bar).

    `direction` (+1/-1 per row, default all long) mirrors the barriers for
    shorts, which is what meta-labelling needs: "would THIS trade have won?".
    Barriers are measured from the current close one bar early — the real entry
    is the next bar's open — so the label is a slightly optimistic proxy for
    the trade, never a leak: it only ever reads bars strictly after t.
    """
    n = len(df)
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    a = atr_s.to_numpy(float)
    d = np.ones(n) if direction is None else np.asarray(direction, float)
    tp_px = c + d * tp_atr * a           # profit barrier for this direction
    sl_px = c - d * sl_atr * a           # loss barrier
    y = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    long_m, short_m = d > 0, d < 0
    for j in range(1, horizon + 1):
        fh = np.full(n, np.nan)
        fl = np.full(n, np.nan)
        fh[: n - j] = hi[j:]
        fl[: n - j] = lo[j:]
        # stop checked first on ambiguous bars, matching the backtester
        hit_sl = (~done) & ((long_m & (fl <= sl_px)) | (short_m & (fh >= sl_px)))
        hit_tp = (~done) & (~hit_sl) & ((long_m & (fh >= tp_px)) | (short_m & (fl <= tp_px)))
        y[hit_sl] = 0.0
        y[hit_tp] = 1.0
        done |= hit_sl | hit_tp
    y[~np.isfinite(a) | (a <= 0) | (d == 0)] = np.nan
    return y


# ------------------------------------------------------------------ models ---

def make_model(kind: str, seed: int = 0, task: str = "clf"):
    kind = str(kind).lower()
    if task == "reg":
        from sklearn.ensemble import HistGradientBoostingRegressor
        if kind in ("hgb", "gb"):
            return HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=False, random_state=seed)
        from sklearn.ensemble import RandomForestRegressor
        if kind == "rf":
            return RandomForestRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=100,
                random_state=seed, n_jobs=-1)
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    if kind == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=50, random_state=seed)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=100,
            random_state=seed, n_jobs=-1)
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs"))
    raise ValueError(f"unknown model kind {kind!r}")


# ------------------------------------------------------------ walk-forward ---

MIN_FIT_ROWS = 400


def walk_forward(X: np.ndarray, y: np.ndarray, horizon: int, refit_every: int,
                 min_train: int, max_train: int, kind: str, task: str = "clf",
                 seed: int = 0, min_fit: int = MIN_FIT_ROWS,
                 sample_mask: np.ndarray | None = None) -> np.ndarray:
    """Return a length-n vector of out-of-sample predictions (NaN where none).

    clf -> P(class 1); reg -> predicted value. Rows < min_train are always NaN.
    `sample_mask` restricts TRAINING rows (meta-labelling trains only on bars
    where the base signal fired); predictions are still emitted for every bar.
    """
    n = X.shape[0]
    out = np.full(n, np.nan)
    finite_X = np.isfinite(X).all(axis=1)
    lbl_ok = np.isfinite(y)
    if sample_mask is not None:
        lbl_ok = lbl_ok & np.asarray(sample_mask, bool)
    anchor = int(min_train)
    R = max(1, int(refit_every))
    min_fit = int(min_fit)
    while anchor < n:
        stop = min(anchor + R, n)
        train_end = anchor - int(horizon) - 1          # inclusive; the purge gap
        if train_end >= min_fit:
            lo = 0 if max_train <= 0 else max(0, train_end + 1 - int(max_train))
            sel = np.zeros(n, dtype=bool)
            sel[lo:train_end + 1] = True
            sel &= finite_X & lbl_ok
            ytr = y[sel]
            if len(ytr) >= min_fit and (task == "reg" or len(np.unique(ytr)) == 2):
                model = make_model(kind, seed=seed, task=task)
                model.fit(X[sel], ytr)
                pidx = np.arange(anchor, stop)
                pm = finite_X[anchor:stop]
                if pm.any():
                    Xp = X[pidx[pm]]
                    if task == "reg":
                        out[pidx[pm]] = model.predict(Xp)
                    else:
                        out[pidx[pm]] = model.predict_proba(Xp)[:, 1]
        anchor = stop
    return out


# ------------------------------------------------------------------- cache ---

def _cache_dir() -> Path:
    d = os.environ.get("ML_GOLD_CACHE") or str(Path(tempfile.gettempdir()) / "ai_trading_ml_cache")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


_MEM: dict[str, np.ndarray] = {}


def frame_id(df: pd.DataFrame) -> str:
    return "|".join([
        str(len(df)), str(df.index[0]), str(df.index[-1]),
        f"{float(df['close'].iloc[0]):.4f}", f"{float(df['close'].iloc[-1]):.4f}",
        f"{float(df['close'].iloc[len(df) // 2]):.4f}",
    ])


def cached_predictions(df: pd.DataFrame, recipe: dict, compute) -> np.ndarray:
    key = hashlib.sha1(
        (frame_id(df) + "::" + json.dumps(recipe, sort_keys=True)).encode()
    ).hexdigest()
    hit = _MEM.get(key)
    if hit is not None and len(hit) == len(df):
        return hit
    path = _cache_dir() / f"{key}.npy"
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
        np.save(path, arr)
    except Exception:
        pass
    return arr
