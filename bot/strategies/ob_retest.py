"""ICT-style order block retest (SMC family, prefix ob_).

Mechanized definition (no discretion):
- DISPLACEMENT up at bar d: close[d] - close[d - disp_bars] >= disp_k x ATR[d]
  AND close[d] breaks structure (closes above the prior struct_lb-bar high,
  shift-safe). Mirror for displacement down.
- The ORDER BLOCK is the last opposite-direction candle strictly BEFORE the
  displacement bar (age-capped so it sits at the leg's origin). Its range
  (wick-to-wick, or body-only via zone_def) becomes the zone, but only if the
  zone is narrower than width_cap_atr x ATR and price actually closed beyond it.
- The zone exists only AFTER the displacement bar closes; the trade is the
  FIRST retest of the zone in the displacement's direction (see ob_common for
  touch/confirm/one-touch/expiry semantics).
- mode "breaker": only zones that FAILED (close through the far edge) and
  flipped direction may signal — the classic breaker block.

Params (scfg.params), stops:
  stop_mode "atr":  SL/TP = scfg.atr_sl_mult / atr_tp_mult x ATR
  stop_mode "zone": SL just beyond the far zone edge (+ zone_pad_atr x ATR,
                    floored at min_sl_atr x ATR), TP = rr x SL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr, ema
from .ob_common import run_zone_machine, zone_evaluate

DEFAULTS = {
    "disp_k": 2.0,          # displacement size in ATRs
    "disp_bars": 3,         # measured over this many bars
    "struct_lb": 12,        # structure break: close beyond prior N-bar extreme
    "ob_age_max": 3,        # OB candle must be within disp_bars + this of the leg end
    "zone_def": "wick",     # "wick" = candle low..high, "body" = open..close body
    "zone_expiry": 36,      # bars the zone stays tradable
    "width_cap_atr": 1.5,   # skip zones wider than this x ATR
    "one_touch": True,      # first touch consumes the zone
    "entry_mode": "touch",  # "touch" | "confirm"
    "mode": "ob",           # "ob" | "breaker"
    "stop_mode": "atr",     # "atr" | "zone"
    "rr": 2.0,
    "zone_pad_atr": 0.15,
    "min_sl_atr": 0.6,
    "sides": "both",        # "both" | "long" | "short"
    "trend_filter": False,  # require close on the EMA(ema_trend) side
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    a = out["atr"].to_numpy(float)
    n = len(out)

    disp_bars = int(p["disp_bars"])
    struct_lb = int(p["struct_lb"])
    k = float(p["disp_k"])
    age_max = disp_bars + int(p["ob_age_max"])
    cap = float(p["width_cap_atr"])

    # shift-safe structure extremes and the N-bar-back close (all past-only)
    hh = out["high"].rolling(struct_lb).max().shift(1).to_numpy(float)
    ll = out["low"].rolling(struct_lb).min().shift(1).to_numpy(float)
    c_back = out["close"].shift(disp_bars).to_numpy(float)

    with np.errstate(invalid="ignore"):
        disp_up = (c - c_back >= k * a) & (c > hh)
        disp_dn = (c_back - c >= k * a) & (c < ll)

    # index of the last bearish / bullish candle as of the PREVIOUS bar
    # (ffill = causal; shift(1) excludes the displacement bar itself)
    idx = np.arange(n, dtype=float)
    last_bear = pd.Series(np.where(c < o, idx, np.nan)).ffill().shift(1).to_numpy()
    last_bull = pd.Series(np.where(c > o, idx, np.nan)).ffill().shift(1).to_numpy()

    if str(p["zone_def"]) == "body":
        z_lo_src = np.minimum(o, c)
        z_hi_src = np.maximum(o, c)
    else:
        z_lo_src, z_hi_src = l, h

    bull_new_lo = np.full(n, np.nan)
    bull_new_hi = np.full(n, np.nan)
    bear_new_lo = np.full(n, np.nan)
    bear_new_hi = np.full(n, np.nan)

    for i in np.nonzero(disp_up)[0]:
        j = last_bear[i]
        if not np.isfinite(j) or i - j > age_max:
            continue
        j = int(j)
        zlo, zhi = z_lo_src[j], z_hi_src[j]
        w = zhi - zlo
        if w <= 0 or not np.isfinite(a[i]) or w > cap * a[i]:
            continue
        if c[i] <= zhi:          # price must have LEFT the zone upward
            continue
        bull_new_lo[i] = zlo
        bull_new_hi[i] = zhi

    for i in np.nonzero(disp_dn)[0]:
        j = last_bull[i]
        if not np.isfinite(j) or i - j > age_max:
            continue
        j = int(j)
        zlo, zhi = z_lo_src[j], z_hi_src[j]
        w = zhi - zlo
        if w <= 0 or not np.isfinite(a[i]) or w > cap * a[i]:
            continue
        if c[i] >= zlo:          # price must have LEFT the zone downward
            continue
        bear_new_lo[i] = zlo
        bear_new_hi[i] = zhi

    long_sig, short_sig, sig_lo, sig_hi = run_zone_machine(
        l, h, c, bull_new_lo, bull_new_hi, bear_new_lo, bear_new_hi,
        expiry=int(p["zone_expiry"]), one_touch=bool(p["one_touch"]),
        entry_mode=str(p["entry_mode"]), breaker=(str(p["mode"]) == "breaker"),
    )
    out["long_sig"] = long_sig.astype(float)
    out["short_sig"] = short_sig.astype(float)
    out["sig_zlo"] = sig_lo
    out["sig_zhi"] = sig_hi
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    base = max(
        int(p["struct_lb"]) + int(p["disp_bars"]),
        int(p["zone_expiry"]) + int(p["disp_bars"]),
        5 * scfg.atr_period,
    )
    if p["trend_filter"]:
        base = max(base, 5 * scfg.ema_trend)
    return base + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    tag = "BRK-OB" if str(p["mode"]) == "breaker" else "OB"
    return zone_evaluate(ind_window, scfg, p, warmup(scfg), tag)
