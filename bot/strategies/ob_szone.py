"""Supply/demand zone strategy: base-candle cluster + explosive leave (prefix ob_).

Mechanized definition (no discretion):
- A BASE is a run of >= base_len consecutive small-body candles
  (|close-open| <= base_body_atr x that bar's ATR).
- An explosive LEAVE at bar i: over the last leave_bars bars price moved from
  the base to close at least leave_k x ATR BEYOND the base extreme
  (close[i] - base_high >= leave_k x ATR for demand; mirror for supply).
- The zone is the base range (min low .. max high of the last base_len base
  candles), capped at width_cap_atr x ATR. It exists only after the leave bar
  CLOSES; trade the FIRST return into it in the leave's direction
  (touch/confirm/one-touch/expiry semantics in ob_common).

Stops as in ob_retest: stop_mode "atr" uses scfg.atr_sl_mult/atr_tp_mult;
"zone" puts the SL beyond the far zone edge with TP = rr x SL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr, ema
from .ob_common import run_zone_machine, zone_evaluate

DEFAULTS = {
    "base_len": 3,          # min consecutive base candles
    "base_body_atr": 0.35,  # base candle body cap in ATRs
    "leave_k": 1.5,         # leave must close this many ATRs beyond the base edge
    "leave_bars": 2,        # bars allowed for the leave move
    "zone_expiry": 48,      # bars the zone stays tradable
    "width_cap_atr": 1.2,   # skip zones wider than this x ATR
    "one_touch": True,
    "entry_mode": "touch",  # "touch" | "confirm"
    "stop_mode": "atr",     # "atr" | "zone"
    "rr": 2.0,
    "zone_pad_atr": 0.15,
    "min_sl_atr": 0.6,
    "sides": "both",
    "trend_filter": False,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    base_len = int(p["base_len"])
    leave_bars = int(p["leave_bars"])
    k = float(p["leave_k"])
    cap = float(p["width_cap_atr"])

    body = (out["close"] - out["open"]).abs()
    is_base = body <= float(p["base_body_atr"]) * out["atr"]

    # consecutive-base run length (causal: block id via cumsum of changes,
    # cumcount = position within the block so far)
    sb = is_base.fillna(False)
    block = (sb != sb.shift()).cumsum()
    run = sb.astype(int) * (sb.groupby(block).cumcount() + 1)

    # base extent over the last base_len bars, referenced from the leave end:
    # at bar i the base ends at i - leave_bars (all values are past-only)
    base_hi = out["high"].rolling(base_len).max().shift(leave_bars)
    base_lo = out["low"].rolling(base_len).min().shift(leave_bars)
    run_at_end = run.shift(leave_bars)

    a = out["atr"]
    width_ok = (base_hi - base_lo > 0) & (base_hi - base_lo <= cap * a)
    demand = (run_at_end >= base_len) & width_ok & (out["close"] - base_hi >= k * a)
    supply = (run_at_end >= base_len) & width_ok & (base_lo - out["close"] >= k * a)

    d = demand.fillna(False).to_numpy(bool)
    s = supply.fillna(False).to_numpy(bool)
    zlo = base_lo.to_numpy(float)
    zhi = base_hi.to_numpy(float)
    n = len(out)
    bull_new_lo = np.where(d, zlo, np.nan)
    bull_new_hi = np.where(d, zhi, np.nan)
    bear_new_lo = np.where(s, zlo, np.nan)
    bear_new_hi = np.where(s, zhi, np.nan)

    long_sig, short_sig, sig_lo, sig_hi = run_zone_machine(
        out["low"].to_numpy(float), out["high"].to_numpy(float),
        out["close"].to_numpy(float),
        bull_new_lo, bull_new_hi, bear_new_lo, bear_new_hi,
        expiry=int(p["zone_expiry"]), one_touch=bool(p["one_touch"]),
        entry_mode=str(p["entry_mode"]), breaker=False,
    )
    out["long_sig"] = long_sig.astype(float)
    out["short_sig"] = short_sig.astype(float)
    out["sig_zlo"] = sig_lo
    out["sig_zhi"] = sig_hi
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    base = max(
        int(p["base_len"]) + int(p["leave_bars"]) + 5,
        int(p["zone_expiry"]) + int(p["leave_bars"]),
        5 * scfg.atr_period,
    )
    if p["trend_filter"]:
        base = max(base, 5 * scfg.ema_trend)
    return base + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    return zone_evaluate(ind_window, scfg, p, warmup(scfg), "SD")
