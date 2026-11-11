"""Single-leg PAIRS / relative value: trade one market when its log-ratio to a
related market is stretched (family: pairs).

    s  = log(own_close) - log(partner_close)          (partner aligned causally)
    z  = (s - SMA(s, lookback)) / STD(s, lookback)

mode "revert": z <= -entry_z -> LONG own (own cheap vs partner),
               z >= +entry_z -> SHORT own; exit when z crosses back through
               +/-exit_z (exit_signal), or ATR stop / wide ATR target.
mode "follow": relative momentum, the mirror: z >= entry_z -> LONG own,
               exit when z falls back below exit_z.
Entries fire on the first bar z crosses the threshold, only on bars with an
exact-stamp partner bar (no stale partner data).

pair: "metals" (GOLD<->SILVER), "indices" (US500<->US100), "crypto" (BTC<->ETH);
the module detects which leg it is trading (lx_common.identify) and uses the
other one as the partner. Markets outside the pair get no trades.

Single leg only: the backtester trades one symbol, so this keeps the traded
market's beta. LIVE: needs the partner symbol's MT5 bars on the same timeframe.

params: pair, mode, lookback, entry_z, exit_z, side, ffill_limit
SL/TP: cfg.atr_sl_mult / cfg.atr_tp_mult x ATR (TP wide; exit comes from z).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, flat, valid_atr
from . import lx_common as lc

DEFAULTS = {
    "pair": "metals",
    "mode": "revert",
    "lookback": 120,
    "entry_z": 2.0,
    "exit_z": 0.0,
    "side": "both",
    "ffill_limit": 3,
}


def _p(scfg: StrategyCfg) -> dict:
    return {**DEFAULTS, **scfg.params}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = _p(scfg)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    partner = lc.resolve_partner(out, scfg.timeframe, p["pair"])
    if partner is None:
        out["z"] = np.nan
        out["fresh"] = False
        return out
    pc, fresh = lc.aligned_close(out, partner, scfg.timeframe, int(p["ffill_limit"]))
    s = np.log(out["close"]) - np.log(pc)
    lb = int(p["lookback"])
    mu = s.rolling(lb, min_periods=lb).mean()
    sd = s.rolling(lb, min_periods=lb).std(ddof=0).replace(0.0, np.nan)
    out["z"] = (s - mu) / sd
    out["fresh"] = fresh.to_numpy()
    return out


def warmup(scfg: StrategyCfg) -> int:
    return int(_p(scfg)["lookback"]) + scfg.atr_period + 5


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    p = _p(scfg)
    last, prev = ind_window.iloc[-1], ind_window.iloc[-2]
    if not valid_atr(last) or not bool(last["fresh"]):
        return flat("no fresh partner bar")
    z, zp = float(last["z"]), float(prev["z"])
    if not (np.isfinite(z) and np.isfinite(zp)):
        return flat("z not formed")
    e = float(p["entry_z"])
    up = z >= e and zp < e
    dn = z <= -e and zp > -e
    if p["mode"] == "revert":
        d = LONG if dn else SHORT if up else 0
    else:
        d = LONG if up else SHORT if dn else 0
    if d == 0:
        return flat("no z cross")
    if (d == LONG and p["side"] == "short") or (d == SHORT and p["side"] == "long"):
        return flat("side off")
    a = float(last["atr"])
    return Signal(d, float(last["close"]), scfg.atr_sl_mult * a, scfg.atr_tp_mult * a, a,
                  f"pairs {p['mode']} z={z:+.2f}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    # Exit on the CROSS of the exit level. For real trades (opened beyond the
    # entry level) this equals "z is back at the exit level"; for random-baseline
    # entries it avoids an instant exit just because z sat on the other side.
    p = _p(scfg)
    z, zp = float(ind_window["z"].iloc[-1]), float(ind_window["z"].iloc[-2])
    if not (np.isfinite(z) and np.isfinite(zp)):
        return False
    x = float(p["exit_z"])
    if p["mode"] == "revert":
        # long was opened on z low: exit once z has come back up to -exit_z
        return (z >= -x > zp) if direction == LONG else (z <= x < zp)
    return (z <= x < zp) if direction == LONG else (z >= -x > zp)
