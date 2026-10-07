"""Intraday EMA-stack pullback day-trading (killzone family, prefix kz_).

Mechanized "trend day" entry: the M5 EMA stack must be aligned
(ema_f > ema_m > ema_s for longs, mirrored for shorts), price must have
PULLED BACK to the fast or mid EMA within the last pull_lb bars (touch
measured on prior bars only — the trigger bar itself is excluded), and the
trigger bar closes back on the trend side of the fast EMA.

The SESSION WINDOW is intentionally NOT handled here: restrict it with the
harness key "session.window" (e.g. "09:00-12:00" London, "15:00-18:00"
NY overlap) so the window itself is a first-class tunable.

Params (scfg.params):
    ema_f / ema_m / ema_s : EMA stack periods (default 20/50/200)
    pull_to    : "fast" (touch ema_f) or "mid" (deeper touch of ema_m)
    pull_lb    : touch must have happened within the last pull_lb CLOSED
                 bars before the trigger bar
    trigger    : "recross" = prev close on the wrong side of ema_f, last
                 close back on the trend side (one clean re-cross);
                 "any" = last close on the trend side is enough
                 (cooldown_bars then spaces repeated entries)
    confirm_hh : if true, additionally require the trigger close to take out
                 the previous bar's extreme (close > prev high for longs)
    slope_lb   : if > 0, require ema_s to be rising (longs) / falling
                 (shorts) versus slope_lb bars ago
    sides      : "both" / "long" / "short"
    min_atr    : skip entries when ATR (in $) is below this floor

SL/TP: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared config knobs).
Causality: the pullback-touch flags are rolling(pull_lb).max().shift(1) —
only strictly-prior bars; EMAs/ATR are standard causal recursions.
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "ema_f": 20,
    "ema_m": 50,
    "ema_s": 200,
    "pull_to": "fast",
    "pull_lb": 6,
    "trigger": "recross",
    "confirm_hh": False,
    "slope_lb": 0,
    "sides": "both",
    "min_atr": 0.0,
}


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["ema_f"] = ema(out["close"], int(p["ema_f"]))
    out["ema_m"] = ema(out["close"], int(p["ema_m"]))
    out["ema_s"] = ema(out["close"], int(p["ema_s"]))
    out["atr"] = atr(out, scfg.atr_period)

    tgt = out["ema_f"] if str(p["pull_to"]) == "fast" else out["ema_m"]
    lb = int(p["pull_lb"])
    # touch flags over the PRIOR lb bars only (shift(1) excludes the trigger bar)
    out["touch_lo"] = (out["low"] <= tgt).astype(float).rolling(lb).max().shift(1)
    out["touch_hi"] = (out["high"] >= tgt).astype(float).rolling(lb).max().shift(1)
    out["ema_s_lag"] = out["ema_s"].shift(max(int(p["slope_lb"]), 1))
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    # 5x the slowest EMA span leaves negligible weight on the ewm seed
    return 5 * int(p["ema_s"]) + int(p["pull_lb"]) + int(p["slope_lb"]) \
        + 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    prev = ind_window.iloc[-2]
    if not valid_atr(last):
        return flat("bad ATR")

    a = float(last["atr"])
    if a < float(p["min_atr"]):
        return flat("ATR below floor")

    ef, em, es = float(last["ema_f"]), float(last["ema_m"]), float(last["ema_s"])
    stack_up = ef > em > es
    stack_dn = ef < em < es
    if not (stack_up or stack_dn):
        return flat("EMA stack not aligned")

    c, pc = float(last["close"]), float(prev["close"])
    pef = float(prev["ema_f"])
    recross = str(p["trigger"]) == "recross"
    slope_lb = int(p["slope_lb"])
    lag = float(last["ema_s_lag"])

    direction = 0
    if stack_up and float(last["touch_lo"]) >= 1.0 and c > ef:
        if (not recross or pc <= pef) \
                and (not p["confirm_hh"] or c > float(prev["high"])) \
                and (slope_lb <= 0 or es > lag):
            direction = LONG
    elif stack_dn and float(last["touch_hi"]) >= 1.0 and c < ef:
        if (not recross or pc >= pef) \
                and (not p["confirm_hh"] or c < float(prev["low"])) \
                and (slope_lb <= 0 or es < lag):
            direction = SHORT
    if direction == 0:
        return flat("no pullback trigger")

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    sl_dist = scfg.atr_sl_mult * a
    tp_dist = scfg.atr_tp_mult * a
    side = "long" if direction == LONG else "short"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"KZ trend {side}: stack {p['ema_f']}/{p['ema_m']}/{p['ema_s']}, "
                  f"pullback->{p['pull_to']} reclaim, ATR {a:.2f}")
