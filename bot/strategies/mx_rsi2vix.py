"""US500 RSI2 dip-buyer (lm_meanrev, config_us500_rsi2.yaml params) gated by
the VIX regime (mx_ family, idea 2b — OBSERVATION ONLY: the RSI2 rule's exam
window 2023-09 -> 2026-11 is already used, so no clean exam is possible).

Story: dips bought in a fearful market (high VIX) rebound harder; in calm
markets dips are smaller and slower. Entries are lm_meanrev's, ANDed with:
  vgate none     baseline (the confirmed rule)
        hi50/lo50  VIX level 126-day percentile above / at-or-below 50
        hi80       percentile above 80
        up/dn      VIX (roll-adjusted) above / below its 20-day SMA
VIX D1, completed days, causal by close time (mx_common.align).
Exit = lm_meanrev's exit_signal (close > SMA(exit_ma), RSI > exit_rsi, max_hold).
params: lm_meanrev params + vgate
"""
from __future__ import annotations

import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal
from . import lm_meanrev as LM
from . import mx_common as M
from .mx_vix import vix_frame

DEFAULTS = {"vgate": "none"}


def _vg(scfg: StrategyCfg) -> str:
    return str({**DEFAULTS, **scfg.params}["vgate"])


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    out = LM.add_indicators(df, scfg)
    g = _vg(scfg)
    if g == "none":
        return out
    f = vix_frame()
    if g in ("hi50", "lo50", "hi80"):
        s = f["pct"]
        cond = {"hi50": s > 50, "lo50": s <= 50, "hi80": s > 80}[g] & s.notna()
    elif g in ("up", "dn"):
        cond = f["above20"] > 0.5 if g == "up" else f["above20"] < 0.5
    else:
        raise ValueError(f"mx_rsi2vix: unknown vgate {g!r}")
    seen, _ = M.align(out.index, str(scfg.timeframe), cond.astype(float), "D1", M.stale_for("D1"))
    ok = (seen > 0.5).fillna(False)
    out["mr_long"] &= ok
    out["mr_short"] &= ok
    return out


def warmup(scfg: StrategyCfg) -> int:
    return LM.warmup(scfg)


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    return LM.evaluate(ind_window, scfg)


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    return LM.exit_signal(ind_window, scfg, direction, bars_held)
