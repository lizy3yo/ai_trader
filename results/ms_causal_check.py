"""Truncation (no-lookahead) test for the ms_ family.

For each strategy/param set: compute add_indicators on the full frame and on
frames truncated at several cut points; every output column must agree on the
rows both frames share (row i may only depend on rows <= i). Also prints the
signal count and the median stop size in ATRs (used to set the random
baseline's cfg.atr_sl_mult).
  python results/ms_causal_check.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bot.config import load_config  # noqa: E402
from bot.strategies import get_strategy  # noqa: E402
from tools.research import load_split  # noqa: E402

CASES = [
    ("ms_smc", {}),
    ("ms_smc", {"bias": "htf", "poi": "fvg", "tp_mode": "liq"}),
    ("ms_smc", {"sweep": 0, "pd": 0}),
    ("ms_trend", {"length": 5}),
    ("ms_trend", {"length": 2, "tp_mode": "trail"}),
    ("ms_sd", {}),
    ("ms_sd", {"enh": "all"}),
]


def main() -> None:
    base = load_config("research_gold.yaml")
    for tf in ("H1", "M15", "H4"):
        tr, va, _ = load_split("GOLD", tf, 60000)
        df = tr.iloc[-9000:]
        for name, params in CASES:
            mod = get_strategy(name)
            scfg = copy.deepcopy(base.strategy)
            scfg.name, scfg.timeframe, scfg.params = name, tf, dict(params)
            full = mod.add_indicators(df, scfg)
            bad = 0
            for cut in (3001, 5003, 7777, 8999):
                part = mod.add_indicators(df.iloc[:cut], scfg)
                for col in part.columns:
                    a = full[col].iloc[:cut].to_numpy()
                    b = part[col].to_numpy()
                    try:
                        same = np.allclose(a.astype(float), b.astype(float), equal_nan=True, rtol=0, atol=1e-9)
                    except (TypeError, ValueError):
                        same = (a == b).all()
                    if not same:
                        bad += 1
                        print(f"  MISMATCH {name} {params} {tf} cut {cut} col {col}")
            sig_cols = [c for c in full.columns if c in ("long_sig", "short_sig", "ms_long", "ms_short")]
            nsig = int(full[sig_cols].sum().sum())
            # median stop distance in ATRs
            stops = []
            w = mod.warmup(scfg)
            for i in np.nonzero(full[sig_cols].sum(axis=1).to_numpy() > 0)[0]:
                if i < w:
                    continue
                s = mod.evaluate(full.iloc[: i + 1], scfg)
                if s.direction != 0:
                    stops.append(s.sl_distance / s.atr)
            med = np.median(stops) if stops else float("nan")
            print(f"{name:<9}{tf:<4}{str(params):<55} signals {nsig:5d} trades {len(stops):5d} "
                  f"median SL {med:.2f} ATR  {'OK' if bad == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
