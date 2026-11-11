"""Truncation (no-lookahead) test + baseline-geometry measurement for the ew_ family.

1) For each model/setting: add_indicators on a frame and on copies truncated at several
   cut points; every column must agree on the shared rows (row i depends on rows <= i).
2) Median structural stop and structural target (in ATRs) per (model, k), measured on
   TRAIN data only (GOLD, BTCUSD, US500Cash x M15/H1/H4/D1), used to set cfg.atr_sl_mult /
   cfg.atr_tp_mult of the random baseline in the grids.
  python results/ew_causal_check.py
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
from tools.factory import MARKETS  # noqa: E402
from tools.research import load_split  # noqa: E402

MB = {"M15": 100000, "H1": 55000, "H4": 13500, "D1": 60000}
CASES = [("ew_w3", {}), ("ew_w3", {"band": "wide", "k": 3.0}), ("ew_w5", {}),
         ("ew_w5", {"band": "wide", "k": 3.0}), ("ew_abc", {}), ("ew_abc", {"band": "wide", "k": 3.0}),
         ("ew_w3", {"side": "long"}), ("ew_abc", {"side": "short"})]


def scfg_for(base, name, tf, params):
    s = copy.deepcopy(base.strategy)
    s.name, s.timeframe, s.params = name, tf, dict(params)
    return s


def truncation() -> int:
    base = load_config(MARKETS["GOLD"])
    bad_total = 0
    for tf in ("H1", "M15", "D1"):
        tr, _, _ = load_split("GOLD", tf, MB[tf])
        df = tr.iloc[-9000:]
        for name, params in CASES:
            mod = get_strategy(name)
            scfg = scfg_for(base, name, tf, params)
            full = mod.add_indicators(df, scfg)
            bad = 0
            for cut in (1501, 3001, 5003, 7777, len(df) - 1):
                part = mod.add_indicators(df.iloc[:cut], scfg)
                for col in part.columns:
                    a = full[col].iloc[:cut].to_numpy(float)
                    b = part[col].to_numpy(float)
                    if not np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9):
                        bad += 1
                        print(f"  MISMATCH {name} {params} {tf} cut {cut} col {col}")
            n = int((full["ew_sig"] != 0).sum())
            print(f"{name:<7}{tf:<4}{str(params):<36} signals {n:5d}  {'OK' if bad == 0 else 'FAIL'}")
            bad_total += bad
    return bad_total


def geometry() -> None:
    print("\nMedian structural stop / target in ATRs (train only, all markets x TFs):")
    for name in ("ew_w3", "ew_w5", "ew_abc"):
        mod = get_strategy(name)
        for k in (2.0, 3.0):
            stops, tgts, nsig = [], [], 0
            for mk in ("GOLD", "BTCUSD", "US500Cash"):
                base = load_config(MARKETS[mk])
                for tf in ("M15", "H1", "H4", "D1"):
                    tr, _, _ = load_split(mk, tf, MB[tf])
                    scfg = scfg_for(base, name, tf, {"k": k, "target": "rr" if name == "ew_abc" else "struct"})
                    ind = mod.add_indicators(tr, scfg)
                    for i in np.nonzero(ind["ew_sig"].to_numpy() != 0)[0]:
                        if i < 200:
                            continue
                        s = mod.evaluate(ind.iloc[: i + 1], scfg)
                        if s.direction == 0:
                            continue
                        nsig += 1
                        stops.append(s.sl_distance / s.atr)
                        if name != "ew_abc":
                            tgts.append(s.tp_distance / s.atr)
            mt = f"{np.median(tgts):.2f}" if tgts else "-"
            print(f"{name:<7} k {k:.0f}: n {nsig:6d}  median SL {np.median(stops):.2f} ATR  "
                  f"median struct TP {mt} ATR  (TP/SL ratio of medians "
                  f"{(np.median(tgts) / np.median(stops)) if tgts else float('nan'):.2f})")


if __name__ == "__main__":
    bad = truncation()
    print("TRUNCATION TEST:", "PASS" if bad == 0 else f"FAIL ({bad})")
    geometry()
