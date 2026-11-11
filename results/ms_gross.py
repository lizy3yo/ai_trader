"""Diagnostic (not a selection step): the pre-registered default of each ms_ method
on train+val with REAL costs vs ZERO costs (no spread, no swap), to see whether
there is any gross edge that costs destroy. Train+val only; TEST never touched.
  python results/ms_gross.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tools.factory import MARKETS, _base, _run, _splits  # noqa: E402
from tools.research import apply_params  # noqa: E402

MB = {"M15": 100000, "H1": 55000, "H4": 13500, "D1": 60000}
CASES = [
    ("ms_smc", {"cfg.atr_sl_mult": 1.5, "cfg.atr_tp_mult": 3.0}, ["M15", "H1"]),
    ("ms_smc", {"sweep": 0, "pd": 0, "cfg.atr_sl_mult": 1.5, "cfg.atr_tp_mult": 3.0}, ["M15", "H1"]),
    ("ms_trend", {"length": 5, "cfg.atr_sl_mult": 1.5, "cfg.atr_tp_mult": 3.0}, ["M15", "H1", "H4", "D1"]),
    ("ms_sd", {"enh": "none", "cfg.atr_sl_mult": 1.0, "cfg.atr_tp_mult": 3.0}, ["M15", "H1", "H4", "D1"]),
    ("ms_sd", {"enh": "all", "cfg.atr_sl_mult": 1.0, "cfg.atr_tp_mult": 3.0}, ["H1", "H4", "D1"]),
]


def main() -> None:
    rows = []
    for strat, params, tfs in CASES:
        for mk in ("GOLD", "BTCUSD", "US500Cash"):
            for tf in tfs:
                cfg = apply_params(_base(mk), strat, tf, params)
                _, _, tv = _splits(mk, tf, MB[tf])
                real = _run(cfg, tv)
                z = copy.deepcopy(cfg)
                z.backtest.spread_points = 0
                z.backtest.use_bar_spread = False
                z.backtest.swap_long_points = 0.0
                z.backtest.swap_short_points = 0.0
                gross = _run(z, tv)
                rows.append(dict(strategy=strat, params={k: v for k, v in params.items() if "." not in k},
                                 market=mk, tf=tf, n=real["trades"], win=real.get("win_rate"),
                                 real_avgR=real["avg_r"], real_t=real["t_stat"],
                                 gross_avgR=gross["avg_r"], gross_t=gross["t_stat"]))
                print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    print(d.to_string())
    print("\nmean avgR by strategy:", d.groupby(["strategy"])[["real_avgR", "gross_avgR"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
