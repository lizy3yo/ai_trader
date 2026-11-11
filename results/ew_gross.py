"""Diagnostic (not a selection step): each ew_ model's pre-registered DEFAULT on train+val with
REAL costs vs ZERO costs (no spread, no swap). Train+val only; TEST never touched.
  python results/ew_gross.py  -> results/ew_gross.txt
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tools.factory import _base, _run, _splits  # noqa: E402
from tools.research import apply_params  # noqa: E402

MB = {"M15": 100000, "H1": 55000, "H4": 13500, "D1": 60000}
CASES = [
    ("ew_w3", {"k": 2.0, "band": "def", "target": "struct", "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 4.0}),
    ("ew_w5", {"k": 2.0, "band": "def", "target": "struct", "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.4}),
    ("ew_abc", {"k": 2.0, "band": "def", "target": "rr", "cfg.atr_sl_mult": 4.0, "cfg.atr_tp_mult": 8.0}),
]


def main() -> None:
    rows = []
    for strat, params in CASES:
        for mk in ("GOLD", "BTCUSD", "US500Cash"):
            for tf in MB:
                cfg = apply_params(_base(mk), strat, tf, params)
                _, _, tv = _splits(mk, tf, MB[tf])
                real = _run(cfg, tv)
                z = copy.deepcopy(cfg)
                z.backtest.spread_points = 0
                z.backtest.use_bar_spread = False
                z.backtest.swap_long_points = 0.0
                z.backtest.swap_short_points = 0.0
                gross = _run(z, tv)
                rows.append(dict(strategy=strat, market=mk, tf=tf, n=real["trades"],
                                 real_avgR=real["avg_r"], real_t=real["t_stat"],
                                 gross_n=gross["trades"], gross_avgR=gross["avg_r"], gross_t=gross["t_stat"]))
    d = pd.DataFrame(rows)
    out = [d.to_string()]
    g = d.groupby("strategy")
    out.append("\nmean avgR by strategy (unweighted over 12 market x TF cells):")
    out.append(g[["real_avgR", "gross_avgR"]].mean().round(3).to_string())
    d["gR"] = d["gross_avgR"] * d["gross_n"]
    d["rR"] = d["real_avgR"] * d["n"]
    w = d.groupby("strategy")[["gR", "rR", "gross_n", "n"]].sum()
    out.append("\ntrade-weighted avgR (all cells pooled): gross " +
               ", ".join(f"{s} {r.gR / r.gross_n:+.3f} (n {int(r.gross_n)})" for s, r in w.iterrows()) +
               " | real " + ", ".join(f"{s} {r.rR / r.n:+.3f}" for s, r in w.iterrows()))
    txt = "\n".join(out)
    print(txt)
    (ROOT / "results" / "ew_gross.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
