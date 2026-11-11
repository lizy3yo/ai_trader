"""Does a strategy beat RANDOM entries? (the core edge test)

Runs the candidate on a data split, then N seeds of rnd_baseline with the
same timeframe, session, direction, ATR stop/target, holding rules, trend
filter and costs, at a firing rate tuned to give roughly the same number of
trades. Reports where the candidate's average R ranks among the random runs.

  p_random = share of random seeds with avgR >= candidate's avgR
  -> < 0.05 means the entry timing itself carries information.

Usage:
  python tools/edge_test.py --config research_btc.yaml --strategy X --timeframe H1 \
      --params '{"lookback": 20, "cfg.atr_sl_mult": 2}' --dir long --trend 1 \
      [--split train|val|all] [--seeds 40]
Never use --split test during research (the locked test split is for the
final check of an already-chosen winner).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                     # noqa: E402

from bot.backtest import run_backtest, load_econ       # noqa: E402
from bot.config import load_config                     # noqa: E402
from tools.research import apply_params, load_split, r_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--dir", default="long", choices=["long", "short"])
    ap.add_argument("--trend", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val", "trainval", "all", "test"])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max-bars", type=int, default=60000)
    args = ap.parse_args()

    base = load_config(args.config)
    econ = load_econ(base.symbol)
    tr, va, te = load_split(base.symbol, args.timeframe, args.max_bars)
    import pandas as pd
    bars = {"train": tr, "val": va, "test": te, "trainval": pd.concat([tr, va]),
            "all": pd.concat([tr, va, te])}[args.split]

    params = json.loads(args.params)
    cfg = apply_params(base, args.strategy, args.timeframe, params)
    res = run_backtest(cfg, bars, econ, verbose=False)
    cand_r, cand_t = r_stats(res)
    n = res.trades
    print(f"candidate {args.strategy} {args.timeframe} [{args.split}]: trades {n}  avgR {cand_r:+.3f}  t {cand_t:+.2f}  PF {res.profit_factor:.2f}")
    if n < 10:
        print("too few trades for an edge test")
        return

    # random baseline keeps every cfg./session./backtest. setting, swaps the entry rule
    rparams = {k: v for k, v in params.items() if k.startswith(("cfg.", "session.", "backtest."))}
    rcfg = apply_params(base, "rnd_baseline", args.timeframe, rparams)
    # calibrate firing rate on seed 0
    prob = max(n / len(bars), 1e-4)
    for _ in range(4):
        rcfg.strategy.params = {"prob": prob, "seed": 0, "dir": args.dir, "trend": args.trend}
        m = run_backtest(rcfg, bars, econ, verbose=False).trades
        if m == 0:
            prob *= 3
            continue
        prob *= n / m
    rs = []
    for seed in range(1, args.seeds + 1):
        rcfg.strategy.params = {"prob": prob, "seed": seed, "dir": args.dir, "trend": args.trend}
        rr, _ = r_stats(run_backtest(rcfg, bars, econ, verbose=False))
        rs.append(rr)
    rs = np.array(rs)
    p = float((rs >= cand_r).mean())
    print(f"random x{args.seeds}: mean avgR {rs.mean():+.3f}  95th pct {np.percentile(rs, 95):+.3f}  "
          f"=> p_random {p:.3f}  {'EDGE' if p < 0.05 else 'NOT distinguishable from random'}")


if __name__ == "__main__":
    main()
