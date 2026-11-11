"""Evaluate a strategy config over its WHOLE data file with research costs.

Takes the strategy/session from a trading config (e.g. config.yaml) and the
risk + cost model from a research base (research_gold.yaml): fixed 1% risk,
real spread floor + recorded bar spreads, real swap. Prints R statistics and
yearly breakdown, optionally with swap switched off to show its cost.

Usage: python tools/full_eval.py config.yaml [--base research_gold.yaml] [--start 2013-06-01] [--noswap]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from bot.backtest import load_csv_bars, load_econ, run_backtest  # noqa: E402
from bot.config import load_config                             # noqa: E402


def merged(trade_cfg_path: str, base_path: str):
    tc, base = load_config(trade_cfg_path), load_config(base_path)
    cfg = copy.deepcopy(base)
    cfg.strategy = copy.deepcopy(tc.strategy)
    cfg.session = copy.deepcopy(tc.session)
    cfg.symbol = tc.symbol
    return cfg, tc.backtest.data_csv


def summarize(res, label: str) -> dict:
    tl = pd.DataFrame(res.trade_log)
    if tl.empty:
        print(f"{label}: no trades")
        return {}
    tl["R"] = tl.pnl / tl.risk_amount
    n = len(tl)
    t = tl.R.mean() / (tl.R.std(ddof=1) / np.sqrt(n)) if n > 1 else 0
    yrs = tl.groupby(pd.to_datetime(tl.open_time).dt.year).R.agg(["count", "mean", "sum"])
    out = dict(trades=n, win=round((tl.pnl > 0).mean() * 100, 1),
               pf=round(res.profit_factor, 2), avgR=round(tl.R.mean(), 3), t=round(t, 2),
               totalR=round(tl.R.sum(), 1), swapR=round((tl.swap / tl.risk_amount).sum(), 1),
               maxdd_pct=round(100 * res.max_drawdown / res.start_equity, 1))
    print(f"{label}: {out}")
    print("   yearly R: " + "  ".join(f"{y}:{r['sum']:+.1f}({int(r['count'])})" for y, r in yrs.iterrows()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--base", default="research_gold.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--noswap", action="store_true", help="also run with swap = 0 for comparison")
    args = ap.parse_args()
    cfg, data_csv = merged(args.config, args.base)
    bars = load_csv_bars(args.data or data_csv)
    if args.start:
        bars = bars[bars.index >= args.start]
    econ = load_econ(cfg.symbol)
    summarize(run_backtest(cfg, bars, econ, verbose=False), f"{cfg.strategy.name} {cfg.strategy.timeframe} (real swap)")
    if args.noswap:
        c2 = copy.deepcopy(cfg)
        c2.backtest.swap_long_points = c2.backtest.swap_short_points = 0.0
        summarize(run_backtest(c2, bars, econ, verbose=False), f"{cfg.strategy.name} {cfg.strategy.timeframe} (no swap)")


if __name__ == "__main__":
    main()
