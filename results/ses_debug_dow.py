"""Scratch diagnostic (ses_ family): weekday breakdown of the ORB r1 winner
on TRAIN+VAL (never the locked test split)."""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, run_backtest
from bot.config import load_config
from tools.research import apply_params, load_split

base = load_config()
params = {"trend_filter": True, "buf_atr": 0.35, "max_range_atr": 99.0,
          "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25}

train, val, _ = load_split(base.symbol, "M15", 60000)
bars = pd.concat([train, val])
cfg = apply_params(base, "ses_orb", "M15", params)
r = run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False)

names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
by_dow = defaultdict(lambda: [0, 0, 0.0])
by_dir = defaultdict(lambda: [0, 0, 0.0])
for t in r.trade_log:
    for agg in (by_dow[t["open_time"].weekday()], by_dir[t["direction"]]):
        agg[0] += 1
        agg[1] += 1 if t["pnl"] > 0 else 0
        agg[2] += t["pnl"]

print(f"ORB M15 winner on train+val: trades={r.trades} wr={r.win_rate:.1f}% "
      f"pf={r.profit_factor:.3f} end_eq={r.end_equity:.2f}")
print("--- by weekday of entry ---")
for d in sorted(by_dow):
    n, w, p = by_dow[d]
    print(f"  {names[d]}  trades {n:>3}  win {100*w/n:5.1f}%  P&L {p:>+10.2f}")
print("--- by direction ---")
for d, (n, w, p) in by_dir.items():
    print(f"  {d:<6} trades {n:>3}  win {100*w/n:5.1f}%  P&L {p:>+10.2f}")
print("--- data span ---")
print("train:", train.index[0], "->", train.index[-1])
print("val  :", val.index[0], "->", val.index[-1])
