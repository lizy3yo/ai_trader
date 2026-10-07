"""Scratch diagnostic (breakouts family): why did the donchian sweep fail train?
Runs a handful of configs directly on the TRAIN split and prints raw metrics.
Never touches the test split.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, run_backtest
from bot.config import load_config
from tools.research import apply_params, load_split, metrics

base = load_config()

cases = [
    ("brk_donchian", "M30", {"entry_period": 20, "confirm": "close", "trend_filter": 0,
                             "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25,
                             "session.window": "01:00-23:00"}),
    ("brk_donchian", "M30", {"entry_period": 55, "confirm": "close", "trend_filter": 1,
                             "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 1.5,
                             "session.window": "09:00-19:00"}),
    ("brk_donchian", "H1", {"entry_period": 55, "confirm": "close", "trend_filter": 0,
                            "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25,
                            "session.window": "01:00-23:00"}),
    ("brk_donchian", "H1", {"entry_period": 20, "confirm": "wick", "trend_filter": 0,
                            "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 1.5,
                            "session.window": "09:00-19:00"}),
    ("brk_squeeze", "M30", {"squeeze_pct": 0.3, "squeeze_lookback": 100,
                            "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25,
                            "session.window": "01:00-23:00"}),
    ("brk_squeeze", "H1", {"squeeze_pct": 0.3, "squeeze_lookback": 100,
                           "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25,
                           "session.window": "01:00-23:00"}),
]

for strat, tf, params in cases:
    train, val, _ = load_split(base.symbol, tf, 60000)
    cfg = apply_params(base, strat, tf, params)
    m = metrics(run_backtest(cfg, train, DEFAULT_GOLD_ECON, tracker=None, verbose=False))
    print(f"{strat:14s} {tf:4s} {params}")
    print(f"   TRAIN: ret {m['return_pct']:+7.1f}%  trades {m['trades']:4d}  "
          f"win {m['win_rate']:4.1f}%  PF {m['pf']:5.3f}  posM {m['pos_months_pct']:3.0f}%  "
          f"maxDD {m['max_dd']:.0f}")
