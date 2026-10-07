"""Scratch post-mortem: gross-vs-net edge for representative breakout configs.
Runs TRAIN split with the real 55pt spread and with spread=0 to see whether
breakout losses are cost-driven or adverse-selection-driven. Test split untouched.
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
    ("brk_squeeze", "H1", {"squeeze_pct": 0.3, "squeeze_lookback": 100,
                           "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25,
                           "session.window": "01:00-23:00"}),
    ("brk_openrange", "M30", {"range_window": "01:00-08:00", "buffer_atr": 0.0,
                              "direction": "both", "cfg.atr_sl_mult": 2.0,
                              "cfg.atr_tp_mult": 2.5, "session.window": "09:00-15:00"}),
    ("brk_retest", "H1", {"entry_period": 96, "wait_bars": 8, "pb_atr": 0.25,
                          "direction": "long", "cfg.atr_sl_mult": 2.0,
                          "cfg.atr_tp_mult": 1.5, "session.window": "09:00-19:00"}),
]

for strat, tf, params in cases:
    train, val, _ = load_split(base.symbol, tf, 60000)
    line = f"{strat:14s} {tf:4s}"
    for sp in (55.0, 0.0):
        cfg = apply_params(base, strat, tf, params)
        cfg.backtest.spread_points = sp
        m = metrics(run_backtest(cfg, train, DEFAULT_GOLD_ECON, tracker=None, verbose=False))
        line += (f" | spread{sp:3.0f}: PF {m['pf']:5.3f} win {m['win_rate']:4.1f}% "
                 f"ret {m['return_pct']:+6.1f}% n={m['trades']}")
    print(line)
