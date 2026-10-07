"""Scratch diagnostic: does brk_openrange trade at all, and how badly does it lose?
TRAIN split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, run_backtest
from bot.config import load_config
from tools.research import apply_params, load_split, metrics

base = load_config()

cases = [
    ("M30", {"range_window": "01:00-08:00", "buffer_atr": 0.0, "direction": "both",
             "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.5,
             "session.window": "09:00-15:00"}),
    ("M30", {"range_window": "01:00-08:00", "buffer_atr": 0.0, "direction": "long",
             "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 1.5,
             "session.window": "09:00-19:00"}),
    ("H1", {"range_window": "09:00-15:00", "buffer_atr": 0.0, "direction": "both",
            "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.5,
            "session.window": "15:00-19:00"}),
    ("M30", {"range_window": "09:00-15:00", "buffer_atr": 0.3, "direction": "long",
             "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 3.5,
             "session.window": "15:00-22:00"}),
]

for tf, params in cases:
    train, val, _ = load_split(base.symbol, tf, 60000)
    cfg = apply_params(base, "brk_openrange", tf, params)
    r = run_backtest(cfg, train, DEFAULT_GOLD_ECON, tracker=None, verbose=False)
    m = metrics(r)
    print(f"{tf:4s} {params}")
    print(f"   TRAIN: ret {m['return_pct']:+7.1f}%  trades {m['trades']:4d}  "
          f"win {m['win_rate']:4.1f}%  PF {m['pf']:5.3f}  posM {m['pos_months_pct']:3.0f}%")
