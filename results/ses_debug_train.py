"""Scratch diagnostic (ses_ family): run defaults on TRAIN split only, print
trade counts so we can tell 'no trades / bug' apart from 'trades but losing'.
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
    ("ses_orb", "M30", {"mode": "fade"}),
    ("ses_orb", "M15", {"mode": "fade"}),
    ("ses_orb", "M15", {"trend_filter": True}),
    ("ses_nyimp", "M30", {"imp_min_atr": 0.8}),
    ("ses_nyimp", "M30", {"imp_min_atr": 0.8, "mode": "fade"}),
    ("ses_nyimp", "M15", {"imp_min_atr": 0.8}),
    ("ses_meanrev", "M30", {"session.window": "01:00-08:00", "z_in": 2.4,
                            "cfg.atr_sl_mult": 2.5, "cfg.atr_tp_mult": 1.2}),
    ("ses_meanrev", "M15", {"session.window": "01:00-08:00", "z_in": 2.4,
                            "cfg.atr_sl_mult": 2.5, "cfg.atr_tp_mult": 1.2}),
]

for strat, tf, params in cases:
    train, val, _ = load_split(base.symbol, tf, 60000)
    cfg = apply_params(base, strat, tf, params)
    r = run_backtest(cfg, train, DEFAULT_GOLD_ECON, tracker=None, verbose=False)
    m = metrics(r)
    print(f"{strat:12s} {tf} {str(params):55s} trades={m['trades']:4d} "
          f"ret={m['return_pct']:+7.1f}% wr={m['win_rate']:4.1f}% pf={m['pf']:5.2f} "
          f"posM={m['pos_months_pct']:3.0f}%")
    for t in r.trade_log[:3]:
        print(f"    {t['open_time']} {t['direction']} pnl={t['pnl']:+8.2f} exit={t['exit']}")
