"""Parameter sweep with a train/test split, on REAL exported bars.

Tunes the strategy on the FIRST 70% of history and validates the top
configurations on the LAST 30% they never saw. A config only counts as a
candidate if it is profitable in BOTH halves — that is the (imperfect but
honest) guard against curve-fitting. Run tools/export_bars.py first.

Usage: python tools/sweep.py
"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, load_csv_bars, run_backtest  # noqa: E402
from bot.config import load_config                                       # noqa: E402

GRID = {
    "timeframe": ["M15", "M30", "H1"],
    "ema": [(9, 21), (12, 26), (20, 50)],
    "atr_sl_mult": [1.5, 2.0],
    "atr_tp_mult": [2.25, 3.0],
}
MIN_TRADES_TRAIN = 25


def run(cfg, bars, label):
    r = run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False)
    ret = 100.0 * (r.end_equity / r.start_equity - 1.0)
    return {
        "label": label, "return_pct": ret, "trades": r.trades,
        "win_rate": r.win_rate, "pf": r.profit_factor, "dd": r.max_drawdown,
        "end": r.end_equity,
    }


def main() -> None:
    base = load_config()
    results = []
    combos = list(itertools.product(
        GRID["timeframe"], GRID["ema"], GRID["atr_sl_mult"], GRID["atr_tp_mult"]))
    print(f"sweeping {len(combos)} configurations (train = first 70%, test = last 30%)\n")

    data = {}
    for tf in GRID["timeframe"]:
        path = Path(f"data/{base.symbol.lower()}_{tf.lower()}.csv")
        if not path.exists():
            raise SystemExit(f"{path} missing — run: python tools/export_bars.py")
        bars = load_csv_bars(path)
        cut = int(len(bars) * 0.7)
        data[tf] = (bars.iloc[:cut], bars.iloc[cut:])

    for tf, (ef, es), slm, tpm in combos:
        cfg = copy.deepcopy(base)
        cfg.strategy.timeframe = tf
        cfg.strategy.ema_fast, cfg.strategy.ema_slow = ef, es
        cfg.strategy.atr_sl_mult, cfg.strategy.atr_tp_mult = slm, tpm
        train, test = data[tf]
        label = f"{tf} ema{ef}/{es} sl{slm} tp{tpm}"
        tr = run(cfg, train, label)
        if tr["trades"] < MIN_TRADES_TRAIN or tr["return_pct"] <= 0:
            continue
        te = run(cfg, test, label)
        results.append((tr, te))

    if not results:
        print("NO configuration was profitable on the training window.")
        print("Honest conclusion: this entry logic has no edge on recent gold "
              "at this broker's spread. Do not trade it live; the strategy in "
              "bot/strategy.py needs a different idea, not different numbers.")
        return

    results.sort(key=lambda p: p[1]["return_pct"], reverse=True)
    print(f"{'configuration':<28}{'TRAIN ret%':>10}{'PF':>6}{'trades':>7}"
          f"{'TEST ret%':>10}{'PF':>6}{'trades':>7}{'maxDD':>8}")
    for tr, te in results:
        print(f"{tr['label']:<28}{tr['return_pct']:>+10.1f}{tr['pf']:>6.2f}"
              f"{tr['trades']:>7}{te['return_pct']:>+10.1f}{te['pf']:>6.2f}"
              f"{te['trades']:>7}{te['dd']:>8.0f}")

    both = [(tr, te) for tr, te in results if te["return_pct"] > 0]
    print()
    if both:
        tr, te = both[0]
        print(f"BEST train+test survivor: {tr['label']}")
        print(f"  train {tr['return_pct']:+.1f}% (PF {tr['pf']:.2f}, {tr['trades']} trades) | "
              f"test {te['return_pct']:+.1f}% (PF {te['pf']:.2f}, {te['trades']} trades)")
        print("  Update config.yaml strategy: accordingly, then demo-forward-test it.")
        print("  A survivor here is a candidate, NOT a guarantee — markets shift.")
    else:
        print("Every train-profitable config lost on the unseen test window.")
        print("Honest conclusion: the apparent train profits were curve-fit noise.")
        print("Do not trade this live. The entry logic needs a different idea.")


if __name__ == "__main__":
    main()
