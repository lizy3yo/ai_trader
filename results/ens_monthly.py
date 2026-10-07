"""Monthly profile of one config on a $500 account (train and validation).

python results/ens_monthly.py <strategy> <timeframe> <grid_json> <name> [max_bars]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, run_backtest    # noqa: E402
from bot.config import load_config                          # noqa: E402
from tools.research import apply_params, load_split         # noqa: E402


def report(tag, res, start_equity):
    m = res.monthly
    n = len(m) if m is not None else 0
    if not n:
        print(f"  {tag}: no months")
        return
    pos = float((m["pnl"] > 0).mean() * 100)
    print(f"  {tag}: {n} months | avg ${m['pnl'].mean():+.2f}/mo | median "
          f"${m['pnl'].median():+.2f}/mo | pos {pos:.0f}% | worst ${m['pnl'].min():+.0f} | "
          f"best ${m['pnl'].max():+.0f} | trades {res.trades} "
          f"({res.trades / n:.1f}/mo) | end ${res.end_equity:.2f} | maxDD ${res.max_drawdown:.0f}")
    print("    " + "  ".join(f"{r['month']}:{r['pnl']:+.0f}" for _, r in m.iterrows()))


def main() -> None:
    strategy, tf, gridfile, name = sys.argv[1:5]
    max_bars = int(sys.argv[5]) if len(sys.argv) > 5 else 99999
    params = json.loads(Path(gridfile).read_text(encoding="utf-8-sig"))[name]
    params = {k: v for k, v in params.items() if k != "_strategy"}
    base = load_config()
    train, val, _test = load_split(base.symbol, tf, max_bars)
    cfg = apply_params(base, strategy, tf, params)
    print(f"{strategy} {tf} {name} {json.dumps(params)}")
    for tag, bars in (("TRAIN", train), ("VAL  ", val)):
        res = run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False)
        report(tag, res, 500.0)


if __name__ == "__main__":
    main()
