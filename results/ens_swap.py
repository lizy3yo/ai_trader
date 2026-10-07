"""Overnight-swap drag estimate for one config (the backtester does NOT model
swap, so this quantifies what has to be subtracted by hand).

Broker measurement: GOLD long swap = -96.61 points/night/lot in POINTS mode.
point = 0.01 => -$0.9661 per 1.0 lot per night. Weekend triple-swap is applied
on Wednesday nights the way most brokers do (3x).

python results/ens_swap.py <strategy> <timeframe> <grid_json> <name> [max_bars] [split]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, run_backtest    # noqa: E402
from bot.config import load_config                          # noqa: E402
from tools.research import apply_params, load_split         # noqa: E402

SWAP_USD_PER_LOT_PER_NIGHT = 96.61 * 0.01     # -$0.9661, long side


def nights(open_t: pd.Timestamp, close_t: pd.Timestamp) -> float:
    """Count 00:00 rollovers crossed, charging Wednesday nights triple."""
    d0, d1 = open_t.normalize(), close_t.normalize()
    n = 0.0
    d = d0
    while d < d1:
        n += 3.0 if d.weekday() == 2 else 1.0   # Wed night = 3x
        d += pd.Timedelta(days=1)
    return n


def main() -> None:
    strategy, tf, gridfile, name = sys.argv[1:5]
    max_bars = int(sys.argv[5]) if len(sys.argv) > 5 else 99999
    split = sys.argv[6] if len(sys.argv) > 6 else "val"
    params = json.loads(Path(gridfile).read_text(encoding="utf-8-sig"))[name]
    params = {k: v for k, v in params.items() if k != "_strategy"}
    base = load_config()
    train, val, _test = load_split(base.symbol, tf, max_bars)
    bars = {"train": train, "val": val, "trainval": pd.concat([train, val])}[split]
    cfg = apply_params(base, strategy, tf, params)
    res = run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False)

    tot_nights = tot_swap = 0.0
    held = []
    for t in res.trade_log:
        n = nights(t["open_time"], t["close_time"])
        held.append(n)
        tot_nights += n
        tot_swap += n * t["lots"] * SWAP_USD_PER_LOT_PER_NIGHT
    n_tr = len(res.trade_log)
    months = len(res.monthly) if res.monthly is not None else 0
    print(f"{strategy} {tf} {name} [{split}]  trades {n_tr}  net P&L "
          f"${res.end_equity - res.start_equity:+.2f} over {months} months")
    if not n_tr:
        return
    print(f"  avg swap-nights/trade {tot_nights/n_tr:5.2f} (max {max(held):.0f}) | "
          f"trades held overnight {100*sum(1 for h in held if h > 0)/n_tr:4.0f}%")
    print(f"  TOTAL swap drag ${-tot_swap:,.2f}  =  ${-tot_swap/max(months,1):.2f}/month  "
          f"= {100*tot_swap/max(res.end_equity - res.start_equity, 1e-9):.1f}% of gross P&L")
    print("  (backtest P&L above does NOT include this — subtract it)")


if __name__ == "__main__":
    main()
