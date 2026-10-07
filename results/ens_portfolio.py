"""PORTFOLIO ensemble: run two strategies as two independent sub-accounts that
split the $500, then add their monthly P&L.

The backtester holds ONE position at a time and runs ONE strategy, so a genuine
two-strategy portfolio cannot be simulated inside it. This does the honest
approximation: each leg gets its own slice of the capital, compounds on its own,
and the monthly tables are summed on the calendar month. That answers the
question the user actually cares about — does splitting across the two things
that work raise the % of POSITIVE MONTHS? — while the caveats stay explicit:
  * two legs can be in the market at once, so peak risk is the SUM of the legs
  * combined drawdown is measured at monthly granularity, not bar by bar
  * the broker's 0.01-lot minimum bites hard on a half-sized sub-account; legs
    that end up skipping trades for "even min lot risks too much" show up here
    as a collapsed trade count, which is itself the answer.

python results/ens_portfolio.py <legs.json>
legs.json: {"legA": {"strategy":..,"timeframe":..,"alloc":250,"params":{...}}, ...}
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


def leg_monthly(base, spec: dict, split: str, max_bars: int, span: dict | None = None):
    tf = spec["timeframe"]
    train, val, _test = load_split(base.symbol, tf, spec.get("max_bars", max_bars))
    bars = {"train": train, "val": val, "trainval": pd.concat([train, val])}[split]
    if span:
        # a COMMON calendar window for every leg. The slice happens after
        # load_split, so a leg's own TEST bars are already gone and can never
        # be pulled back in by a generous "to" date.
        lo = pd.Timestamp(span["from"]) if span.get("from") else bars.index[0]
        hi = pd.Timestamp(span["to"]) if span.get("to") else bars.index[-1]
        bars = bars.loc[(bars.index >= lo) & (bars.index <= hi)]
    cfg = apply_params(base, spec["strategy"], tf, spec["params"])
    cfg.capital.allocated = float(spec["alloc"])
    cfg.capital.hard_max = float(spec["alloc"])
    res = run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False)
    m = res.monthly[["month", "pnl"]].copy() if res.monthly is not None else pd.DataFrame()
    return res, m


def describe(tag: str, m: pd.DataFrame, trades: int) -> None:
    if not len(m):
        print(f"  {tag:<26} no months")
        return
    pos = float((m["pnl"] > 0).mean() * 100)
    print(f"  {tag:<26} months {len(m):3d} | avg ${m['pnl'].mean():+7.2f}/mo | "
          f"median ${m['pnl'].median():+7.2f} | pos {pos:3.0f}% | "
          f"worst ${m['pnl'].min():+7.0f} | best ${m['pnl'].max():+7.0f} | "
          f"trades {trades:4d} ({trades/len(m):.1f}/mo)")


def main() -> None:
    legs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    split = sys.argv[2] if len(sys.argv) > 2 else "trainval"
    max_bars = int(sys.argv[3]) if len(sys.argv) > 3 else 99999
    span = legs.pop("_span", None)
    base = load_config()

    print(f"PORTFOLIO ENSEMBLE  split={split}  max_bars={max_bars}  span={span}")
    tables, trades = {}, {}
    for name, spec in legs.items():
        res, m = leg_monthly(base, spec, split, max_bars, span)
        tables[name] = m.set_index("month")["pnl"] if len(m) else pd.Series(dtype=float)
        trades[name] = res.trades
        describe(f"{name} (${spec['alloc']:.0f})", m, res.trades)

    common = pd.concat(tables.values(), axis=1, keys=tables.keys()).fillna(0.0)
    combo = common.sum(axis=1).reset_index()
    combo.columns = ["month", "pnl"]
    describe("PORTFOLIO (sum)", combo, sum(trades.values()))
    print("\n  monthly: " + "  ".join(
        f"{r['month']}:{r['pnl']:+.0f}" for _, r in combo.iterrows()))
    tot = float(combo["pnl"].sum())
    print(f"\n  total P&L ${tot:+.2f} on ${sum(l['alloc'] for l in legs.values()):.0f} "
          f"over {len(combo)} months")


if __name__ == "__main__":
    main()
