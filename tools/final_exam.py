"""FINAL EXAM: one run of a finalist on the locked TEST split, recorded so it
cannot quietly be retried.

  python tools/final_exam.py --strategy X --market GOLD --tf H1 --params '<json>' [--max-bars 60000]

The test window is the last 20% of the most recent `max_bars` bars (same split
as the factory). Indicators warm up on the preceding bars only (no trades
there). Results append to results/final_exams.csv. A second exam of the same
strategy + market + timeframe is refused: re-tuning after seeing the exam
turns it into practice data. Also runs the random-entry check on the test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tools.factory import MARKETS, _base, edge_p   # noqa: E402
from tools.research import apply_params, load_split, run_one  # noqa: E402

LOG = ROOT / "results" / "final_exams.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--market", required=True, choices=list(MARKETS))
    ap.add_argument("--tf", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--max-bars", type=int, default=60000)
    ap.add_argument("--dir", default="long", choices=["long", "short", "both"])
    ap.add_argument("--trend", type=int, default=0)
    ap.add_argument("--exit-from-strategy", action="store_true")
    args = ap.parse_args()

    if LOG.exists():
        done = pd.read_csv(LOG)
        hit = done[(done.strategy == args.strategy) & (done.market == args.market) & (done.tf == args.tf)]
        if len(hit):
            raise SystemExit(f"already examined on {hit.iloc[0]['ts']} — an exam can only be taken once")

    params = json.loads(args.params)
    cfg = apply_params(_base(args.market), args.strategy, args.tf, params)
    tr, va, te = load_split(args.market, args.tf, args.max_bars)
    ctx = pd.concat([tr, va]).iloc[-10000:]
    m = run_one(cfg, te, ctx)
    _, rmean, rp95, p = edge_p(args.market, args.tf, args.strategy, params, te, args.dir,
                               args.trend, args.exit_from_strategy, seeds=30)
    passed = m["avg_r"] > 0 and p < 0.10
    print(f"FINAL EXAM {args.strategy} {args.market} {args.tf}  window {te.index[0]} -> {te.index[-1]}")
    print(f"  trades {m['trades']}  avgR {m['avg_r']:+.3f}  t {m['t_stat']:+.2f}  PF {m['pf']}  "
          f"posM {m['pos_months_pct']}%  maxDD {m['max_dd']:.0f}  swap {m['swap']:+.0f}")
    print(f"  random entries on the same window: mean {rmean:+.3f}, 95th pct {rp95:+.3f}, p_random {p:.3f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["ts", "strategy", "market", "tf", "params", "start", "end", "trades",
                        "avgR", "t", "pf", "rand_mean", "p_random", "verdict"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.strategy, args.market, args.tf,
                    json.dumps(params, sort_keys=True), te.index[0], te.index[-1], m["trades"],
                    m["avg_r"], m["t_stat"], m["pf"], round(rmean, 3), round(p, 3),
                    "PASS" if passed else "FAIL"])


if __name__ == "__main__":
    main()
