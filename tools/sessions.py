"""Which trading sessions are actually profitable?

Runs a strategy over the FULL 24h week (Asia + London + NY, weekend off) on
TRAIN+VALIDATION data (never the locked test split), then breaks results down
by the trade's ENTRY hour on the broker-server clock:

  - per-hour: trades, win rate, total P&L
  - per-session bucket (server time, see the volatility profile to map them):
      asia   ~ 01:00-08:59   london ~ 09:00-15:59
      newyork~ 16:00-23:59   (typical for UTC+2/+3 gold brokers)
  - an hourly avg bar-range profile of the RAW DATA, so you can see where
    Asia's quiet hours and the London/NY ramps really sit on this broker's
    clock instead of guessing timezones.

Usage:
  python tools/sessions.py --strategy ema_cross --timeframe M30 \
      --params-file results/best.json            # {"cfg.ema_fast": 20, ...}
  python tools/sessions.py --profile-only --timeframe H1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, run_backtest                 # noqa: E402
from bot.config import load_config                                       # noqa: E402
from tools.research import apply_params, load_split                      # noqa: E402

BUCKETS = [("asia", 1, 9), ("london", 9, 16), ("newyork", 16, 24), ("rollover", 0, 1)]


def bucket_of(hour: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= hour < hi:
            return name
    return "?"


def profile(bars) -> None:
    rng = (bars["high"] - bars["low"])
    by_hour = rng.groupby(bars.index.hour).mean()
    top = set(by_hour.sort_values(ascending=False).head(6).index)
    print("\n--- raw data: avg bar range ($) by server hour ---")
    for h in range(24):
        if h in by_hour.index:
            v = by_hour[h]
            mark = "  <-- busy" if h in top else ""
            print(f"  {h:02d}:00  {v:6.2f}  {'#' * int(v / by_hour.max() * 40)}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy")
    ap.add_argument("--timeframe", default="M30")
    ap.add_argument("--params", help="param-set JSON inline")
    ap.add_argument("--params-file", help="param-set JSON file")
    ap.add_argument("--max-bars", type=int, default=60000)
    ap.add_argument("--profile-only", action="store_true",
                    help="just print the hourly volatility profile of the data")
    args = ap.parse_args()

    base = load_config()
    train, val, _test = load_split(base.symbol, args.timeframe, args.max_bars)
    bars = pd.concat([train, val])
    profile(bars)
    if args.profile_only:
        return
    if not args.strategy:
        raise SystemExit("--strategy required (or use --profile-only)")

    params = {}
    if args.params_file:
        params = json.loads(Path(args.params_file).read_text(encoding="utf-8-sig"))
    elif args.params:
        params = json.loads(args.params)
    params = {**params, "session.window": "00:00-23:59"}   # open EVERY session
    cfg = apply_params(base, args.strategy, args.timeframe, params)

    result = run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False)
    if not result.trade_log:
        print("\nno trades — nothing to bucket")
        return

    by_hour = defaultdict(lambda: [0, 0, 0.0])   # trades, wins, pnl
    by_bucket = defaultdict(lambda: [0, 0, 0.0])
    for t in result.trade_log:
        h = t["open_time"].hour
        for agg in (by_hour[h], by_bucket[bucket_of(h)]):
            agg[0] += 1
            agg[1] += 1 if t["pnl"] > 0 else 0
            agg[2] += t["pnl"]

    print(f"\n--- {args.strategy} {args.timeframe}: P&L by ENTRY hour "
          f"(train+val, 24h window) ---")
    print(f"  {'hour':<7}{'trades':>7}{'win%':>7}{'P&L':>10}")
    for h in sorted(by_hour):
        n, w, p = by_hour[h]
        print(f"  {h:02d}:00 {n:>7}{100.0 * w / n:>7.1f}{p:>+10.2f}")
    print("\n--- by session bucket (server clock) ---")
    for name, *_ in BUCKETS:
        if name in by_bucket:
            n, w, p = by_bucket[name]
            print(f"  {name:<9} trades {n:>4}  win {100.0 * w / n:>5.1f}%  P&L {p:>+10.2f}")
    print("\nUse a 'session.window' grid key to restrict entries to the "
          "profitable hours, then re-validate — hours cherry-picked from this "
          "table are a hypothesis, not a proven edge.")


if __name__ == "__main__":
    main()
