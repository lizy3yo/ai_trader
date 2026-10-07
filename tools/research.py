"""Strategy research harness with a LOCKED final-test split.

Data per timeframe is split 60% / 20% / 20% into TRAIN / VALIDATION / TEST.
Normal runs tune on TRAIN and rank on VALIDATION. The TEST split is touched
only via --final, exactly once, for the already-chosen winner — running it
repeatedly turns the test set into a second validation set and the results
into fiction. Do NOT pass --final during research.

Grid file (JSON): {"param": [v1, v2], "cfg.atr_sl_mult": [1.5, 2.0], ...}
Keys starting with "cfg." set StrategyCfg fields; others go to strategy params.

Examples:
  python tools/research.py --strategy ema_cross --timeframes M30 \
      --grid '{"cfg.ema_fast": [9, 20], "cfg.ema_slow": [21, 50]}'
  python tools/research.py --strategy my_strat --timeframes M30,H1 \
      --grid-file grids/my_strat.json --out results/my_strat.json
  python tools/research.py --final --strategy my_strat --timeframe M30 \
      --params '{"cfg.atr_sl_mult": 2.0, "lookback": 20}'
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, load_csv_bars, run_backtest  # noqa: E402
from bot.config import load_config                                       # noqa: E402

MAX_COMBOS = 200          # per invocation — keep grids small, overfitting grows with them
MIN_TRADES_TRAIN = 30     # fewer trades than this on train = statistical noise
DEFAULT_MAX_BARS = 60000  # most recent bars used per timeframe


def load_split(symbol: str, tf: str, max_bars: int):
    path = Path(f"data/{symbol.lower()}_{tf.lower()}.csv")
    if not path.exists():
        raise SystemExit(f"{path} missing — run: python tools/export_bars.py 99999")
    bars = load_csv_bars(path)
    if len(bars) > max_bars:
        bars = bars.iloc[-max_bars:]
    a, b = int(len(bars) * 0.6), int(len(bars) * 0.8)
    return bars.iloc[:a], bars.iloc[a:b], bars.iloc[b:]


def apply_params(cfg, strategy: str, tf: str, params: dict):
    """Grid keys: "cfg.X" -> StrategyCfg field, "session.X" -> SessionCfg field
    (trade_start/trade_end/friday_close as "HH:MM"), anything else ->
    strategy params dict. The session keys make the TRADING WINDOW itself a
    tunable — Asia-only, London-only, full 24h, etc."""
    cfg = copy.deepcopy(cfg)
    cfg.strategy.name = strategy
    cfg.strategy.timeframe = tf
    cfg.strategy.params = {}
    for key, value in params.items():
        if key.startswith("cfg."):
            field = key[4:]
            if not hasattr(cfg.strategy, field):
                raise SystemExit(f"unknown StrategyCfg field in grid: {key}")
            setattr(cfg.strategy, field, value)
        elif key == "session.window":
            # "HH:MM-HH:MM" sets start+end together (avoids nonsense
            # cross-products of independent start/end lists)
            start, end = (s.strip() for s in str(value).split("-"))
            cfg.session.trade_start = start
            cfg.session.trade_end = end
            cfg.session.friday_close = end
        elif key.startswith("session."):
            field = key[8:]
            if not hasattr(cfg.session, field):
                raise SystemExit(f"unknown SessionCfg field in grid: {key}")
            setattr(cfg.session, field, value)
        elif key.startswith("backtest."):
            # e.g. "backtest.spread_points": stress-test a config's cost
            # sensitivity — a scalp edge that dies at +20pts is not an edge
            field = key[9:]
            if not hasattr(cfg.backtest, field):
                raise SystemExit(f"unknown BacktestCfg field in grid: {key}")
            setattr(cfg.backtest, field, value)
        else:
            cfg.strategy.params[key] = value
    return cfg


def metrics(result) -> dict:
    monthly = result.monthly
    pnl = monthly["pnl"] if monthly is not None and len(monthly) else []
    pos_months = float((monthly["pnl"] > 0).mean() * 100) if len(pnl) else 0.0
    return {
        "return_pct": round(100.0 * (result.end_equity / result.start_equity - 1.0), 1),
        "end_equity": round(result.end_equity, 2),
        "trades": result.trades,
        "win_rate": round(result.win_rate, 1),
        "pf": round(result.profit_factor, 3) if result.trades else 0.0,
        "max_dd": round(result.max_drawdown, 0),
        "months": len(pnl),
        "pos_months_pct": round(pos_months, 0),
        "worst_month": round(float(monthly["pnl"].min()), 0) if len(pnl) else 0.0,
    }


def score(val: dict) -> float:
    """Ranking score on validation: consistency first, then quality of edge."""
    if val["return_pct"] <= 0 or val["trades"] < 10:
        return -1e9
    return (val["pos_months_pct"] * 2.0 + val["win_rate"]
            + 40.0 * min(val["pf"], 2.0) - val["max_dd"] / 10.0)


def run_one(cfg, bars):
    return metrics(run_backtest(cfg, bars, DEFAULT_GOLD_ECON, tracker=None, verbose=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--timeframes", default="M30")
    ap.add_argument("--grid", help="grid JSON inline")
    ap.add_argument("--grid-file", help="grid JSON file")
    ap.add_argument("--params", help="single param-set JSON (with --final)")
    ap.add_argument("--final", action="store_true",
                    help="evaluate ONE param set on the locked TEST split (once, at the very end)")
    ap.add_argument("--timeframe", help="timeframe for --final")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS)
    ap.add_argument("--out", help="write full ranked results JSON here")
    args = ap.parse_args()

    base = load_config()

    if args.final:
        if not (args.params and args.timeframe):
            raise SystemExit("--final needs --timeframe and --params")
        params = json.loads(args.params)
        cfg = apply_params(base, args.strategy, args.timeframe, params)
        train, val, test = load_split(base.symbol, args.timeframe, args.max_bars)
        print("=== FINAL TEST-SPLIT EVALUATION (do this exactly once) ===")
        for name, bars in (("train", train), ("val", val), ("TEST", test)):
            m = run_one(cfg, bars)
            print(f"  {name:>5}: ret {m['return_pct']:+7.1f}%  PF {m['pf']:5.2f}  "
                  f"win {m['win_rate']:4.1f}%  trades {m['trades']:4d}  "
                  f"posM {m['pos_months_pct']:3.0f}%  worstM {m['worst_month']:+6.0f}  "
                  f"maxDD {m['max_dd']:4.0f}")
        return

    if args.grid_file:
        grid = json.loads(Path(args.grid_file).read_text(encoding="utf-8-sig"))
    elif args.grid:
        grid = json.loads(args.grid)
    else:
        grid = {}
    keys = list(grid)
    combos = [dict(zip(keys, values))
              for values in itertools.product(*(grid[k] for k in keys))] or [{}]
    tfs = [t.strip().upper() for t in args.timeframes.split(",")]
    total = len(combos) * len(tfs)
    if total > MAX_COMBOS:
        raise SystemExit(f"{total} combos > cap {MAX_COMBOS} — shrink the grid "
                         "(big grids = curve-fitting, not research)")

    print(f"strategy={args.strategy} | {len(combos)} combos x {tfs} "
          f"= {total} runs | train->validate (test split stays locked)\n")
    rows = []
    for tf in tfs:
        train, val, _test = load_split(base.symbol, tf, args.max_bars)
        for params in combos:
            cfg = apply_params(base, args.strategy, tf, params)
            try:
                tr = run_one(cfg, train)
            except Exception as e:
                print(f"  [error] {tf} {params}: {e}")
                continue
            if tr["trades"] < MIN_TRADES_TRAIN or tr["return_pct"] <= 0:
                continue
            va = run_one(cfg, val)
            rows.append({"strategy": args.strategy, "timeframe": tf,
                         "params": params, "train": tr, "val": va,
                         "score": round(score(va), 1)})

    rows.sort(key=lambda r: r["score"], reverse=True)
    if not rows:
        print("No combo was profitable (with enough trades) on TRAIN. "
              "That is a result: this idea has no edge here — report it honestly.")
    else:
        print(f"{'tf':<5}{'params':<52}{'TRAIN ret%':>10}{'PF':>6}"
              f"{'VAL ret%':>9}{'PF':>6}{'win%':>6}{'posM%':>6}{'score':>8}")
        for r in rows[: args.top]:
            p = json.dumps(r["params"], separators=(",", ":"))[:50]
            print(f"{r['timeframe']:<5}{p:<52}{r['train']['return_pct']:>+10.1f}"
                  f"{r['train']['pf']:>6.2f}{r['val']['return_pct']:>+9.1f}"
                  f"{r['val']['pf']:>6.2f}{r['val']['win_rate']:>6.1f}"
                  f"{r['val']['pos_months_pct']:>6.0f}{r['score']:>8.1f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nfull results -> {args.out}")


if __name__ == "__main__":
    main()
