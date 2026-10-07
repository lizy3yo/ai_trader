"""Walk-forward re-optimisation harness (ens_ family).

Question: does RE-PICKING a strategy's parameters on a schedule, using only
prior data, beat leaving them fixed?

Protocol (strictly causal, TEST split never touched)
---------------------------------------------------
* Bars = the harness TRAIN + VALIDATION splits concatenated (tools.research
  .load_split); the locked TEST split is discarded on load and never read.
* The span is cut into calendar-month blocks. Trading starts after
  --min-hist months of history.
* At every re-optimisation boundary the harness backtests EVERY candidate
  parameter set over the SELECTION WINDOW, which ends at the boundary and
  starts either at the very beginning (--lookback expanding) or --lookback
  months earlier (rolling). Only bars strictly before the boundary are used.
* The winner by --objective is then traded FORWARD over the next
  --reopt months, unseen at selection time. Equity carries over from block to
  block (capital.allocated is reset to the running equity, so the % risk keeps
  compounding exactly as it does live).
* Warm-up: each block is fed exactly strat.warmup(cfg) bars of prior context,
  so run_backtest's first tradeable index lands on the block's first bar and
  no trade is ever counted twice.

Baselines printed alongside: every candidate run FIXED over the identical
forward span (same start equity, same costs), plus the best/worst of them.

Usage
  python results/ens_wf.py --strategy trend_donchian --timeframe H1 \
      --cands grids/ens_wf_donchian.json --reopt 12 --lookback 48 \
      --min-hist 48 --max-bars 99999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import DEFAULT_GOLD_ECON, _monthly_table, run_backtest  # noqa: E402
from bot.config import load_config                                        # noqa: E402
from bot.strategies import get_strategy                                   # noqa: E402
from tools.research import apply_params, load_split, metrics, score       # noqa: E402


def month_starts(idx: pd.DatetimeIndex) -> list[int]:
    """Positional index of the first bar of every calendar month."""
    per = idx.to_period("M")
    starts = [0]
    for i in range(1, len(idx)):
        if per[i] != per[i - 1]:
            starts.append(i)
    return starts


def run_slice(cfg, bars, lo: int, hi: int, strat, start_equity: float, clamp: bool = False):
    """Backtest bars[lo:hi) with exactly warmup bars of prior context, so the
    first tradeable bar is `lo`. Returns (BTResult, ok).

    clamp=True (selection windows only) starts the window at the earliest bar
    for which a full warm-up exists instead of failing — it shortens the
    window, it never reaches backwards past bar 0, so it stays causal."""
    w = strat.warmup(cfg.strategy)
    if lo - w < 0:
        if not clamp or hi - w <= 0:
            return None, False
        lo = w
    cfg.capital.allocated = start_equity
    cfg.capital.hard_max = start_equity
    sl = bars.iloc[lo - w: hi]
    return run_backtest(cfg, sl, DEFAULT_GOLD_ECON, tracker=None, verbose=False), True


def build(base, default_strategy: str, tf: str, params: dict):
    """A candidate may name its OWN strategy module via the "_strategy" key,
    which turns the walk-forward into a strategy-selection ensemble (e.g.
    donchian-vs-ORB re-chosen every N months) instead of a pure param sweep."""
    p = dict(params)
    sname = str(p.pop("_strategy", default_strategy))
    cfg = apply_params(base, sname, tf, p)
    return cfg, get_strategy(sname)


def objective_value(m: dict, name: str) -> float:
    if m["trades"] < 1:
        return -1e9
    if name == "score":
        return score(m)
    if name == "return":
        return m["return_pct"]
    if name == "pf":
        return m["pf"] if m["return_pct"] > 0 else -1e9
    if name == "posm":
        return m["pos_months_pct"] + m["return_pct"] / 100.0
    raise SystemExit(f"unknown objective {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--cands", required=True, help="JSON: {name: {param: value}}")
    ap.add_argument("--reopt", type=int, default=6, help="forward block length, months")
    ap.add_argument("--lookback", default="expanding",
                    help="'expanding' or an integer number of months (rolling)")
    ap.add_argument("--min-hist", type=int, default=24,
                    help="months of history before the first forward block")
    ap.add_argument("--objective", default="score",
                    choices=["score", "return", "pf", "posm"])
    ap.add_argument("--min-trades", type=int, default=8,
                    help="candidates with fewer trades in the selection window are skipped")
    ap.add_argument("--max-bars", type=int, default=99999)
    ap.add_argument("--equity", type=float, default=500.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    base = load_config()
    train, val, _test = load_split(base.symbol, args.timeframe, args.max_bars)
    bars = pd.concat([train, val])          # TEST split dropped here, on purpose
    del _test
    cands = json.loads(Path(args.cands).read_text(encoding="utf-8-sig"))

    starts = month_starts(bars.index)
    n_months = len(starts)
    print(f"WALK-FORWARD  {args.strategy} {args.timeframe}  "
          f"{bars.index[0]} -> {bars.index[-1]}  ({len(bars)} bars, {n_months} months)")
    print(f"  candidates={len(cands)}  reopt every {args.reopt}mo  "
          f"lookback={args.lookback}  min_hist={args.min_hist}mo  obj={args.objective}\n")

    boundaries = list(range(args.min_hist, n_months, args.reopt))
    if not boundaries:
        raise SystemExit("no forward blocks — lower --min-hist or --reopt")
    first_fwd = starts[boundaries[0]]

    equity = args.equity
    curve: list = []
    picks: list = []
    n_bt = 0

    for bi, mstart in enumerate(boundaries):
        b_lo = starts[mstart]
        m_end = min(mstart + args.reopt, n_months - 1)
        b_hi = starts[m_end] if m_end < n_months else len(bars)
        if m_end >= n_months - 1:
            b_hi = len(bars)
        if b_hi <= b_lo:
            break

        # ---- selection window: strictly before the boundary --------------
        if str(args.lookback).lower() == "expanding":
            s_lo = starts[0]
        else:
            s_lo = starts[max(0, mstart - int(args.lookback))]
        s_hi = b_lo

        best_name, best_obj, best_m = None, -1e18, None
        for name, params in cands.items():
            cfg, strat = build(base, args.strategy, args.timeframe, params)
            res, ok = run_slice(cfg, bars, s_lo, s_hi, strat, args.equity, clamp=True)
            n_bt += 1
            if not ok:
                continue
            m = metrics(res)
            if m["trades"] < args.min_trades:
                continue
            v = objective_value(m, args.objective)
            if v > best_obj:
                best_name, best_obj, best_m = name, v, m
        if best_name is None:
            best_name = picks[-1][1] if picks else list(cands)[0]
            best_m = None

        # ---- trade the winner forward -----------------------------------
        cfg, strat = build(base, args.strategy, args.timeframe, cands[best_name])
        res, ok = run_slice(cfg, bars, b_lo, b_hi, strat, equity)
        n_bt += 1
        if not ok:
            continue
        eq_before = equity
        equity = res.end_equity
        curve.extend(res.equity_curve)
        picks.append((str(bars.index[b_lo].date()), best_name, res.trades,
                      round(equity - eq_before, 2), round(equity, 2)))
        sel = (f"sel ret {best_m['return_pct']:+7.1f}% tr {best_m['trades']:4d}"
               if best_m else "sel  (no qualifying candidate — kept previous)")
        print(f"  {str(bars.index[b_lo].date())} -> {str(bars.index[b_hi-1].date())} "
              f"| pick {best_name:<14} | {sel} | fwd tr {res.trades:3d} "
              f"pnl {equity-eq_before:+8.2f} eq {equity:9.2f}")
        if equity < base.risk.min_equity_stop:
            print("  !! equity below min_equity_stop — walk-forward halted")
            break

    wf_monthly = _monthly_table(curve, args.equity)
    wf = {
        "return_pct": round(100.0 * (equity / args.equity - 1.0), 1),
        "end_equity": round(equity, 2),
        "trades": sum(p[2] for p in picks),
        "months": len(wf_monthly),
        "pos_months_pct": round(float((wf_monthly["pnl"] > 0).mean() * 100), 0)
        if len(wf_monthly) else 0.0,
        "worst_month": round(float(wf_monthly["pnl"].min()), 0) if len(wf_monthly) else 0.0,
        "avg_month_usd": round(float(wf_monthly["pnl"].mean()), 2) if len(wf_monthly) else 0.0,
    }

    # ---- fixed-parameter baselines over the identical forward span -------
    print(f"\n  --- FIXED baselines over the same forward span "
          f"({bars.index[first_fwd].date()} -> {bars.index[-1].date()}) ---")
    fixed = {}
    for name, params in cands.items():
        cfg, strat = build(base, args.strategy, args.timeframe, params)
        res, ok = run_slice(cfg, bars, first_fwd, len(bars), strat, args.equity)
        n_bt += 1
        if not ok:
            continue
        m = metrics(res)
        m["avg_month_usd"] = (round(float(res.monthly["pnl"].mean()), 2)
                              if res.monthly is not None and len(res.monthly) else 0.0)
        fixed[name] = m
        print(f"    {name:<16} ret {m['return_pct']:+8.1f}%  eq {m['end_equity']:9.2f}  "
              f"tr {m['trades']:4d}  pf {m['pf']:5.2f}  posM {m['pos_months_pct']:3.0f}%  "
              f"avg/mo ${m['avg_month_usd']:+8.2f}  maxDD {m['max_dd']:5.0f}")

    print(f"\n  === WALK-FORWARD  ret {wf['return_pct']:+8.1f}%  eq {wf['end_equity']:9.2f}  "
          f"tr {wf['trades']:4d}  posM {wf['pos_months_pct']:3.0f}%  "
          f"avg/mo ${wf['avg_month_usd']:+8.2f}  worstM {wf['worst_month']:+6.0f}  "
          f"months {wf['months']}")
    if fixed:
        rets = {k: v["return_pct"] for k, v in fixed.items()}
        bk = max(rets, key=rets.get)
        med = sorted(rets.values())[len(rets) // 2]
        print(f"      best fixed  = {bk} {rets[bk]:+.1f}% | median fixed = {med:+.1f}% | "
              f"WF {wf['return_pct']:+.1f}%  -> adaptivity "
              f"{'HELPS vs median' if wf['return_pct'] > med else 'does NOT beat median'}"
              f", {'beats' if wf['return_pct'] > rets[bk] else 'loses to'} best-fixed (hindsight)")
    print(f"  ({n_bt} backtests)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"wf": wf, "picks": picks, "fixed": fixed, "args": vars(args)},
            indent=1, default=str), encoding="utf-8")
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
