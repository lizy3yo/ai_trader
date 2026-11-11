"""DIVERSIFIED TREND FOLLOWING test (2026-11-09): the managed-futures idea.
One simple breakout rule on EVERY market at once (indices, energies, metals,
FX). Each market alone is weak; the hope is that many uncorrelated weak edges
add up to a steady portfolio.

PRE-REGISTERED (written before any result was seen; textbook values, no tuning):
  universe  data/multi/ (XM cash indices + energies, spot metals, FX majors and
            minors) with >= 1500 D1 bars and estimated spread <= 0.15% of price
  rules     trend_donchian D1, close above/below EMA200 in the trade direction,
            both sides, the user's framework (ATR stop, fixed reward >= 1:1.5):
            T55   55-day breakout, stop 3 ATR, target 5 ATR  (gold rule's 1:1.67)
            T20   20-day breakout, stop 3 ATR, target 5 ATR
            T55w  55-day breakout, stop 3 ATR, target 15 ATR (1:5, lets trends run)
  costs     1.3 x max(live, recorded) spread scaled to price; broker swap per
            night (scaled to price), triple day from MT5
  baseline  random entries, both sides, same EMA200 filter, same stop/target,
            same markets and costs, ~same trade count (4 seeds)
  pass      pooled avg R > 0 in all 3 periods, beats every random seed, and the
            K-position portfolio has monthly t >= 3

Usage: python tools/trend_universe.py [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.backtest import load_csv_bars, run_backtest   # noqa: E402
from tools.universe_test import econ_of, est_spread, log_of, make_cfg, portfolio   # noqa: E402

DIR = Path("data/multi")
OUT = Path("results/trend_universe")
RULES = {
    "T55": ({"lookback": 55}, 3.0, 5.0),
    "T20": ({"lookback": 20}, 3.0, 5.0),
    "T55w": ({"lookback": 55}, 3.0, 15.0),
}
SEEDS = 4
PERIODS = [("2008-15", "2000-01-01", "2016-01-01"), ("2016-20", "2016-01-01", "2021-01-01"),
           ("2021-26", "2021-01-01", "2027-01-01")]
MAX_SPREAD = 0.0015


def universe() -> list[str]:
    keep = []
    for f in sorted(DIR.glob("*_econ.json")):
        e = json.loads(f.read_text(encoding="utf-8"))
        csv = DIR / f"{e['symbol'].lower()}_d1.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, usecols=["time", "close", "spread"])
        if len(df) < 1500 or e["symbol"] in ("GOLD24-7",):     # duplicate of GOLD
            continue
        if est_spread(e, df) / float(df.close.iloc[-1]) > MAX_SPREAD:
            continue
        keep.append(e["symbol"])
    return keep


def cfg_for(sym, e, df, strat, params, sl, tp):
    cfg = make_cfg(sym, e, df, strat, params, {})
    cfg.strategy.atr_sl_mult, cfg.strategy.atr_tp_mult = sl, tp
    if e["swap_mode"] != 5:          # points: financing is a % of notional, scale with price
        cfg.backtest.swap_ref_price = float(df.close.iloc[-1])
    return cfg


def job(sym: str) -> list[dict]:
    e = json.loads((DIR / f"{sym.lower()}_econ.json").read_text(encoding="utf-8"))
    bars = load_csv_bars(str(DIR / f"{sym.lower()}_d1.csv"))
    df = pd.read_csv(DIR / f"{sym.lower()}_d1.csv", usecols=["close", "spread"])
    econ = econ_of(e)
    rows = []
    for rule, (params, sl, tp) in RULES.items():
        res = run_backtest(cfg_for(sym, e, df, "trend_donchian", params, sl, tp), bars, econ, verbose=False)
        rows += log_of(res, sym, rule)
        n = res.trades
        if n == 0:
            continue
        rcfg = cfg_for(sym, e, df, "rnd_baseline", {}, sl, tp)
        prob = max(n / len(bars), 1e-3)
        for _ in range(3):
            rcfg.strategy.params = {"prob": prob, "seed": 0, "dir": "both", "trend": 1}
            m = run_backtest(rcfg, bars, econ, verbose=False).trades
            prob = prob * 3 if m == 0 else min(prob * n / m, 1.0)
        for sd in range(1, SEEDS + 1):
            rcfg.strategy.params = {"prob": prob, "seed": sd, "dir": "both", "trend": 1}
            rows += log_of(run_backtest(rcfg, bars, econ, verbose=False), sym, rule, sd)
    return rows


def report(t: pd.DataFrame, syms: list[str]) -> str:
    lines = [f"DIVERSIFIED TREND test on {len(syms)} markets: {', '.join(syms)}\n"]
    for rule in RULES:
        real, rnd = t[(t.rule == rule) & (t.seed == -1)], t[(t.rule == rule) & (t.seed > 0)]
        if real.empty:
            continue
        r = real.R
        lines.append(f"== {rule}: {len(r)} trades, win {100*(r>0).mean():.0f}%, avg {r.mean():+.3f}R "
                     f"(swap {real.swapR.mean():+.3f}R), t {r.mean()/(r.std(ddof=1)/np.sqrt(len(r))):.1f}")
        seeds = rnd.groupby("seed").R.mean()
        lines.append("   random entries: " + ", ".join(f"{v:+.3f}" for v in seeds)
                     + f"  -> rule beats {int((seeds < r.mean()).sum())}/{len(seeds)}")
        lines.append("   " + " | ".join(
            f"{n}: {real[(real.entry>=a)&(real.entry<b)].R.mean():+.3f}R n={len(real[(real.entry>=a)&(real.entry<b)])}"
            f" (rnd {rnd[(rnd.entry>=a)&(rnd.entry<b)].R.mean():+.3f})" for n, a, b in PERIODS))
        by = real.groupby("symbol").R.agg(["count", "mean"])
        lines.append(f"   markets positive: {int((by['mean']>0).sum())}/{len(by)}; best "
                     + ", ".join(f"{s} {v:+.2f}" for s, v in by['mean'].nlargest(5).items())
                     + "; worst " + ", ".join(f"{s} {v:+.2f}" for s, v in by['mean'].nsmallest(5).items()))
        for k in (5, 10, 20):
            m = portfolio(real[real.entry >= "2008-01-01"], k)
            idx = pd.period_range(m.index.min(), m.index.max(), freq="M")
            m = m.reindex(idx, fill_value=0.0)
            eq = m.cumsum()
            dd = float((eq.cummax() - eq).max())
            ts = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if m.std() > 0 else 0
            yr = m.groupby(m.index.year).sum()
            mr = portfolio(rnd[(rnd.seed == 1) & (rnd.entry >= "2008-01-01")], k).reindex(idx, fill_value=0.0)
            lines.append(f"   max {k:>2} open: {m.mean():+.2f}R/month, t {ts:.1f}, months up {100*(m>0).mean():.0f}%, "
                         f"worst month {m.min():+.1f}R, max DD {dd:.1f}R, losing years {int((yr<0).sum())}/{len(yr)}"
                         f"   (random {mr.mean():+.2f}R/month)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    syms = universe()
    print(f"universe: {len(syms)} markets", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, syms):
            rows += r
    t = pd.DataFrame(rows)
    t["entry"], t["exit"] = pd.to_datetime(t.entry), pd.to_datetime(t.exit)
    t.to_csv(OUT / "trades.csv", index=False)
    txt = report(t, syms)
    print(txt)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
