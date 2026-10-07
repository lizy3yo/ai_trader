"""Final reporting pass for the grid/martingale audit (see results/grid_sim.py).

Produces the numbers the brief demands per config -- avg $/month on $500,
% positive months, baskets/month -- plus three stress tests that decide WHY
the strategy family fails:

  swap stress : the mandated $1.00/lot/night vs the literal MT5 points reading
                (96.61 pts = $96.61/lot/night). Grids carry legs for weeks.
  leverage    : 1:100 (the brief) vs 1:500 (what vendors tell you to use), to
                check whether ruin is a margin artefact or a price-path fact.
  ZERO COST   : spread 0, swap 0. If the grid still ruins with no costs at all,
                the failure is structural, not a cost problem.

  python results/grid_report.py --csv data/gold_m30.csv --offsets 60
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_sim import GridCfg, add_indicators, load_bars, run_window  # noqa: E402

HEADLINE = {
    "flat L10 s1.0 tp0.5": dict(mult=1.0, step_atr=1.0, max_legs=10, tp_frac=0.5,
                                on_max_legs="hold"),
    "flat L5 s1.5 tp0.5": dict(mult=1.0, step_atr=1.5, max_legs=5, tp_frac=0.5,
                               on_max_legs="hold"),
    "flat L3 s3.0 tp0.5": dict(mult=1.0, step_atr=3.0, max_legs=3, tp_frac=0.5,
                               on_max_legs="hold"),
    "mart x2.0 L5 s1.0": dict(mult=2.0, step_atr=1.0, max_legs=5, tp_frac=0.5,
                              on_max_legs="hold"),
    "ctl flat L10 CUT": dict(mult=1.0, step_atr=1.0, max_legs=10, tp_frac=0.5,
                             on_max_legs="cut"),
    "ctl single L1 CUT": dict(mult=1.0, step_atr=1.0, max_legs=1, tp_frac=0.5,
                              on_max_legs="cut"),
}


def med(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return round(float(np.median(v)), 2) if v else None


def study(bars, cfg, offsets, horizon):
    ind = add_indicators(bars, cfg)
    n = len(bars["close"])
    warm = max(cfg.atr_period, cfg.rsi_period, cfg.ema_slow) * 5
    last = max(warm, n - horizon)
    starts = np.unique(np.linspace(warm, last, offsets).astype(int))
    rows = [run_window(bars, ind, cfg, int(s), min(n, int(s) + horizon)) for s in starts]
    ruined = [r for r in rows if r["ruin"]]
    alive = [r for r in rows if not r["ruin"]]
    return {
        "cfg": cfg.name, "runs": len(rows),
        "ruin_pct": round(100.0 * len(ruined) / len(rows), 1),
        "median_days_to_ruin": (round(float(np.median([r["days_alive"] for r in ruined])), 1)
                                if ruined else None),
        "p10_days_to_ruin": (round(float(np.percentile([r["days_alive"] for r in ruined], 10)), 1)
                             if ruined else None),
        # ---- the mandated per-config monthly reporting, on a $500 account ----
        "median_win_rate_pct": med(rows, "win_rate"),
        "median_avg_usd_per_month": med(rows, "avg_month_usd"),
        "median_pos_months_pct": med(rows, "pos_months_pct"),
        "median_baskets_per_month": med(rows, "baskets_per_month_rate"),
        "median_worst_month_usd": med(rows, "worst_month"),
        # the marketing number: $/month over the climb to peak equity
        "median_usd_per_month_while_climbing": med(rows, "usd_per_month_while_climbing"),
        "median_peak_equity": med(rows, "peak_equity"),
        "median_days_to_peak": med(rows, "days_to_peak"),
        "survivors": len(alive),
        "median_avg_usd_per_month_survivors": (med(alive, "avg_month_usd") if alive else None),
        # ---- risk ----
        "worst_basket_dd_usd": round(max(r["max_basket_dd"] for r in rows), 2),
        "median_basket_dd_usd": med(rows, "max_basket_dd"),
        "worst_margin_used_usd": round(max(r["max_used_margin"] for r in rows), 2),
        "median_max_legs_seen": med(rows, "max_legs_seen"),
        "margin_refusals_total": int(sum(r["margin_blocked"] for r in rows)),
        "stopouts_total": int(sum(r["stopouts"] for r in rows)),
        "swap_paid_median": med(rows, "swap_paid"),
        "spread_paid_median": med(rows, "spread_paid"),
        "full_cfg": asdict(cfg),
    }


HDR = (f"{'variant':<34}{'RUIN%':>7}{'medTTR':>8}{'win%':>7}{'$/mo':>9}{'posM%':>7}"
       f"{'bsk/mo':>8}{'climb$/mo':>11}{'peak$':>8}{'bskDD':>9}{'margin':>8}")


def line(s, label=None):
    return (f"{(label or s['cfg']):<34}{s['ruin_pct']:>7.1f}"
            f"{(s['median_days_to_ruin'] or 0):>8.0f}{(s['median_win_rate_pct'] or 0):>7.1f}"
            f"{(s['median_avg_usd_per_month'] or 0):>9.2f}{(s['median_pos_months_pct'] or 0):>7.0f}"
            f"{(s['median_baskets_per_month'] or 0):>8.1f}"
            f"{(s['median_usd_per_month_while_climbing'] or 0):>11.2f}"
            f"{(s['median_peak_equity'] or 0):>8.0f}{(s['median_basket_dd_usd'] or 0):>9.0f}"
            f"{s['worst_margin_used_usd']:>8.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/gold_m30.csv")
    ap.add_argument("--offsets", type=int, default=60)
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--out", default="results/grid_report.json")
    args = ap.parse_args()

    bars = load_bars(args.csv)
    bpd = max(1, 1440 // bars["tf_min"])
    horizon = args.horizon_days * bpd
    print(f"{args.csv}: {len(bars['close'])} bars {bars['time'][0].date()} -> "
          f"{bars['time'][-1].date()} | {args.offsets} windows x {args.horizon_days}d "
          f"| $500 start, ruin at $100, 1:100\n")

    out = {"data": args.csv, "windows": args.offsets,
           "horizon_days": args.horizon_days, "baseline": [], "stress": []}

    print("BASELINE  (spread 55pts, swap $1.00/lot/night, 1:100)")
    print(HDR)
    for name, kw in HEADLINE.items():
        s = study(bars, replace(GridCfg(), name=name, **kw), args.offsets, horizon)
        out["baseline"].append(s)
        print(line(s))

    print("\nSTRESS TESTS on 'flat L10 s1.0 tp0.5'")
    print(HDR)
    base = replace(GridCfg(), name="flat L10 s1.0 tp0.5", **HEADLINE["flat L10 s1.0 tp0.5"])
    variants = [
        ("swap = $96.61/lot/nt (literal MT5 pts)", replace(base, swap_per_lot_night=96.61)),
        ("leverage 1:500 (vendor advice)", replace(base, leverage=500.0)),
        ("ZERO COST: spread 0, swap 0", replace(base, spread_points=0.0, swap_per_lot_night=0.0)),
        ("ZERO COST + 1:500 leverage", replace(base, spread_points=0.0, swap_per_lot_night=0.0,
                                               leverage=500.0)),
        ("$5000 account (10x capital)", replace(base, start_equity=5000.0, ruin_equity=1000.0)),
    ]
    for label, cfg in variants:
        s = study(bars, replace(cfg, name=label), args.offsets, horizon)
        out["stress"].append(s)
        print(line(s, label))

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
