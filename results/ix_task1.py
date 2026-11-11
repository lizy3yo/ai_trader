"""Round 6 / ix_: pre-registered out-of-sample test of the exam-passing US500 RSI2 rule
on 10 new stock indices (D1, train+val only, never TEST).

  python results/ix_task1.py [--seeds 30]

Per market: train / val / train+val trades, avgR, t; fair coin-flip p (random entries with the
strategy's own exit, never entering where that exit fires at once; tools.factory.edge_p);
+50% spread avgR; swap per trade in R; and a zero-cost diagnostic (spread floor 0, no swap)
to show how much of the result is cost. US500Cash / US100Cash are shown as reference.
Writes results/ix_task1.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import factory as F                      # noqa: E402  (chdirs to ROOT)
from tools.research import apply_params             # noqa: E402

STRATEGY = "lm_meanrev"
PARAMS = {"mode": "rsi2", "rsi_th": 15.0, "exit_ma": 3, "exit_rsi": 70.0, "max_hold": 10,
          "side": "long", "trend": 0, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 20.0,
          "cfg.cooldown_bars": 0, "cfg.ema_trend": 200}
MAX_BARS = 6000
REF = ["US500Cash", "US100Cash"]
NEW = ["US30Cash", "US2000Cash", "GER40Cash", "UK100Cash", "JP225Cash", "EU50Cash",
       "FRA40Cash", "AUS200Cash", "HK50Cash", "CA60Cash"]


def one(job):
    market, seeds = job
    cfg = apply_params(F._base(market), STRATEGY, "D1", PARAMS)
    tr, va, tv = F._splits(market, "D1", MAX_BARS)
    mt, mv, mtv = F._run(cfg, tr), F._run(cfg, va, tr), F._run(cfg, tv)
    st = copy.deepcopy(cfg)
    st.backtest.spread_points = cfg.backtest.spread_points * 1.5
    spr = F._run(st, tv)["avg_r"]
    z = copy.deepcopy(cfg)
    z.backtest.spread_points = 0.0
    z.backtest.use_bar_spread = False
    z.backtest.swap_long_points = 0.0
    z.backtest.swap_short_points = 0.0
    zero = F._run(z, tv)
    _, rmean, rp95, p = F.edge_p(market, "D1", STRATEGY, PARAMS, tv, "long", 0, True, seeds=seeds)
    return {"market": market, "period": f"{tr.index[0]:%Y-%m}..{va.index[-1]:%Y-%m}",
            "tr_n": mt["trades"], "tr_avgR": mt["avg_r"], "tr_t": mt["t_stat"],
            "va_n": mv["trades"], "va_avgR": mv["avg_r"], "va_t": mv["t_stat"],
            "tv_n": mtv["trades"], "tv_avgR": mtv["avg_r"], "tv_t": mtv["t_stat"],
            "tv_pf": mtv["pf"], "tv_swapR": mtv["swapR"], "spread50_avgR": spr,
            "zero_cost_avgR": zero["avg_r"], "zero_cost_t": zero["t_stat"],
            "rand_mean": round(rmean, 3), "p_random": round(p, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(m, a.seeds) for m in REF + NEW]))
    out = ROOT / "results" / "ix_task1.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = f"{'market':<11}{'period':<17}{'tr n/avgR/t':>20}{'va n/avgR/t':>20}{'tv n/avgR/t':>20}{'swapR':>7}{'+50%':>7}{'0cost':>7}{'p_rnd':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['market']:<11}{r['period']:<17}"
              f"{r['tr_n']:>5} {r['tr_avgR']:+.3f} {r['tr_t']:+5.2f}"
              f"{r['va_n']:>6} {r['va_avgR']:+.3f} {r['va_t']:+5.2f}"
              f"{r['tv_n']:>6} {r['tv_avgR']:+.3f} {r['tv_t']:+5.2f}"
              f"{r['tv_swapR']:>+7.3f}{r['spread50_avgR']:>+7.3f}{r['zero_cost_avgR']:>+7.3f}{r['p_random']:>7.2f}")
    pos = [r["market"] for r in rows if r["market"] in NEW and r["tv_avgR"] > 0]
    print(f"new indices positive on train+val: {len(pos)}/10 {pos}")


if __name__ == "__main__":
    main()
