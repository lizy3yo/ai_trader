"""Train/val evaluation identical to tools/research.py splits, but each split is
preceded by prior history so long (daily) regime filters are formed at the
split start. Entries are blocked before the split's own warmup index via the
strategy param start_after (same start as research.py). TEST never touched.

python ext_eval.py --strategy gsw_donchian --tf H1 --max-bars 80000 --grid '<json>' --out results/gsw_x.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

ROOT = r"C:\00-SideProject\AI_Trading"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

from bot.backtest import load_csv_bars, load_econ, run_backtest  # noqa: E402
from bot.config import load_config                               # noqa: E402
from bot.strategies import get_strategy                          # noqa: E402
from tools.research import apply_params, metrics                 # noqa: E402

PRE = {"H1": 9000, "H4": 2600}
G = {}


def init(cfgpath, tf, max_bars, strategy):
    base = load_config(cfgpath)
    full = load_csv_bars(f"data/{base.symbol.lower()}_{tf.lower()}.csv")
    bars = full.iloc[-max_bars:] if len(full) > max_bars else full
    a, b = int(len(bars) * 0.6), int(len(bars) * 0.8)
    G.update(base=base, full=full, tr=bars.iloc[:a], va=bars.iloc[a:b], tf=tf,
             strategy=strategy, econ=load_econ(base.symbol))


def run_split(params, split):
    base, full, tf = G["base"], G["full"], G["tf"]
    sp = G[split]
    cfg = apply_params(base, G["strategy"], tf, params)
    w = get_strategy(G["strategy"]).warmup(cfg.strategy)
    start = sp.index[min(w, len(sp) - 1)]
    cfg.strategy.params["start_after"] = str(start)
    end_pos = full.index.get_loc(sp.index[-1])
    beg_pos = max(0, full.index.get_loc(sp.index[0]) - max(PRE[tf], w + 10))
    ext = full.iloc[beg_pos:end_pos + 1]
    res = run_backtest(cfg, ext, G["econ"], verbose=False)
    m = metrics(res)
    tl = pd.DataFrame(res.trade_log)
    if len(tl):
        tl["R"] = tl.pnl / tl.risk_amount
        tl["swapR"] = tl.swap / tl.risk_amount
        tl["year"] = pd.to_datetime(tl.open_time).dt.year
        tl["dir"] = tl.direction
        tl["hold_h"] = (pd.to_datetime(tl.close_time) - pd.to_datetime(tl.open_time)).dt.total_seconds() / 3600
    return m, tl


def job(params):
    out = {"params": params}
    tls = []
    for s in ("tr", "va"):
        m, tl = run_split(params, s)
        out["train" if s == "tr" else "val"] = m
        tls.append(tl)
    tl = pd.concat([t for t in tls if len(t)]) if any(len(t) for t in tls) else pd.DataFrame()
    if len(tl):
        n = len(tl)
        out["tv"] = {"n": n, "avgR": round(tl.R.mean(), 3),
                     "t": round(tl.R.mean() / (tl.R.std(ddof=1) / np.sqrt(n)), 2),
                     "swapR": round(tl.swapR.mean(), 3), "hold_h": round(tl.hold_h.median(), 0),
                     "years": {int(y): round(v, 1) for y, v in tl.groupby("year").R.sum().items()},
                     "short_n": int((tl.dir == "SHORT").sum()),
                     "short_avgR": round(tl[tl.dir == "SHORT"].R.mean(), 3) if (tl.dir == "SHORT").any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="gsw_donchian")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--max-bars", type=int, default=80000)
    ap.add_argument("--config", default="research_gold.yaml")
    ap.add_argument("--grid")
    ap.add_argument("--grid-file")
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--years", action="store_true")
    a = ap.parse_args()
    grid = json.loads(open(a.grid_file).read() if a.grid_file else a.grid)
    if isinstance(grid, list):
        combos = grid
    else:
        keys = list(grid)
        combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    print(f"{len(combos)} combos {a.tf} max_bars={a.max_bars}", flush=True)
    with Pool(min(a.procs, len(combos)), initializer=init,
              initargs=(a.config, a.tf, a.max_bars, a.strategy)) as pool:
        rows = pool.map(job, combos)
    for r in rows:
        tr, va, tv = r["train"], r["val"], r.get("tv", {})
        p = json.dumps({k: v for k, v in r["params"].items()}, separators=(",", ":"))
        print(f"{p}\n   TR n{tr['trades']:4d} PF{tr['pf']:5.2f} R{tr['avg_r']:+.3f} t{tr['t_stat']:+.2f} pM{tr['pos_months_pct']:3.0f} dd{tr['max_dd']:6.0f}"
              f" | VA n{va['trades']:4d} PF{va['pf']:5.2f} R{va['avg_r']:+.3f} t{va['t_stat']:+.2f} pM{va['pos_months_pct']:3.0f} dd{va['max_dd']:6.0f}"
              f" | TV t{tv.get('t', 0):+.2f} R{tv.get('avgR', 0):+.3f} swR{tv.get('swapR', 0):+.3f} hold{tv.get('hold_h', 0):.0f}h sh{tv.get('short_n', 0)}/{tv.get('short_avgR')}")
        if a.years:
            print("   years:", " ".join(f"{y}:{v:+.1f}" for y, v in tv.get("years", {}).items()))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
    main()
