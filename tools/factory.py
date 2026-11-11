"""Strategy FACTORY: run one strategy family over many markets x timeframes x
settings, apply the automatic checks, and log EVERY test in a registry.

  python tools/factory.py --study grids/lib_trend_brk.json [--workers 3]

Study file (JSON):
  {
    "strategy": "lib_trend",                 # module in bot/strategies/
    "family": "trend",                       # label for the leaderboard
    "markets": ["GOLD", "BTCUSD"] | "all",
    "timeframes": ["H1", "H4", "D1"],
    "grid": {"lookback": [20, 55], "cfg.atr_sl_mult": [2, 3], ...},
    "dir": "long" | "short" | "both",        # random baseline side
    "trend": 0 | 1,                          # baseline uses the EMA filter
    "exit_from_strategy": false,             # baseline reuses the strategy's exit_signal
    "max_bars": 60000                        # most recent bars per timeframe
  }

Checks (automatic; the locked TEST split is never touched here):
  1 real costs      every run uses the market's research_<x>.yaml (spread + swap)
  2 practice/quiz   train avgR > 0, val avgR > 0, val t >= VAL_T
  3 coin flip       random-entry baseline (same session/stops/exits/costs),
                    p_random < 0.05 on train+val
  4 wider spread    train+val still positive at +50% spread
  5 nearby settings most grid neighbours positive on train AND val
  + luck bar        train+val t >= T_BAR (raised because thousands of
                    configurations are tested; see leaderboard.py)
A configuration that passes everything is a FINALIST for tools/final_exam.py.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from bot.backtest import load_econ, run_backtest      # noqa: E402
from bot.config import load_config                    # noqa: E402
from tools.research import apply_params, load_split, metrics  # noqa: E402

def _discover_markets() -> dict:
    """symbol -> research config, for every research_*.yaml in the project root."""
    import yaml
    out = {}
    for f in sorted(ROOT.glob("research_*.yaml")):
        sym = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("symbol")
        if sym:
            out[str(sym)] = f.name
    return out


MARKETS = _discover_markets()
REGISTRY = ROOT / "results" / "registry_v2"   # one CSV per study; v2 = fixed backtester (round 5+)
T_BAR = 3.0          # train+val t-stat required (multiple-testing bar)
VAL_T = 1.0          # validation must agree on its own
MIN_TRADES = 30      # train+val trades
SEEDS = 20           # random-baseline seeds per finalist check
MAX_COMBOS = 64      # per market x timeframe
GAUNTLET_TOP = 3     # best configs per market x timeframe sent to checks 3-5

FIELDS = ["ts", "study", "family", "strategy", "market", "tf", "params",
          "tr_n", "tr_avgR", "tr_t", "va_n", "va_avgR", "va_t", "va_pf",
          "tv_n", "tv_avgR", "tv_t", "tv_pf", "tv_posM", "tv_maxdd_pct", "tv_swapR",
          "neigh_ok", "spread_avgR", "rand_mean", "rand_p95", "p_random", "verdict"]


@lru_cache(maxsize=None)
def _base(market: str):
    return load_config(MARKETS[market])


@lru_cache(maxsize=None)
def _splits(market: str, tf: str, max_bars: int):
    tr, va, _te = load_split(market, tf, max_bars)
    return tr, va, pd.concat([tr, va])


CTX = 10000   # warm-up bars prepended to later splits (indicators only, no trades)


def _run(cfg, bars, ctx=None):
    start = None
    if ctx is not None and len(ctx):
        start, bars = bars.index[0], pd.concat([ctx.iloc[-CTX:], bars])
    res = run_backtest(cfg, bars, load_econ(cfg.symbol), verbose=False, start_at=start)
    m = metrics(res)
    rs = np.array([t["swap"] / t["risk_amount"] for t in res.trade_log if t.get("risk_amount")])
    m["swapR"] = round(float(rs.mean()), 3) if len(rs) else 0.0
    return m


def stage_a(job):
    """Train for every combo; validation only for train-positive ones."""
    strategy, market, tf, params, max_bars = job
    cfg = apply_params(_base(market), strategy, tf, params)
    tr, va, _ = _splits(market, tf, max_bars)
    try:
        mt = _run(cfg, tr)
        mv = _run(cfg, va, tr) if mt["avg_r"] > 0 and mt["trades"] >= 10 else None
    except Exception as e:  # a broken combo is logged, not fatal
        return {"market": market, "tf": tf, "params": params, "error": repr(e)}
    return {"market": market, "tf": tf, "params": params, "tr": mt, "va": mv}


def edge_p(market, tf, strategy, params, bars, side, trend, exit_from, seeds=SEEDS):
    base = _base(market)
    cfg = apply_params(base, strategy, tf, params)
    cand = run_backtest(cfg, bars, load_econ(base.symbol), verbose=False)
    from tools.research import r_stats
    c_r, _ = r_stats(cand)
    n = cand.trades
    rparams = {k: v for k, v in params.items() if k.startswith(("cfg.", "session.", "backtest."))}
    rcfg = apply_params(base, "rnd_baseline", tf, rparams)
    extra = {"exit_from": strategy, "exit_params": {k: v for k, v in params.items() if "." not in k}} if exit_from else {}
    prob = max(n / max(len(bars), 1), 1e-4)
    for _ in range(4):   # calibrate firing rate to ~same trade count
        rcfg.strategy.params = {"prob": prob, "seed": 0, "dir": side, "trend": trend, **extra}
        m = run_backtest(rcfg, bars, load_econ(base.symbol), verbose=False).trades
        prob = prob * 3 if m == 0 else min(prob * n / m, 1.0)
    rs = []
    for s in range(1, seeds + 1):
        rcfg.strategy.params = {"prob": prob, "seed": s, "dir": side, "trend": trend, **extra}
        r, _ = r_stats(run_backtest(rcfg, bars, load_econ(base.symbol), verbose=False))
        rs.append(r)
    rs = np.array(rs)
    return c_r, float(rs.mean()), float(np.percentile(rs, 95)), float((rs >= c_r).mean())


def stage_b(job):
    """Checks 2-5 + luck bar for one shortlisted configuration."""
    strategy, market, tf, params, max_bars, side, trend, exit_from, neigh_ok = job
    base = _base(market)
    _tr, _va, tv = _splits(market, tf, max_bars)
    cfg = apply_params(base, strategy, tf, params)
    m = _run(cfg, tv)
    out = {"tv": m, "neigh_ok": neigh_ok}
    stress = copy.deepcopy(cfg)
    stress.backtest.spread_points = cfg.backtest.spread_points * 1.5
    out["spread_avgR"] = _run(stress, tv)["avg_r"]
    if m["t_stat"] >= T_BAR * 0.8 and m["trades"] >= MIN_TRADES:   # only pay for the null when close
        _, rmean, rp95, p = edge_p(market, tf, strategy, params, tv, side, trend, exit_from)
        out.update(rand_mean=round(rmean, 3), rand_p95=round(rp95, 3), p_random=round(p, 3))
    return market, tf, params, out


def neighbours(params: dict, grid: dict) -> list[dict]:
    out = []
    for k, vals in grid.items():
        if len(vals) < 2 or params[k] not in vals:
            continue
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                q = dict(params)
                q[k] = vals[j]
                out.append(q)
    return out


def key(p: dict) -> str:
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--markets", help="override study markets (comma list)")
    ap.add_argument("--registry", default=None, help="registry folder (default results/registry)")
    args = ap.parse_args()
    global REGISTRY
    if args.registry:
        REGISTRY = ROOT / args.registry
    st = json.loads(Path(args.study).read_text(encoding="utf-8-sig"))
    strategy, grid = st["strategy"], st.get("grid", {})
    markets = list(MARKETS) if st.get("markets", "all") == "all" else st["markets"]
    if args.markets:
        markets = args.markets.split(",")
    markets = [m for m in markets if Path(MARKETS[m]).exists()]
    tfs = st["timeframes"]
    max_bars = int(st.get("max_bars", 60000))
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())] or [{}]
    if len(combos) > MAX_COMBOS:
        raise SystemExit(f"{len(combos)} combos > {MAX_COMBOS}: shrink the grid")
    jobs = [(strategy, mk, tf, c, max_bars) for mk in markets for tf in tfs for c in combos
            if Path(f"data/{mk.lower()}_{tf.lower()}.csv").exists()]
    print(f"[factory] {strategy}: {len(combos)} settings x {len(markets)} markets x {len(tfs)} TFs "
          f"= {len(jobs)} configurations | workers {args.workers}")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res_a = list(ex.map(stage_a, jobs, chunksize=2))
    print(f"[factory] stage A done in {time.time() - t0:.0f}s")

    # index results per market/tf for neighbour checks
    by_mt: dict = {}
    for r in res_a:
        by_mt.setdefault((r["market"], r["tf"]), {})[key(r["params"])] = r

    def pos(r):
        return r and "error" not in r and r["va"] is not None and r["tr"]["avg_r"] > 0 and r["va"]["avg_r"] > 0

    shortlist = []
    for (mk, tf), rows in by_mt.items():
        cands = [r for r in rows.values() if pos(r) and r["va"]["t_stat"] >= VAL_T
                 and r["tr"]["trades"] + r["va"]["trades"] >= MIN_TRADES]
        cands.sort(key=lambda r: r["tr"]["t_stat"] + r["va"]["t_stat"], reverse=True)
        for r in cands[:GAUNTLET_TOP]:
            nb = [rows.get(key(q)) for q in neighbours(r["params"], grid)]
            nb = [x for x in nb if x is not None]
            ok = sum(1 for x in nb if pos(x))
            neigh_ok = f"{ok}/{len(nb)}"
            shortlist.append((strategy, mk, tf, r["params"], max_bars, st.get("dir", "long"),
                              int(st.get("trend", 0)), bool(st.get("exit_from_strategy", False)), neigh_ok))
    print(f"[factory] {len(shortlist)} configurations shortlisted for checks 3-5")
    with ProcessPoolExecutor(args.workers) as ex:
        res_b = list(ex.map(stage_b, shortlist))
    b_idx = {(mk, tf, key(p)): o for mk, tf, p, o in res_b}

    REGISTRY.mkdir(parents=True, exist_ok=True)
    reg_file = REGISTRY / (Path(args.study).stem + ".csv")
    new = not reg_file.exists()
    finalists = []
    with reg_file.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        stamp = time.strftime("%Y-%m-%d %H:%M")
        for r in res_a:
            row = {"ts": stamp, "study": Path(args.study).name, "family": st.get("family", ""),
                   "strategy": strategy, "market": r["market"], "tf": r["tf"], "params": key(r["params"])}
            if "error" in r:
                row["verdict"] = "error: " + r["error"][:80]
                w.writerow(row)
                continue
            row.update(tr_n=r["tr"]["trades"], tr_avgR=r["tr"]["avg_r"], tr_t=r["tr"]["t_stat"])
            if r["va"]:
                row.update(va_n=r["va"]["trades"], va_avgR=r["va"]["avg_r"], va_t=r["va"]["t_stat"], va_pf=r["va"]["pf"])
            b = b_idx.get((r["market"], r["tf"], key(r["params"])))
            verdict = "fail: practice" if r["tr"]["avg_r"] <= 0 else (
                "fail: quiz" if not pos(r) or r["va"]["t_stat"] < VAL_T else "fail: not shortlisted")
            if b:
                tv = b["tv"]
                row.update(tv_n=tv["trades"], tv_avgR=tv["avg_r"], tv_t=tv["t_stat"], tv_pf=tv["pf"],
                           tv_posM=tv["pos_months_pct"], tv_maxdd_pct=round(tv["max_dd"] / 1000, 1),
                           tv_swapR=tv["swapR"], neigh_ok=b["neigh_ok"], spread_avgR=b["spread_avgR"],
                           rand_mean=b.get("rand_mean", ""), rand_p95=b.get("rand_p95", ""),
                           p_random=b.get("p_random", ""))
                ok_n, tot_n = (int(x) for x in b["neigh_ok"].split("/"))
                checks = [
                    ("luck bar", tv["t_stat"] >= T_BAR),
                    ("coin flip", b.get("p_random", 1.0) < 0.05),
                    ("wider spread", b["spread_avgR"] > 0),
                    ("nearby settings", tot_n == 0 or ok_n / tot_n >= 0.6),
                ]
                failed = [n for n, ok in checks if not ok]
                verdict = "FINALIST" if not failed else "fail: " + ", ".join(failed)
                if not failed:
                    finalists.append(row)
            row["verdict"] = verdict
            w.writerow(row)
    print(f"[factory] logged {len(res_a)} configurations -> {reg_file}")
    for (mk, tf, p), o in sorted(b_idx.items(), key=lambda kv: -kv[1]["tv"]["t_stat"])[:10]:
        tv = o["tv"]
        print(f"  {mk:<10}{tf:<4} t {tv['t_stat']:+5.2f} avgR {tv['avg_r']:+.3f} n {tv['trades']:4d} "
              f"neigh {o['neigh_ok']:<5} +50%spr {o['spread_avgR']:+.3f} p_rand {o.get('p_random', '-')}  {p[:70]}")
    print(f"[factory] FINALISTS: {len(finalists)}")
    for f in finalists:
        print(f"  {f['market']} {f['tf']} {f['params']}  t {f['tv_t']} avgR {f['tv_avgR']}")


if __name__ == "__main__":
    main()
