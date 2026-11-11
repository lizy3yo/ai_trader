"""BREADTH test (2026-11-09): do the dip-buying rules that passed on indices
also work across hundreds of individual US stocks?

PRE-REGISTERED (written before any stock result was seen):
  universe  every exported US stock CFD (data/stocks/) with history from
            2014 or earlier, last close >= $10 and estimated spread <= 0.20%
            of price. These are TODAY's listed stocks, so dead companies are
            missing (survivorship bias); the random-entry baseline on the
            SAME stocks absorbs that bias: the rule must beat it.
  rules     frozen, no tuning on stocks:
            A rsi2      = config_us500_rsi2.yaml exactly (the primary test)
            B rsi2_tr   = A + only above the 200 EMA (textbook Connors version)
            C pullback  = config_jp225_pullback.yaml exactly
  costs     spread = 1.3 x max(live, recent recorded) spread, scaled to price
            back in time; financing = the broker's annual long/short rate
            (swap mode 5) charged per night, Friday x3.
  pass      pooled avg R > 0 in each of 3 periods (2013-18, 2019-22, 2023-26),
            beats every random-entry seed (same stocks, exits, costs), and a
            K-position portfolio has monthly t >= 3.

Usage: python tools/universe_test.py [--workers 6] [--limit 0]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.backtest import load_csv_bars, run_backtest   # noqa: E402
from bot.risk import SymbolEconomics                              # noqa: E402
from bot.config import load_config                                     # noqa: E402

DIR = Path("data/stocks")
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("results/universe")
RULES = {
    "rsi2": ("lm_meanrev", {"mode": "rsi2", "rsi_th": 15.0, "exit_ma": 3, "exit_rsi": 70.0,
                            "max_hold": 10, "side": "long", "trend": 0}, {}),
    "rsi2_tr": ("lm_meanrev", {"mode": "rsi2", "rsi_th": 15.0, "exit_ma": 3, "exit_rsi": 70.0,
                               "max_hold": 10, "side": "long", "trend": 1}, {}),
    "pullback": ("lt_pullback", {"trig": "rsi", "n": 3, "thr": 10, "side": "long",
                                 "exit": "rec", "exit_n": 5}, {"max_hold_bars": 20}),
}
SEEDS = 4
# sensitivity knobs (env): MAX_SPREAD = universe spread cap (fraction of price),
# COST_MULT = scale on spread AND financing (0 = gross edge before costs)
MAX_SPREAD = float(__import__("os").environ.get("MAX_SPREAD", "0.002"))
COST_MULT = float(__import__("os").environ.get("COST_MULT", "1.0"))
PERIODS = [("2013-18", "2013-01-01", "2019-01-01"), ("2019-22", "2019-01-01", "2023-01-01"),
           ("2023-26", "2023-01-01", "2027-01-01")]


def universe() -> list[str]:
    keep = []
    for f in sorted(DIR.glob("*_econ.json")):
        e = json.loads(f.read_text(encoding="utf-8"))
        sym = e["symbol"]
        csv = DIR / f"{sym.lower()}_d1.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, usecols=["time", "close", "spread"])
        if pd.Timestamp(df.time.iloc[0]) > pd.Timestamp("2014-12-31"):
            continue
        last = float(df.close.iloc[-1])
        if last < 10:
            continue
        if est_spread(e, df) / last > MAX_SPREAD:
            continue
        keep.append(sym)
    return keep


def est_spread(e: dict, df: pd.DataFrame) -> float:
    rec = float(df.spread.tail(250).median()) * e["point"]
    live = e.get("live_spread")
    live = float(live) if live is not None and np.isfinite(live) else 0.0
    return 1.3 * max(rec, live) * COST_MULT


def make_cfg(sym: str, e: dict, df: pd.DataFrame, strat: str, params: dict, extra: dict):
    cfg = load_config("config_us500_rsi2.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.symbol = sym
    cfg.capital.allocated = cfg.capital.hard_max = 100000.0
    cfg.risk.base_risk_pct = cfg.risk.max_risk_pct = 1.0
    cfg.risk.step_up_pct = 0.0
    cfg.risk.daily_loss_limit = 1e9
    cfg.risk.max_spread_points = 1e9
    s = cfg.strategy
    s.name, s.timeframe, s.params = strat, "D1", dict(params)
    s.ema_trend, s.atr_period, s.atr_sl_mult, s.atr_tp_mult, s.cooldown_bars = 200, 14, 3.0, 20.0, 0
    s.max_hold_bars = int(extra.get("max_hold_bars", 0))
    ref = float(df.close.iloc[-1])
    b = cfg.backtest
    b.use_bar_spread = COST_MULT > 0
    b.spread_points = est_spread(e, df) / e["point"]
    b.spread_ref_price = ref
    b.commission_per_lot = 0.0
    if e["swap_mode"] == 5:      # annual % of the current price
        b.swap_long_points = COST_MULT * ref * e["swap_long_points"] / 100 / 360 / e["point"]
        b.swap_short_points = COST_MULT * ref * e["swap_short_points"] / 100 / 360 / e["point"]
        b.swap_ref_price = ref
    else:
        b.swap_long_points, b.swap_short_points, b.swap_ref_price = e["swap_long_points"], e["swap_short_points"], 0.0
    r3 = int(e.get("swap_rollover3days_mt5", 5))
    b.swap_triple_weekday = (r3 - 1) % 7 if r3 < 7 else 7   # MT5 Sun=0 -> Python Mon=0
    b.swap_every_day = False
    return cfg


def econ_of(e: dict) -> SymbolEconomics:
    return SymbolEconomics(tick_size=e["tick_size"], tick_value=e["tick_value"],
                           volume_min=e["volume_min"], volume_max=e["volume_max"],
                           volume_step=e["volume_step"], point=e["point"])


def log_of(res, sym, rule, seed=-1) -> list[dict]:
    return [{"symbol": sym, "rule": rule, "seed": seed, "entry": t["open_time"], "exit": t["close_time"],
             "R": t["pnl"] / t["risk_amount"], "swapR": t["swap"] / t["risk_amount"]}
            for t in res.trade_log if t.get("risk_amount")]


def job(sym: str) -> list[dict]:
    e = json.loads((DIR / f"{sym.lower()}_econ.json").read_text(encoding="utf-8"))
    bars = load_csv_bars(str(DIR / f"{sym.lower()}_d1.csv"))
    df = pd.read_csv(DIR / f"{sym.lower()}_d1.csv", usecols=["close", "spread"])
    econ = econ_of(e)
    rows = []
    for rule, (strat, params, extra) in RULES.items():
        cfg = make_cfg(sym, e, df, strat, params, extra)
        res = run_backtest(cfg, bars, econ, verbose=False)
        rows += log_of(res, sym, rule)
        n = res.trades
        if n == 0:
            continue
        # random-entry twin: same stock, same exit rule, same costs, ~same trade count
        rp = {"dir": "long", "trend": int(params.get("trend", 1 if strat == "lt_pullback" else 0)),
              "exit_from": strat, "exit_params": params}
        rcfg = make_cfg(sym, e, df, "rnd_baseline", {}, extra)
        prob = max(n / len(bars), 1e-3)
        for _ in range(3):
            rcfg.strategy.params = {"prob": prob, "seed": 0, **rp}
            m = run_backtest(rcfg, bars, econ, verbose=False).trades
            prob = prob * 3 if m == 0 else min(prob * n / m, 1.0)
        for sd in range(1, SEEDS + 1):
            rcfg.strategy.params = {"prob": prob, "seed": sd, **rp}
            rows += log_of(run_backtest(rcfg, bars, econ, verbose=False), sym, rule, sd)
    return rows


def portfolio(t: pd.DataFrame, k: int) -> pd.Series:
    """Take trades in entry order (ties: by symbol) while fewer than k are open;
    each risks 1 unit (1R). Returns R per calendar month (by exit)."""
    t = t.sort_values(["entry", "symbol"])
    open_exits: list[pd.Timestamp] = []
    took = []
    for row in t.itertuples():
        open_exits = [x for x in open_exits if x > row.entry]
        if len(open_exits) < k:
            open_exits.append(row.exit)
            took.append(row.Index)
    s = t.loc[took]
    return s.groupby(s.exit.dt.to_period("M")).R.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/universe")
    a = ap.parse_args()
    syms = universe()
    if a.limit:
        syms = syms[: a.limit]
    print(f"universe: {len(syms)} stocks", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, syms), 1):
            rows += r
            if i % 25 == 0:
                print(f"  {i}/{len(syms)}", flush=True)
    t = pd.DataFrame(rows)
    t["entry"], t["exit"] = pd.to_datetime(t.entry), pd.to_datetime(t.exit)
    t.to_csv(OUT / "trades.csv", index=False)
    report(t, len(syms))


def report(t: pd.DataFrame, n_syms: int) -> None:
    lines = [f"BREADTH TEST on {n_syms} US stocks (pre-registered rules, real costs)\n"]
    for rule in RULES:
        real = t[(t.rule == rule) & (t.seed == -1)]
        rnd = t[(t.rule == rule) & (t.seed > 0)]
        if real.empty:
            continue
        r = real.R
        lines.append(f"== {rule}: {len(r)} trades, win {100*(r>0).mean():.0f}%, avg {r.mean():+.3f}R "
                     f"(financing {real.swapR.mean():+.3f}R)")
        seeds = rnd.groupby("seed").R.mean()
        lines.append(f"   random entries, same exits: " + ", ".join(f"{v:+.3f}" for v in seeds)
                     + f"  -> rule beats {int((seeds < r.mean()).sum())}/{len(seeds)} seeds")
        per = []
        for name, a, b in PERIODS:
            rr = real[(real.entry >= a) & (real.entry < b)].R
            nn = rnd[(rnd.entry >= a) & (rnd.entry < b)].R
            per.append(f"{name}: {rr.mean():+.3f}R n={len(rr)} (random {nn.mean():+.3f})")
        lines.append("   " + " | ".join(per))
        for k in (5, 10, 20):
            m = portfolio(real, k)
            idx = pd.period_range(m.index.min(), m.index.max(), freq="M")
            m = m.reindex(idx, fill_value=0.0)
            eq = m.cumsum()
            dd = float((eq.cummax() - eq).max())
            tstat = m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if m.std() > 0 else 0
            mr = portfolio(rnd[rnd.seed == 1], k).reindex(idx, fill_value=0.0)
            lines.append(f"   portfolio max {k:>2} open: {m.mean():+.2f}R/month, t {tstat:.1f}, "
                         f"months up {100*(m>0).mean():.0f}%, worst month {m.min():+.1f}R, "
                         f"max drawdown {dd:.1f}R   (random: {mr.mean():+.2f}R/month)")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
