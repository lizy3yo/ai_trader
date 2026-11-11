"""Diversified trend PORTFOLIO (analysis, not a factory study) — lx_ prefix.

One fixed, untuned rule (bot/strategies/lx_trend.py: D1 100-day breakout,
50-day opposite-channel exit, 3 x ATR(20) initial stop, long+short) on all 7
markets, TRAIN + VALIDATION splits only (the locked test split is never
loaded beyond load_split's return value, which is discarded).

Per leg: tools.research.load_split + bot.backtest.run_backtest with the
market's research_<x>.yaml (real spread + swap, 1% fixed risk -> trades in R).
The validation run gets the last 10k train bars as indicator warm-up only,
exactly like tools/factory.py.

Portfolio: every trade risks 1R in its own leg (equal risk per leg). Monthly
portfolio return = mean over the legs ACTIVE that month (inside that leg's own
train+val span) of the leg's summed R for trades closed that month, so the
result reads "R per month per unit of risk budget". Reported: pooled avgR/t
per trade, monthly mean/t/%positive/maxDD, Sharpe, leg correlations.

Random baseline: per leg, rnd_baseline with identical stops, the same trailing
exit (exit_from=lx_trend), random side, firing rate calibrated to the leg's
trade count, SEEDS seeds; the portfolio is rebuilt from the random legs for
every seed -> p_random = share of seeds whose pooled avgR >= the real one.

  python results/lx_portfolio.py [--seeds 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402
os.chdir(ROOT)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from bot.backtest import load_econ, run_backtest      # noqa: E402
from bot.config import load_config                    # noqa: E402
from tools.research import apply_params, load_split   # noqa: E402

MARKETS = {
    "GOLD": "research_gold.yaml", "SILVER": "research_silver.yaml",
    "US500Cash": "research_us500.yaml", "US100Cash": "research_us100.yaml",
    "EURUSD": "research_eurusd.yaml", "BTCUSD": "research_btc.yaml",
    "ETHUSD": "research_eth.yaml",
}
TF = "D1"
PARAMS = {"entry": 100, "exit": 50, "side": "both", "cfg.atr_period": 20,
          "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 100.0, "cfg.cooldown_bars": 0}
CTX = 10000


def trades(cfg, tr, va, mtm=False):
    """R per trade with close month, train run + val run (warm-up ctx).
    mtm=True also returns per-month mark-to-market R (dict month -> R)."""
    econ = load_econ(cfg.symbol)
    per_unit = econ.money_per_price_unit_per_lot()
    out, monthly = [], {}
    r1 = run_backtest(cfg, tr, econ, verbose=False)
    bars = pd.concat([tr.iloc[-CTX:], va])
    r2 = run_backtest(cfg, bars, econ, verbose=False, start_at=va.index[0])
    for split, res, px in (("train", r1, tr["close"]), ("val", r2, bars["close"])):
        for t in res.trade_log:
            if not t.get("risk_amount"):
                continue
            R = t["pnl"] / t["risk_amount"]
            out.append({"split": split, "close": pd.Timestamp(t["close_time"]), "R": R,
                        "dir": t["direction"], "swapR": t["swap"] / t["risk_amount"]})
            if mtm:
                # open-position P&L marked at each month-end close; the residual
                # (fill prices, spread, swap) lands in the exit month so the sum
                # over months equals the trade's R exactly
                d = 1.0 if t["direction"] == "LONG" else -1.0
                r_unit = t["risk_amount"] / (t["lots"] * per_unit)       # price move = 1R
                o, c = pd.Timestamp(t["open_time"]), pd.Timestamp(t["close_time"])
                seg = px[(px.index >= o) & (px.index <= c)]
                me = seg.groupby(seg.index.to_period("M")).last()
                start = float(px[px.index <= o].iloc[-1])
                acc = 0.0
                for mo, v in list(me.items())[:-1]:
                    x = d * (float(v) - start) / r_unit
                    monthly[mo] = monthly.get(mo, 0.0) + x
                    acc += x
                    start = float(v)
                last_mo = c.to_period("M")
                monthly[last_mo] = monthly.get(last_mo, 0.0) + (R - acc)
    df = pd.DataFrame(out)
    return (df, monthly) if mtm else df


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def monthly_matrix(mtm: dict, spans: dict) -> pd.DataFrame:
    """months x legs of mark-to-market R; NaN outside a leg's train+val span."""
    months = pd.period_range(min(s[0] for s in spans.values()).to_period("M"),
                             max(s[1] for s in spans.values()).to_period("M"), freq="M")
    m = pd.DataFrame(np.nan, index=months, columns=list(mtm))
    for mk, dct in mtm.items():
        a, b = spans[mk]
        act = (months >= a.to_period("M")) & (months <= b.to_period("M"))
        m.loc[act, mk] = 0.0
        for mo, v in dct.items():
            if mo in m.index and act[m.index.get_loc(mo)]:
                m.loc[mo, mk] += v
    return m


def port_stats(m: pd.DataFrame, min_legs: int = 1) -> dict:
    n_act = m.notna().sum(axis=1)
    port = m.mean(axis=1, skipna=True)[n_act >= min_legs]
    if not len(port):
        return {}
    cum = port.cumsum()
    dd = float((cum.cummax() - cum).max())
    return {"months": int(len(port)), "from": str(port.index[0]), "to": str(port.index[-1]),
            "mean_R_per_month": round(float(port.mean()), 4), "t_monthly": round(tstat(port), 2),
            "pos_months_pct": round(float((port > 0).mean() * 100), 1),
            "pos_of_nonflat_pct": round(float((port[port != 0] > 0).mean() * 100), 1),
            "max_dd_R": round(dd, 2),
            "sharpe_ann": round(float(port.mean() / port.std(ddof=1) * np.sqrt(12)), 2) if port.std(ddof=1) > 0 else 0.0,
            "avg_legs_active": round(float(n_act[n_act >= min_legs].mean()), 2)}


def random_legs(base, mk, tr, va, n_target, seeds):
    rparams = {k: v for k, v in PARAMS.items() if k.startswith("cfg.")}
    rcfg = apply_params(base, "rnd_baseline", TF, rparams)
    extra = {"exit_from": "lx_trend", "exit_params": {k: v for k, v in PARAMS.items() if "." not in k}}
    nbars = len(tr) + len(va)
    prob = max(n_target / max(nbars, 1), 1e-4)
    for _ in range(4):
        rcfg.strategy.params = {"prob": prob, "seed": 0, "dir": "both", "trend": 0, **extra}
        n = len(trades(rcfg, tr, va))
        prob = prob * 3 if n == 0 else min(prob * n_target / n, 1.0)
    out = []
    for s in range(1, seeds + 1):
        rcfg.strategy.params = {"prob": prob, "seed": s, "dir": "both", "trend": 0, **extra}
        out.append(trades(rcfg, tr, va, mtm=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    legs, mtms, spans, rows, rnd = {}, {}, {}, [], {}
    for mk, cf in MARKETS.items():
        base = load_config(cf)
        cfg = apply_params(base, "lx_trend", TF, PARAMS)
        tr, va, _locked = load_split(base.symbol, TF, 60000)
        del _locked                      # never evaluated
        df, mtms[mk] = trades(cfg, tr, va, mtm=True)
        legs[mk] = df
        spans[mk] = (tr.index[0] + pd.Timedelta(days=150), va.index[-1])   # ~warm-up excluded
        r = df["R"].to_numpy() if len(df) else np.array([])
        trd, vad = df[df.split == "train"], df[df.split == "val"]
        rows.append({"market": mk, "span": f"{tr.index[0].date()}..{va.index[-1].date()}",
                     "n": len(df), "avgR": round(float(r.mean()), 3) if len(r) else 0, "t": round(tstat(r), 2),
                     "pf": round(float(r[r > 0].sum() / -r[r < 0].sum()), 2) if (r < 0).any() else 0,
                     "train_n": len(trd), "train_avgR": round(float(trd.R.mean()), 3) if len(trd) else 0,
                     "train_t": round(tstat(trd.R), 2),
                     "val_n": len(vad), "val_avgR": round(float(vad.R.mean()), 3) if len(vad) else 0,
                     "val_t": round(tstat(vad.R), 2),
                     "long_avgR": round(float(df[df.dir == "LONG"].R.mean()), 3) if len(df) else 0,
                     "short_avgR": round(float(df[df.dir == "SHORT"].R.mean()), 3) if len(df) else 0,
                     "swapR": round(float(df.swapR.mean()), 3) if len(df) else 0})
        if args.seeds:
            rnd[mk] = random_legs(base, mk, tr, va, len(df), args.seeds)
            rr = np.array([x[0]["R"].mean() if len(x[0]) else 0.0 for x in rnd[mk]])
            rows[-1]["rand_mean"] = round(float(rr.mean()), 3)
            rows[-1]["rand_p95"] = round(float(np.percentile(rr, 95)), 3)
            rows[-1]["p_random"] = round(float((rr >= rows[-1]["avgR"]).mean()), 3)
        print(rows[-1], flush=True)

    pooled = pd.concat([d.assign(market=k) for k, d in legs.items()])
    m = monthly_matrix(mtms, spans)
    res = {
        "rule": PARAMS, "timeframe": TF, "legs": rows,
        "pooled": {"n": len(pooled), "avgR": round(float(pooled.R.mean()), 3), "t": round(tstat(pooled.R), 2),
                   "train_avgR": round(float(pooled[pooled.split == "train"].R.mean()), 3),
                   "train_t": round(tstat(pooled[pooled.split == "train"].R), 2),
                   "val_avgR": round(float(pooled[pooled.split == "val"].R.mean()), 3),
                   "val_t": round(tstat(pooled[pooled.split == "val"].R), 2)},
        "pooled_ex_eurusd": {"n": int((pooled.market != "EURUSD").sum()),
                             "avgR": round(float(pooled[pooled.market != "EURUSD"].R.mean()), 3),
                             "t": round(tstat(pooled[pooled.market != "EURUSD"].R), 2)},
        "portfolio_all_months": port_stats(m, 1),
        "portfolio_ge4_legs": port_stats(m, 4),
        "corr_monthly": m.corr(min_periods=24).round(2).to_dict(),
    }
    if args.seeds:
        rp, rm, rm4 = [], [], []
        for s in range(args.seeds):
            pr = pd.concat([rnd[k][s][0] for k in legs if len(rnd[k][s][0])])
            rp.append(float(pr.R.mean()))
            mm = monthly_matrix({k: rnd[k][s][1] for k in legs}, spans)
            rm.append(port_stats(mm, 1)["mean_R_per_month"])
            rm4.append(port_stats(mm, 4)["mean_R_per_month"])
        rp, rm, rm4 = np.array(rp), np.array(rm), np.array(rm4)
        res["random_portfolio"] = {
            "pooled_avgR_mean": round(float(rp.mean()), 3), "pooled_avgR_p95": round(float(np.percentile(rp, 95)), 3),
            "p_random_pooled": round(float((rp >= res["pooled"]["avgR"]).mean()), 3),
            "monthly_mean_rand": round(float(rm.mean()), 4),
            "p_random_monthly": round(float((rm >= res["portfolio_all_months"]["mean_R_per_month"]).mean()), 3),
            "monthly_mean_rand_ge4": round(float(rm4.mean()), 4),
            "p_random_monthly_ge4": round(float((rm4 >= res["portfolio_ge4_legs"]["mean_R_per_month"]).mean()), 3)}
    out = ROOT / "results" / "lx_portfolio.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("legs",)}, indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
