"""Round 6 / ix_: multi-index portfolio of the UNCHANGED US500 RSI2 dip-buyer.
TRAIN+VAL splits only (each market's own 60%+20% of its last 6000 D1 bars); never TEST.

  python results/ix_portfolio.py

Rule (config_us500_rsi2.yaml, exact): lm_meanrev rsi2, RSI(2)<15 buy next open, exit close >
SMA3 or RSI>70 or 10 days, 3 ATR disaster stop, no trend filter. Real costs per research_<x>.yaml.

Per-trade R (pnl / risk, costs + swap included) is combined by CALENDAR month of the exit at a
fixed 1% risk per trade (monthly R = monthly % return, non-compounded). Calendar window:
2011-09 .. 2023-07 = last complete month before the earliest val end of all 12 markets
(HK50 2023-08-15), so every month uses the same market set (late starters US2000 2017-09,
AUS200 2014-06, CA60 2020-05 join when their data starts). Trades outside the window are
dropped (they are still train/val, never test).

Sets: ALL (US500 + US100 + 10 new), POS (US500 + US100 + the new indices positive on
train+val in task 1 -> IN-SAMPLE selection), US500 alone (reference).
Writes results/ix_portfolio.csv (per-market) and results/ix_portfolio_sets.csv.
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import factory as F                      # noqa: E402  (chdirs to ROOT)
from tools.research import apply_params             # noqa: E402
from bot.backtest import load_econ, run_backtest    # noqa: E402

STRATEGY = "lm_meanrev"
PARAMS = {"mode": "rsi2", "rsi_th": 15.0, "exit_ma": 3, "exit_rsi": 70.0, "max_hold": 10,
          "side": "long", "trend": 0, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 20.0,
          "cfg.cooldown_bars": 0, "cfg.ema_trend": 200}
MAX_BARS = 6000
REF = ["US500Cash", "US100Cash"]
NEW = ["US30Cash", "US2000Cash", "GER40Cash", "UK100Cash", "JP225Cash", "EU50Cash",
       "FRA40Cash", "AUS200Cash", "HK50Cash", "CA60Cash"]
W0 = pd.Period("2011-09", "M")
TARGET = 500.0     # $ per month
RISK = 0.01        # fixed risk per trade


def trades(market: str):
    cfg = apply_params(F._base(market), STRATEGY, "D1", PARAMS)
    _tr, va, tv = F._splits(market, "D1", MAX_BARS)
    res = run_backtest(cfg, tv, load_econ(cfg.symbol), verbose=False)
    d = pd.DataFrame(res.trade_log)
    d["R"] = d["pnl"] / d["risk_amount"]
    d["open"] = pd.to_datetime(d["open_time"])
    d["close"] = pd.to_datetime(d["close_time"])
    d["market"] = market
    return d, tv.index[0], va.index[-1]


def max_dd(x: np.ndarray) -> float:
    eq = np.concatenate([[0.0], np.cumsum(x)])
    return float((np.maximum.accumulate(eq) - eq).max())


def summarise(name, mk, T, months, active):
    t = T[T.market.isin(mk)].sort_values("close")
    M = pd.DataFrame(0.0, index=months, columns=mk)
    g = t.groupby([t.close.dt.to_period("M"), "market"]).R.sum()
    for (p, m), v in g.items():
        M.loc[p, m] = v
    for m in mk:                          # not yet trading -> NaN (not 0)
        M.loc[M.index < active[m], m] = np.nan
    port = M.sum(axis=1, min_count=1).fillna(0.0)
    n_m = len(months)
    avg_m = port.mean()
    # steady state: every market active every month
    steady = sum(M[m].mean() for m in mk)
    cors = []
    for a, b in itertools.combinations(mk, 2):
        x = M[[a, b]].dropna()
        if len(x) >= 24 and x[a].std() > 0 and x[b].std() > 0:
            cors.append(x[a].corr(x[b]))
    day = t.groupby(t.open.dt.normalize()).agg(n=("market", "nunique"), R=("R", "sum"))
    # max simultaneously open positions
    ev = pd.concat([pd.Series(1, index=t.open), pd.Series(-1, index=t.close)]).sort_index(kind="stable")
    conc = int(ev.groupby(level=0).sum().cumsum().max())
    rs = t.R.to_numpy()
    tstat = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))
    big = day[day.n >= 3]
    return {
        "set": name, "markets": len(mk), "months": n_m, "trades": len(t),
        "trades_per_month": round(len(t) / n_m, 2), "avgR_trade": round(rs.mean(), 3),
        "t_pooled": round(tstat, 2),
        "avgR_month": round(avg_m, 3), "steady_avgR_month": round(steady, 3),
        "pos_months_pct": round(100 * (port > 0).mean(), 1),
        "pos_months_pct_traded": round(100 * (port[port != 0] > 0).mean(), 1),
        "worst_month_R": round(port.min(), 2),
        "maxdd_R_trades": round(max_dd(rs), 2), "maxdd_R_monthly": round(max_dd(port.to_numpy()), 2),
        "avg_pair_corr": round(float(np.mean(cors)), 3) if cors else np.nan,
        "days_ge2": int((day.n >= 2).sum()), "days_ge3": int((day.n >= 3).sum()),
        "days_ge5": int((day.n >= 5).sum()), "max_same_day": int(day.n.max()),
        "pct_trades_on_ge3_days": round(100 * t.open.dt.normalize().isin(big.index).mean(), 1),
        "avgR_trade_ge3_days": round(t[t.open.dt.normalize().isin(big.index)].R.mean(), 3) if len(big) else np.nan,
        "worst_cluster_day_R": round(day.R.min(), 2), "max_concurrent": conc,
        "account_for_500": round(TARGET / (RISK * avg_m)) if avg_m > 0 else "never (avgR<=0)",
        "account_for_500_steady": round(TARGET / (RISK * steady)) if steady > 0 else "never (avgR<=0)",
    }


def main():
    allm = REF + NEW
    parts, active, vend = [], {}, {}
    for m in allm:
        d, start, end = trades(m)
        parts.append(d)
        active[m] = max(pd.Timestamp(start).to_period("M") + 1, W0)  # first full month
        vend[m] = pd.Timestamp(end)
    w1 = (min(vend.values()).to_period("M") - 1)
    months = pd.period_range(W0, w1, freq="M")
    T = pd.concat(parts, ignore_index=True)
    T = T[(T.close.dt.to_period("M") >= W0) & (T.close.dt.to_period("M") <= w1)
          & (T.open.dt.to_period("M") >= W0)]
    T = T[[T.close.dt.to_period("M").iat[i] >= active[T.market.iat[i]] for i in range(len(T))]]
    print(f"window {months[0]}..{months[-1]} ({len(months)} months); per-market val ends: "
          + ", ".join(f"{m[:-4]} {vend[m]:%Y-%m}" for m in allm))

    rows = []
    for m in allm:
        t = T[T.market == m]
        nm = len(pd.period_range(active[m], w1, freq="M"))
        rs = t.R.to_numpy()
        rows.append({"market": m, "from": str(active[m]), "months": nm, "trades": len(t),
                     "avgR": round(rs.mean(), 3), "t": round(rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs))), 2),
                     "R_per_month": round(rs.sum() / nm, 3), "maxdd_R": round(max_dd(rs), 2)})
    pm = pd.DataFrame(rows)
    print(pm.to_string(index=False))
    pm.to_csv(ROOT / "results" / "ix_portfolio.csv", index=False)

    task1 = ROOT / "results" / "ix_task1.csv"
    pos_new = [m for m in NEW if task1.exists() and
               float(pd.read_csv(task1).set_index("market").loc[m, "tv_avgR"]) > 0]
    sets = [("US500 only", ["US500Cash"]), ("US500+US100", REF),
            ("ALL 12", allm), ("10 new only", NEW),
            (f"POS (in-sample): US500+US100+{'+'.join(x[:-4] for x in pos_new)}", REF + pos_new)]
    out = [summarise(n, mk, T, months, active) for n, mk in sets]
    S = pd.DataFrame(out).set_index("set").T
    print(S.to_string())
    pd.DataFrame(out).to_csv(ROOT / "results" / "ix_portfolio_sets.csv", index=False)


if __name__ == "__main__":
    main()
