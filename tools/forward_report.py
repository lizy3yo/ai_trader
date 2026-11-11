"""FORWARD-TEST TRACKER: is each strategy behaving live (demo) like its backtest?

New, never-seen trades are the only proof that can't be over-fitted. This
compares every live/demo trade in a config's journal with what the backtest
expects, in R (profit / amount risked), so account size doesn't matter.

  python tools/forward_report.py --build      # once: save backtest expectations
  python tools/forward_report.py              # any time: live vs expected

Per strategy it shows:
  trades so far vs expected by now (too few = bot not running / not sizing)
  live avg R vs backtest avg R, and how far apart in standard errors
  live drawdown (R) vs the 95% worst drawdown the backtest would produce
  STATUS: TOO EARLY (< 15 trades) / ON TRACK / WATCH / STOP
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from bot.backtest import load_csv_bars, load_econ, run_backtest  # noqa: E402
from bot.config import load_config                              # noqa: E402

PORTFOLIO = ["config_us500_rsi2.yaml", "config_jp225_pullback.yaml", "config.yaml"]
EXPECT = ROOT / "results" / "forward_expectations.json"
MIN_TRADES = 15


def build() -> None:
    from tools.factory import MARKETS
    out = {}
    for path in PORTFOLIO:
        tc = load_config(path)
        base = load_config(MARKETS[tc.symbol])
        cfg = base
        cfg.strategy, cfg.session, cfg.symbol = tc.strategy, tc.session, tc.symbol
        bars = load_csv_bars(tc.backtest.data_csv)
        if tc.symbol == "GOLD":
            bars = bars[bars.index >= "2013-06-01"]   # real hourly history starts here
        res = run_backtest(cfg, bars, load_econ(tc.symbol), verbose=False)
        tl = pd.DataFrame(res.trade_log)
        r = (tl.pnl / tl.risk_amount).to_numpy()
        months = (bars.index[-1] - bars.index[0]).days / 30.44
        rng = np.random.default_rng(1)
        dds = []
        for _ in range(2000):                      # bootstrap drawdowns over 50 trades
            eq = np.cumsum(rng.choice(r, 50))
            dds.append(float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()))
        out[path] = {"symbol": tc.symbol, "strategy": tc.strategy.name, "db": tc.tracking.db_path,
                     "trades_per_month": round(len(r) / months, 2), "avg_r": round(float(r.mean()), 4),
                     "sd_r": round(float(r.std(ddof=1)), 4), "win_rate": round(float((r > 0).mean()), 3),
                     "dd95_50trades": round(float(np.percentile(dds, 95)), 2),
                     "built": datetime.now().strftime("%Y-%m-%d"), "n_backtest": int(len(r))}
        print(f"{path:<28} {tc.symbol:<10} {len(r)} trades  {out[path]['trades_per_month']}/mo  "
              f"avgR {out[path]['avg_r']:+.3f}  win {out[path]['win_rate']:.0%}  dd95 {out[path]['dd95_50trades']}R")
    EXPECT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> {EXPECT}")


def live_trades(db: str) -> pd.DataFrame:
    if not Path(db).exists():
        return pd.DataFrame()
    with sqlite3.connect(db) as c:
        df = pd.read_sql("select * from trades where mode='live' and close_time is not null "
                         "and risk_amount > 0 order by close_time", c)
    if len(df):
        df["R"] = df.pnl / df.risk_amount
    return df


def report() -> None:
    if not EXPECT.exists():
        raise SystemExit("run with --build first")
    exp = json.loads(EXPECT.read_text(encoding="utf-8"))
    print(f"FORWARD TEST  {datetime.now():%Y-%m-%d %H:%M}   (R = profit / amount risked)\n")
    for path, e in exp.items():
        df = live_trades(e["db"])
        n = len(df)
        if n == 0:
            print(f"{path:<28} no closed live trades yet (expect ~{e['trades_per_month']}/month)\n")
            continue
        first = pd.to_datetime(df.open_time.iloc[0])
        months = max((datetime.now() - first.to_pydatetime()).days / 30.44, 0.5)
        r = df.R.to_numpy()
        se = e["sd_r"] / np.sqrt(n)
        z = (r.mean() - e["avg_r"]) / se if se > 0 else 0.0
        eq = np.cumsum(r)
        dd = float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max())
        if n < MIN_TRADES:
            status = "TOO EARLY"
        elif dd > e["dd95_50trades"] or z < -2.5:
            status = "STOP — worse than the backtest allows"
        elif z < -1.5:
            status = "WATCH — below expectations"
        else:
            status = "ON TRACK"
        print(f"{path:<28} {e['symbol']} {e['strategy']}")
        print(f"   trades {n} (expected ~{e['trades_per_month'] * months:.0f} by now)  "
              f"win {np.mean(r > 0):.0%} (bt {e['win_rate']:.0%})")
        print(f"   avg R {r.mean():+.3f} (bt {e['avg_r']:+.3f}, {z:+.1f} SE)  total {eq[-1]:+.1f}R  "
              f"drawdown {dd:.1f}R (bt 95% worst {e['dd95_50trades']}R)")
        print(f"   STATUS: {status}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    build() if a.build else report()
