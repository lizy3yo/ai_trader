"""Expectancy by trading style: win rate x reward:risk, measured on real trades.

For every configuration in results/registry_v2 that made money on BOTH practice
(train) and quiz (val), re-run train+val with real costs and record:
  win rate, average win (R), average loss (R), realised reward:risk
  (avg win / avg loss), expectancy (avg R per trade), median holding time,
  trades per month, t-stat.
Style is decided by the REAL median holding time:
  scalp < 2 hours <= day trade < 24 hours <= swing.

  python tools/style_analysis.py   ->  results/style_analysis.csv + printed summary
"""
from __future__ import annotations

import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from bot.backtest import load_econ, run_backtest      # noqa: E402
from tools.factory import _base, _splits              # noqa: E402
from tools.research import apply_params               # noqa: E402


def one(row):
    try:
        study = json.loads((ROOT / "grids" / row["study"]).read_text(encoding="utf-8-sig"))
        mb = int(study.get("max_bars", 60000))
    except Exception:
        study, mb = {}, 60000
    try:
        params = json.loads(row["params"])
        _, _, tv = _splits(row["market"], row["tf"], mb)
        cfg = apply_params(_base(row["market"]), row["strategy"], row["tf"], params)
        tl = pd.DataFrame(run_backtest(cfg, tv, load_econ(row["market"]), verbose=False).trade_log)
        if len(tl) < 10:
            return None
        r = (tl.pnl / tl.risk_amount).to_numpy()
        hold_h = ((pd.to_datetime(tl.close_time) - pd.to_datetime(tl.open_time)).dt.total_seconds() / 3600).median()
        wins, losses = r[r > 0], r[r <= 0]
        avg_w = wins.mean() if len(wins) else 0.0
        avg_l = -losses.mean() if len(losses) else np.nan
        months = max((tv.index[-1] - tv.index[0]).days / 30.44, 1)
        return {"family": row["family"], "strategy": row["strategy"], "market": row["market"], "tf": row["tf"],
                "study": row["study"], "params": row["params"], "trades": len(r),
                "trades_per_month": round(len(r) / months, 2), "win_rate": round(float((r > 0).mean()), 3),
                "avg_win_R": round(float(avg_w), 3), "avg_loss_R": round(float(avg_l), 3),
                "real_RR": round(float(avg_w / avg_l), 2) if avg_l and avg_l > 0 else np.nan,
                "expectancy_R": round(float(r.mean()), 3),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if r.std() > 0 else 0.0,
                "hold_hours": round(float(hold_h), 1)}
    except Exception as e:
        return {"error": repr(e)[:120], "study": row["study"]}


def main() -> None:
    fs = [f for f in glob.glob(str(ROOT / "results" / "registry_v2" / "*.csv")) if "factory_smoke" not in f]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[(df.tr_avgR > 0) & (df.va_avgR > 0)].drop_duplicates(subset=["strategy", "market", "tf", "params"])
    rows = df.to_dict("records")
    print(f"re-running {len(rows)} configurations that passed practice and quiz ...", flush=True)
    with ProcessPoolExecutor(11) as ex:
        out = [o for o in ex.map(one, rows, chunksize=4) if o]
    errs = [o for o in out if "error" in o]
    res = pd.DataFrame([o for o in out if "error" not in o])
    res["style"] = pd.cut(res.hold_hours, [-1, 2, 24, 1e9], labels=["scalp (<2h)", "day trade (2-24h)", "swing (1d+)"])
    res.to_csv(ROOT / "results" / "style_analysis.csv", index=False)
    print(f"done: {len(res)} analysed, {len(errs)} errors\n")
    pd.set_option("display.width", 220)
    g = res.groupby("style", observed=True)
    print(g.agg(configs=("expectancy_R", "size"),
                median_win=("win_rate", "median"), median_RR=("real_RR", "median"),
                median_expectancy=("expectancy_R", "median"),
                best_expectancy=("expectancy_R", "max"),
                win50_and_RR2=("win_rate", lambda s: int(((s >= 0.5) & (res.loc[s.index, "real_RR"] >= 2)).sum())),
                t_ge_3=("t", lambda s: int((s >= 3).sum()))).to_string(), "\n")
    for st, grp in g:
        print(f"== {st}: top 5 by t-stat (train+val, real costs)")
        print(grp.sort_values("t", ascending=False).head(5)[
            ["family", "market", "tf", "trades", "trades_per_month", "win_rate", "real_RR",
             "expectancy_R", "t", "hold_hours"]].to_string(index=False), "\n")
    hi = res[(res.win_rate >= 0.5) & (res.real_RR >= 2)]
    print(f"configs with win rate >= 50% AND reward:risk >= 2: {len(hi)}")
    if len(hi):
        print(hi.sort_values("t", ascending=False).head(10)[
            ["family", "market", "tf", "trades", "win_rate", "real_RR", "expectancy_R", "t", "style"]].to_string(index=False))


if __name__ == "__main__":
    main()
