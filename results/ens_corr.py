"""Why the vote ensemble adds nothing: measure how INDEPENDENT the component
signals actually are.

An ensemble only beats its parts when the parts make uncorrelated errors. This
prints the pairwise correlation of the ens_vote component votes and the share
of bars on which the committee is unanimous, per timeframe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import load_csv_bars                      # noqa: E402
from bot.config import load_config                          # noqa: E402
from bot.strategies import get_strategy                     # noqa: E402
from tools.research import apply_params                     # noqa: E402

COLS = ["v_ema", "v_trend", "v_dc", "v_rsi", "v_mom", "v_ses"]


def main() -> None:
    base = load_config()
    strat = get_strategy("ens_vote")
    params = {"cfg.ema_fast": 20, "cfg.ema_slow": 50, "cfg.ema_trend": 200,
              "w_ema": 1, "w_trend": 1, "w_dc": 1, "w_rsi": 1, "w_mom": 1, "w_ses": 1,
              "dc_lb": 96, "roc_n": 10, "ses_h0": 9, "ses_h1": 16, "k_thresh": 1.0}
    for tf, path, mb in (("M15", "data/gold_m15.csv", 99999),
                         ("M30", "data/gold_m30.csv", 99999),
                         ("H1", "data/gold_h1.csv", 79000)):
        cfg = apply_params(base, "ens_vote", tf, params)
        bars = load_csv_bars(path)
        bars = bars.iloc[-mb:] if len(bars) > mb else bars
        bars = bars.iloc[: int(len(bars) * 0.8)]      # train+val only, TEST untouched
        ind = strat.add_indicators(bars, cfg.strategy).iloc[strat.warmup(cfg.strategy):]
        v = ind[COLS].astype(float).dropna()
        print(f"\n=== {tf}  ({len(v)} bars, {bars.index[0].date()} -> {bars.index[-1].date()}) ===")
        print("  pairwise vote correlation")
        print(v.corr().round(2).to_string())
        core = v[["v_ema", "v_trend", "v_dc", "v_rsi"]]
        unan4 = float((core.abs().sum(axis=1).eq(4) & core.sum(axis=1).abs().eq(4)).mean() * 100)
        agree_et = float((v["v_ema"] == v["v_trend"]).mean() * 100)
        agree_dt = float((v["v_dc"] == v["v_trend"]).mean() * 100)
        agree_rt = float((v["v_rsi"] == v["v_trend"]).mean() * 100)
        print(f"  unanimous (4 core votes all +1 or all -1): {unan4:5.1f}% of bars")
        print(f"  v_ema == v_trend {agree_et:5.1f}% | v_dc == v_trend {agree_dt:5.1f}% | "
              f"v_rsi == v_trend {agree_rt:5.1f}%")
        s = (v[["v_ema", "v_trend", "v_dc", "v_rsi"]].sum(axis=1) / 4.0)
        print(f"  |score| distribution: " + "  ".join(
            f"{q}:{s.abs().quantile(q):.2f}" for q in (0.25, 0.5, 0.75, 0.9)))


if __name__ == "__main__":
    main()
