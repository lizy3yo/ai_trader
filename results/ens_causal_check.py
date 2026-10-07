"""Causality truncation test for ens_vote: add_indicators(df[:i+1]).iloc[-1]
must equal add_indicators(df).iloc[i] for every checked i."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.backtest import load_csv_bars          # noqa: E402
from bot.config import load_config              # noqa: E402
from bot.strategies import get_strategy         # noqa: E402

COLS = ["v_ema", "v_trend", "v_dc", "v_rsi", "v_mom", "v_ses", "score", "vol_ok", "atr"]


def main() -> None:
    cfg = load_config()
    s = cfg.strategy
    s.name = "ens_vote"
    s.timeframe = "M15"
    s.ema_fast, s.ema_slow, s.ema_trend = 9, 21, 100
    s.params = {"w_ema": 1, "w_trend": 1, "w_dc": 1, "w_rsi": 1, "w_mom": 1,
                "w_ses": 1, "dc_lb": 20, "roc_n": 10, "vol_q": 0.4,
                "vol_win": 120, "ses_h0": 9, "ses_h1": 12}
    strat = get_strategy("ens_vote")

    bars = load_csv_bars("data/gold_m15.csv").iloc[-4000:]
    full = strat.add_indicators(bars, s)

    bad = 0
    rng = np.random.default_rng(7)
    idxs = sorted(rng.choice(range(2000, len(bars)), size=40, replace=False))
    for i in idxs:
        trunc = strat.add_indicators(bars.iloc[: i + 1], s).iloc[-1]
        ref = full.iloc[i]
        for c in COLS:
            a, b = trunc[c], ref[c]
            if isinstance(a, (bool, np.bool_)):
                ok = bool(a) == bool(b)
            else:
                a, b = float(a), float(b)
                ok = (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9
            if not ok:
                bad += 1
                print(f"  MISMATCH i={i} {bars.index[i]} col={c}: trunc={a} full={b}")
    print(f"checked {len(idxs)} truncation points x {len(COLS)} cols -> "
          f"{'PASS (0 mismatches)' if bad == 0 else f'FAIL ({bad} mismatches)'}")


if __name__ == "__main__":
    main()
