"""g6s_: side-by-side train+val view of every setting in the given studies
(same configurations the factory logged; no new ones). Prints tv trades,
avgR, t, swapR, mean hold (h) and R per calendar year, baseline first.

  python results/g6s_compare.py grids/g6s_dc_be.json [more studies] [--market GOLD]
Never touches the TEST split.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bot.backtest import load_econ, run_backtest  # noqa: E402
from tools.factory import _base, _splits  # noqa: E402
from tools.research import apply_params, r_stats  # noqa: E402


def main() -> None:
    market = "GOLD"
    if "--market" in sys.argv:
        market = sys.argv[sys.argv.index("--market") + 1]
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != market]
    for f in args:
        st = json.loads(Path(f).read_text())
        grid = st["grid"]
        tf = st["timeframes"][0]
        _tr, _va, tv = _splits(market, tf, int(st.get("max_bars", 60000)))
        print(f"== {Path(f).stem} {market} {tf}")
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid, combo))
            cfg = apply_params(_base(market), st["strategy"], tf, params)
            res = run_backtest(cfg, tv, load_econ(cfg.symbol), verbose=False)
            avg, t = r_stats(res)
            log = res.trade_log
            r = np.array([x["pnl"] / x["risk_amount"] for x in log])
            sw = np.array([x["swap"] / x["risk_amount"] for x in log])
            hold = np.mean([(x["close_time"] - x["open_time"]).total_seconds() / 3600 for x in log]) if log else 0
            yrs = {}
            for x, rr in zip(log, r):
                yrs[x["open_time"].year] = yrs.get(x["open_time"].year, 0) + rr
            var = {k: v for k, v in params.items() if len(grid[k]) > 1}
            ys = " ".join(f"{y % 100:02d}:{v:+.1f}" for y, v in sorted(yrs.items()))
            print(f"  {json.dumps(var):<44} n {len(r):4d} avgR {avg:+.3f} t {t:+.2f} "
                  f"swapR {sw.mean() if len(sw) else 0:+.3f} hold {hold:5.0f}h | {ys}")


if __name__ == "__main__":
    main()
