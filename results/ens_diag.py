"""Quick diagnostic: run ens_vote (or any strategy) on train+val and print
raw metrics, so I can sanity-check trade frequency before sweeping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.config import load_config                      # noqa: E402
from tools.research import apply_params, load_split, run_one   # noqa: E402


def main() -> None:
    strategy = sys.argv[1]
    tf = sys.argv[2]
    params = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8-sig"))
    max_bars = int(sys.argv[4]) if len(sys.argv) > 4 else 99999
    base = load_config()
    train, val, _test = load_split(base.symbol, tf, max_bars)
    print(f"# {tf} max_bars={max_bars} train {train.index[0]}->{train.index[-1]} "
          f"val {val.index[0]}->{val.index[-1]}")
    for name, combo in params.items():
        cfg = apply_params(base, strategy, tf, combo)
        tr = run_one(cfg, train)
        va = run_one(cfg, val)
        print(f"{name:<22} TRAIN ret {tr['return_pct']:+8.1f}% tr {tr['trades']:5d} "
              f"pf {tr['pf']:5.2f} win {tr['win_rate']:4.1f} | "
              f"VAL ret {va['return_pct']:+8.1f}% tr {va['trades']:5d} "
              f"pf {va['pf']:5.2f} win {va['win_rate']:4.1f} posM {va['pos_months_pct']:3.0f} "
              f"mo {va['months']:3d}")


if __name__ == "__main__":
    main()
