"""Re-check chosen finalists with the FAIR coin-flip (random entries never
placed where the strategy's own exit would fire immediately), on train+val
only. Prints one line per candidate.

  python tools/verify_finalists.py results/verify_candidates.json [--seeds 30]

Candidates file: [{"name": ..., "study": "grids/x.json", "market": ..., "tf": ...,
                   "params": {...}, "max_bars": optional override}]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.factory import _run, _splits, _base, edge_p          # noqa: E402
from tools.research import apply_params                       # noqa: E402


def check(args):
    c, seeds = args
    st = json.loads(Path(c["study"]).read_text(encoding="utf-8-sig"))
    mb = int(c.get("max_bars", st.get("max_bars", 60000)))
    tr, va, tv = _splits(c["market"], c["tf"], mb)
    cfg = apply_params(_base(c["market"]), st["strategy"], c["tf"], c["params"])
    m = _run(cfg, tv)
    mtr, mva = _run(cfg, tr), _run(cfg, va, tr)
    cr, rmean, rp95, p = edge_p(c["market"], c["tf"], st["strategy"], c["params"], tv,
                                st.get("dir", "long"), int(st.get("trend", 0)),
                                bool(st.get("exit_from_strategy", False)), seeds=seeds)
    return (f"{c['name']:<34} {c['market']:<10}{c['tf']:<4} span {tv.index[0].date()}..{tv.index[-1].date()} "
            f"| train n{mtr['trades']} t{mtr['t_stat']:+.2f} | val n{mva['trades']} t{mva['t_stat']:+.2f} "
            f"| tv n{m['trades']} avgR {m['avg_r']:+.3f} t {m['t_stat']:+.2f} "
            f"| fair random mean {rmean:+.3f} p95 {rp95:+.3f} p {p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    cands = json.loads(Path(a.candidates).read_text(encoding="utf-8"))
    with ProcessPoolExecutor(a.workers) as ex:
        for line in ex.map(check, [(c, a.seeds) for c in cands]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
