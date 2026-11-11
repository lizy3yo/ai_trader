"""Leaderboard over the factory registry (results/registry.csv).

Shows how much was searched (the reason the pass mark is high), how far each
family got through the checks, the best configurations, and the finalists.

  python tools/leaderboard.py [--top 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.factory import REGISTRY, T_BAR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--registry", default=None, help="registry folder (default results/registry)")
    args = ap.parse_args()
    reg = ROOT / args.registry if args.registry else REGISTRY
    files = sorted(reg.glob("*.csv")) if reg.exists() else []
    files = [f for f in files if not f.stem.startswith("factory_smoke")]
    if not files:
        raise SystemExit("no registry yet — run tools/factory.py first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n = len(df)
    print(f"configurations tested: {n}   (strategy x market x timeframe x setting)")
    print(f"expected lucky 'winners' at the usual 5% level: ~{n * 0.05:.0f}  -> pass mark t >= {T_BAR}\n")

    stage = df["verdict"].fillna("").str.replace(r"^fail: ", "", regex=True).str.split(",").str[0]
    print("where configurations stopped:")
    print(stage.value_counts().to_string(), "\n")

    fam = df.groupby("family").agg(
        tested=("verdict", "size"),
        practice_ok=("tr_avgR", lambda s: int((s > 0).sum())),
        quiz_ok=("va_avgR", lambda s: int((s > 0).sum())),
        finalists=("verdict", lambda s: int((s == "FINALIST").sum())),
    )
    print("by family:")
    print(fam.sort_values("finalists", ascending=False).to_string(), "\n")

    best = df.dropna(subset=["tv_t"]).sort_values("tv_t", ascending=False).head(args.top)
    cols = ["family", "market", "tf", "tv_n", "tv_avgR", "tv_t", "va_t", "neigh_ok",
            "spread_avgR", "p_random", "verdict", "params"]
    print(f"top {args.top} by train+val t:")
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
