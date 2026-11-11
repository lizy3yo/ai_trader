"""Summarise every ew_ study in results/registry_v2 (configs, verdicts, best rows)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "results" / "registry_v2").glob("ew_*.csv"))
d = pd.concat([pd.read_csv(f).assign(file=f.stem) for f in files], ignore_index=True)
print(f"{len(d)} configurations in {len(files)} studies")
print(d["verdict"].value_counts().to_string())
d["sumt"] = d["tr_t"].fillna(-99) + d["va_t"].fillna(-99)
cols = ["file", "market", "tf", "params", "tr_n", "tr_avgR", "tr_t", "va_n", "va_avgR", "va_t",
        "tv_n", "tv_avgR", "tv_t", "neigh_ok", "spread_avgR", "p_random", "verdict"]
print("\nShortlisted (gauntlet):")
print(d[d["tv_t"].notna()].sort_values("tv_t", ascending=False)[cols].to_string(max_colwidth=60))
print("\nMean train avgR / share positive by strategy x market:")
print(d.groupby(["strategy", "market"])["tr_avgR"].agg(["mean", "count", lambda s: (s > 0).mean()]).round(3).to_string())
print("\nMean train avgR by strategy x tf:")
print(d.groupby(["strategy", "tf"])["tr_avgR"].agg(["mean", "count", "median"]).round(3).to_string())
print("\nMedian train trades by strategy x tf:")
print(d.groupby(["strategy", "tf"])["tr_n"].median().to_string())
