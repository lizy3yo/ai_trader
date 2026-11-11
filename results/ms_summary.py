"""Summarise every ms_ study in results/registry_v2 (configs, verdicts, best rows)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "results" / "registry_v2").glob("ms_*.csv"))
d = pd.concat([pd.read_csv(f).assign(file=f.stem) for f in files], ignore_index=True)
print(f"{len(d)} configurations in {len(files)} studies")
d["stage"] = d["verdict"].str.replace(r"fail: (luck bar|coin flip|wider spread|nearby settings).*", "fail: gauntlet", regex=True)
print(d.groupby("file")["stage"].value_counts().unstack(fill_value=0).to_string())
d["sumt"] = d["tr_t"].fillna(-99) + d["va_t"].fillna(-99)
cols = ["file", "market", "tf", "params", "tr_n", "tr_avgR", "tr_t", "va_n", "va_avgR", "va_t",
        "tv_t", "neigh_ok", "spread_avgR", "p_random", "verdict"]
print("\nShortlisted (checks 3-5):")
print(d[d["tv_t"].notna()].sort_values("tv_t", ascending=False)[cols].to_string(max_colwidth=70))
print("\nBest train+val by sum of t (any verdict):")
print(d.sort_values("sumt", ascending=False)[cols].head(15).to_string(max_colwidth=70))
print("\nMean train avgR by strategy x market:")
print(d.groupby(["strategy", "market"])["tr_avgR"].agg(["mean", "count", lambda s: (s > 0).mean()]).round(3).to_string())
