"""Deep-dive stats from the backtest trade journals. Prints JSON per strategy."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DBS = {
    "trend_donchian": "data/trades.db",
    "ses_orb": "data/trades_orb.db",
    "mr_rsi2": "data/trades_rsi2.db",
}


def analyze(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE mode='backtest' AND close_time IS NOT NULL "
        "ORDER BY close_time", conn)
    conn.close()
    if df.empty:
        return {"error": "no trades"}
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    df["hold_hours"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 3600
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    monthly = df.set_index("close_time").groupby(pd.Grouper(freq="ME"))["pnl"].sum()
    monthly = monthly[monthly != 0]
    years = df.groupby(df["close_time"].dt.year)["pnl"].agg(["sum", "count"])
    yearly = {str(y): {"pnl": round(r["sum"], 0), "trades": int(r["count"])}
              for y, r in years.iterrows()}
    exits = df.groupby("exit_reason")["pnl"].agg(["count", "sum"])
    return {
        "period": f"{df['open_time'].min().date()} -> {df['close_time'].max().date()}",
        "trades": len(df),
        "wins": len(wins),
        "win_rate_pct": round(100 * len(wins) / len(df), 1),
        "total_pnl": round(df["pnl"].sum(), 0),
        "end_equity": round(df["equity_after"].iloc[-1], 0),
        "avg_win": round(wins["pnl"].mean(), 2),
        "avg_loss": round(losses["pnl"].mean(), 2),
        "biggest_win": round(df["pnl"].max(), 0),
        "biggest_loss": round(df["pnl"].min(), 0),
        "expectancy_per_trade": round(df["pnl"].mean(), 2),
        "profit_factor": round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2),
        "median_hold_hours": round(df["hold_hours"].median(), 1),
        "max_hold_days": round(df["hold_hours"].max() / 24, 1),
        "months_traded": len(monthly),
        "positive_months_pct": round(100 * (monthly > 0).mean(), 0),
        "best_month": round(monthly.max(), 0),
        "worst_month": round(monthly.min(), 0),
        "avg_month": round(monthly.mean(), 0),
        "yearly": yearly,
        "exit_breakdown": {ix: {"count": int(r["count"]), "pnl": round(r["sum"], 0)}
                           for ix, r in exits.iterrows()},
        "trades_per_month": round(len(df) / max(len(monthly), 1), 1),
    }


def main() -> None:
    out = {name: analyze(db) for name, db in DBS.items() if Path(db).exists()}
    print(json.dumps(out, indent=1))
    Path("results/full_history_stats.json").write_text(json.dumps(out, indent=1),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
