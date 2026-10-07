"""Trade journal: every trade goes into SQLite and mirrors to CSV.

Also persists the bot's risk state (consecutive wins, current risk %,
cumulative P&L) so restarts don't reset compounding or the adaptive risk.
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket INTEGER,                 -- MT5 position ticket (NULL in backtests)
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,        -- LONG / SHORT
    lots REAL NOT NULL,
    open_time TEXT NOT NULL,        -- ISO-8601 UTC
    open_price REAL NOT NULL,
    sl REAL, tp REAL,
    close_time TEXT,
    close_price REAL,
    pnl REAL,                       -- net, includes commission+swap when known
    risk_pct REAL,                  -- adaptive risk % used for this trade
    risk_amount REAL,               -- $ intended risk
    equity_after REAL,              -- bot equity after the trade closed
    reason TEXT,                    -- signal description
    exit_reason TEXT,               -- tp / sl / breaker / manual / eod
    mode TEXT NOT NULL DEFAULT 'live'   -- live / backtest
);
CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL
);
"""

CSV_COLUMNS = [
    "id", "ticket", "symbol", "direction", "lots", "open_time", "open_price",
    "sl", "tp", "close_time", "close_price", "pnl", "risk_pct", "risk_amount",
    "equity_after", "reason", "exit_reason", "mode",
]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TradeTracker:
    def __init__(self, db_path: str, csv_path: str | None = None):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.csv_path = csv_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------- trade lifecycle ----------------

    def open_trade(
        self, *, ticket: int | None, symbol: str, direction: str, lots: float,
        open_time: str, open_price: float, sl: float, tp: float,
        risk_pct: float, risk_amount: float, reason: str, mode: str = "live",
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (ticket, symbol, direction, lots, open_time,
               open_price, sl, tp, risk_pct, risk_amount, reason, mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticket, symbol, direction, lots, open_time, open_price, sl, tp,
             risk_pct, risk_amount, reason, mode),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def close_trade(
        self, trade_id: int, *, close_time: str, close_price: float,
        pnl: float, equity_after: float, exit_reason: str,
    ) -> None:
        self.conn.execute(
            """UPDATE trades SET close_time=?, close_price=?, pnl=?,
               equity_after=?, exit_reason=? WHERE id=?""",
            (close_time, close_price, pnl, equity_after, exit_reason, trade_id),
        )
        self.conn.commit()
        self.export_csv()

    def find_open_trade_by_ticket(self, ticket: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM trades WHERE ticket=? AND close_time IS NULL", (ticket,)
        ).fetchone()

    def open_trades(self, mode: str = "live") -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE close_time IS NULL AND mode=?", (mode,)
        ).fetchall()

    # ---------------- persisted bot state ----------------

    def save_state(self, **kv: float) -> None:
        for key, value in kv.items():
            self.conn.execute(
                "INSERT INTO bot_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, float(value)),
            )
        self.conn.commit()

    def load_state(self, key: str, default: float = 0.0) -> float:
        row = self.conn.execute(
            "SELECT value FROM bot_state WHERE key=?", (key,)
        ).fetchone()
        return float(row["value"]) if row else default

    # ---------------- aggregates ----------------

    def cumulative_pnl(self, mode: str = "live") -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS s FROM trades "
            "WHERE pnl IS NOT NULL AND mode=?", (mode,),
        ).fetchone()
        return float(row["s"])

    def realized_pnl_for_day(self, day_iso: str, mode: str = "live") -> float:
        """day_iso like '2026-08-30' (UTC close date)."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS s FROM trades "
            "WHERE pnl IS NOT NULL AND mode=? AND substr(close_time, 1, 10)=?",
            (mode, day_iso),
        ).fetchone()
        return float(row["s"])

    def daily_summary(self, mode: str = "live") -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT substr(close_time,1,10) AS day,
                      COUNT(*) AS trades,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                      ROUND(SUM(pnl), 2) AS pnl
               FROM trades WHERE close_time IS NOT NULL AND mode=?
               GROUP BY day ORDER BY day""", (mode,),
        ).fetchall()

    def monthly_summary(self, mode: str = "live") -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT substr(close_time,1,7) AS month,
                      COUNT(*) AS trades,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                      ROUND(SUM(pnl), 2) AS pnl,
                      ROUND(MIN(pnl), 2) AS worst,
                      ROUND(MAX(pnl), 2) AS best
               FROM trades WHERE close_time IS NOT NULL AND mode=?
               GROUP BY month ORDER BY month""", (mode,),
        ).fetchall()

    def all_closed(self, mode: str = "live") -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE close_time IS NOT NULL AND mode=? "
            "ORDER BY close_time", (mode,),
        ).fetchall()

    # ---------------- CSV mirror ----------------

    def export_csv(self) -> None:
        if not self.csv_path:
            return
        try:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            rows = self.conn.execute("SELECT * FROM trades ORDER BY id").fetchall()
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)
                for row in rows:
                    writer.writerow([row[c] for c in CSV_COLUMNS])
        except OSError as e:
            # Windows: the CSV being open in Excel locks it. SQLite stays the
            # source of truth — never let the mirror kill the trading loop.
            print(f"[tracker] CSV mirror skipped ({e}) — journal in {self.db_path} is intact")
