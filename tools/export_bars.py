"""Export real GOLD bars from the running MT5 terminal to CSVs for offline
backtesting/tuning.  Usage: python tools/export_bars.py [bars]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import mt5_client as mc          # noqa: E402
from bot.config import load_config       # noqa: E402


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cfg = load_config()
    mc.connect(cfg.mt5)
    try:
        mc.ensure_symbol(cfg.symbol)
        Path("data").mkdir(exist_ok=True)
        for tf in ("M15", "M30", "H1"):
            bars = mc.closed_bars(cfg.symbol, tf, n)
            out = Path(f"data/{cfg.symbol.lower()}_{tf.lower()}.csv")
            bars.reset_index()[["time", "open", "high", "low", "close"]].to_csv(
                out, index=False)
            print(f"{tf}: {len(bars)} bars {bars.index[0]} -> {bars.index[-1]}  => {out}")
    finally:
        mc.shutdown()


if __name__ == "__main__":
    main()
