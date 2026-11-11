"""Export real bars + symbol economics from the running MT5 terminal.

Writes data/<symbol>_<tf>.csv (time, OHLC, tick_volume, spread) and
data/<symbol>_econ.json (tick size/value, lots, swap) so offline backtests
use exactly this broker's contract, spread history and overnight fees.

Usage: python tools/export_bars.py [--symbol GOLD] [--tfs M15,M30,H1] [--bars 99999]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import mt5_client as mc          # noqa: E402
from bot.config import load_config       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="default: symbol in config.yaml")
    ap.add_argument("--tfs", default="M15,M30,H1")
    ap.add_argument("--bars", type=int, default=99999)
    args = ap.parse_args()

    cfg = load_config()
    symbol = args.symbol or cfg.symbol
    mc.connect(cfg.mt5)
    try:
        mc.ensure_symbol(symbol)
        Path("data").mkdir(exist_ok=True)
        info = mc.mt5.symbol_info(symbol)
        econ = {
            "symbol": symbol, "tick_size": info.trade_tick_size,
            # account-currency value (trade_tick_value can be in the profit currency)
            "tick_value": mc.account_tick_value(symbol), "point": info.point,
            "tick_value_profit_ccy": info.trade_tick_value, "currency_profit": info.currency_profit,
            "contract_size": info.trade_contract_size,
            "volume_min": info.volume_min, "volume_max": info.volume_max,
            "volume_step": info.volume_step, "swap_mode": info.swap_mode,
            "swap_long_points": info.swap_long, "swap_short_points": info.swap_short,
            # MT5 numbers days 0=Sunday..6=Saturday; 7 = no triple day
            "swap_rollover3days_mt5": info.swap_rollover3days,
        }
        econ_path = Path(f"data/{symbol.lower()}_econ.json")
        econ_path.write_text(json.dumps(econ, indent=1), encoding="utf-8")
        print(f"economics => {econ_path}")
        for tf in (t.strip().upper() for t in args.tfs.split(",")):
            rates = mc.mt5.copy_rates_from_pos(symbol, mc._tf_const(tf), 1, args.bars)
            if rates is None or len(rates) == 0:
                print(f"{tf}: no data ({mc.mt5.last_error()})")
                continue
            import pandas as pd
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            out = Path(f"data/{symbol.lower()}_{tf.lower()}.csv")
            df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].to_csv(
                out, index=False)
            print(f"{tf}: {len(df)} bars {df.time.iloc[0]} -> {df.time.iloc[-1]}  => {out}")
    finally:
        mc.shutdown()


if __name__ == "__main__":
    main()
