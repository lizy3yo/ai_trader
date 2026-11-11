r"""Bulk export of D1 bars + economics for a whole MT5 symbol group (2026-11-09).

Writes data/stocks/<symbol>_d1.csv and data/stocks/<symbol>_econ.json for every
symbol whose MT5 path starts with --group (default Stocks\US), plus
data/stocks/_universe.csv (symbol, bars, first date, live spread %).

Usage: python tools/export_universe.py [--group "Stocks\US"] [--bars 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import mt5_client as mc          # noqa: E402
from bot.config import load_config       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="Stocks/US")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--out", default="data/stocks")
    ap.add_argument("--skip", default="", help="comma list of path prefixes to skip")
    args = ap.parse_args()
    mc.connect(load_config().mt5)
    mt5 = mc.mt5
    out_dir = Path(args.out)
    groups = [g.strip() for g in args.group.split(",")]
    skips = [g.strip() for g in args.skip.split(",") if g.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for s in mt5.symbols_get():
            pth = s.path.replace("\\", "/")
            if not any(pth.startswith(g) for g in groups) or any(pth.startswith(k) for k in skips):
                continue
            if (out_dir / f"{s.name.lower()}_d1.csv").exists():
                continue
            name = s.name
            print(f"{name} ...", end=" ", flush=True)
            mt5.symbol_select(name, True)
            rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_D1, 1, args.bars)
            if rates is None or len(rates) < 300:
                print("skip", flush=True)
                continue
            print(len(rates), flush=True)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].to_csv(
                out_dir / f"{name.lower()}_d1.csv", index=False)
            info = mt5.symbol_info(name)
            tick = mt5.symbol_info_tick(name)
            live_sp = (tick.ask - tick.bid) if tick and tick.bid > 0 else float("nan")
            econ = {
                "symbol": name, "tick_size": info.trade_tick_size,
                "tick_value": mc.account_tick_value(name), "point": info.point,
                "tick_value_profit_ccy": info.trade_tick_value, "currency_profit": info.currency_profit,
                "contract_size": info.trade_contract_size, "volume_min": info.volume_min,
                "volume_max": info.volume_max, "volume_step": info.volume_step,
                "swap_mode": info.swap_mode, "swap_long_points": info.swap_long,
                "swap_short_points": info.swap_short,
                "swap_rollover3days_mt5": info.swap_rollover3days,
                "live_spread": live_sp, "live_bid": tick.bid if tick else None,
            }
            (out_dir / f"{name.lower()}_econ.json").write_text(json.dumps(econ, indent=1), encoding="utf-8")
            med_sp = float((df.spread.tail(250) * info.point / df.close.tail(250)).median())
            rows.append({"symbol": name, "bars": len(df), "first": df.time.iloc[0].date(),
                         "last": df.time.iloc[-1].date(), "close": df.close.iloc[-1],
                         "bar_spread_pct": 100 * med_sp,
                         "live_spread_pct": 100 * live_sp / tick.bid if tick and tick.bid > 0 else None,
                         "swap_long": info.swap_long, "swap_mode": info.swap_mode,
                         "triple": info.swap_rollover3days, "min_lot": info.volume_min,
                         "contract": info.trade_contract_size})
    finally:
        mc.shutdown()
    u = pd.DataFrame(rows)
    u.to_csv(out_dir / "_universe.csv", index=False)
    print(len(u), "symbols exported")
    print(u.describe(include="all").T.to_string())


if __name__ == "__main__":
    main()
