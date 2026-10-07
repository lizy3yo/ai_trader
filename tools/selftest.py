"""Quick self-test of the risk math, strategy edge cases and config validation.

Run:  python tools/selftest.py
"""
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.config import CapitalCfg, RiskCfg, load_config          # noqa: E402
from bot.risk import RiskManager, SymbolEconomics                # noqa: E402
from bot.strategy import rsi                                     # noqa: E402

ECON = SymbolEconomics(tick_size=0.01, tick_value=1.0, volume_min=0.01,
                       volume_max=100.0, volume_step=0.01, point=0.01)


def main() -> None:
    rm = RiskManager(RiskCfg(), CapitalCfg())

    # 1) lot-floor float fix: equity 580, 5% = $29 risk, SL $1.00 -> exactly 0.29 lots
    d = rm.size_position(580.0, 1.00, ECON)
    assert abs(d.lots - 0.29) < 1e-12, d.lots
    print(f"lot floor ok: {d.lots} lots, risk ${d.risk_amount:.2f}")

    # 2) bot_equity: losses always shrink equity, profits compound
    assert rm.bot_equity(-100) == 400.0
    assert rm.bot_equity(+250) == 750.0
    assert rm.bot_equity(+250, account_equity=600.0) == 600.0
    print("bot_equity ok")

    # 3) adaptive ladder: 5 -> 6 after 3 wins -> 7 after 6 -> reset on loss
    for _ in range(3):
        rm.record_trade_result(+10)
    assert rm.current_risk_pct == 6.0, rm.current_risk_pct
    for _ in range(3):
        rm.record_trade_result(+10)
    assert rm.current_risk_pct == 7.0
    rm.record_trade_result(-5)
    assert rm.current_risk_pct == 5.0 and rm.consecutive_wins == 0
    print("adaptive ladder ok")

    # 4) step_up_after_wins=0 with step_up_pct=0 no longer divides by zero
    rm2 = RiskManager(RiskCfg(step_up_after_wins=0, step_up_pct=0.0), CapitalCfg())
    rm2.record_trade_result(+10)
    print("zero-step guard ok")

    # 5) breaker trip-and-stay
    rm3 = RiskManager(RiskCfg(), CapitalCfg())
    assert not rm3.check_daily_breaker("d1", -100, -50)
    assert rm3.check_daily_breaker("d1", -150, -60)
    assert rm3.check_daily_breaker("d1", 0, 0)       # stays tripped
    assert not rm3.check_daily_breaker("d2", 0, 0)   # new day resets
    print("breaker ok")

    # 6) RSI reads 100 (not neutral 50) on a losses-free window
    up = pd.Series(range(1, 40), dtype=float)
    assert rsi(up, 14).iloc[-1] == 100.0
    print("rsi edge ok")

    # 7) config validation catches bad values with readable errors
    base = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    bad = Path("data/_selftest_bad.yaml")
    bad.parent.mkdir(exist_ok=True)
    cases = [
        (lambda c: c["strategy"].__setitem__("timeframe", "M7"), "timeframe"),
        (lambda c: c["session"].__setitem__("trade_start", "25:99"), "trade_start"),
        (lambda c: c["risk"].__setitem__("step_up_after_wins", 0), "step_up_after_wins"),
    ]
    for mutate, want in cases:
        c = copy.deepcopy(base)
        mutate(c)
        bad.write_text(yaml.safe_dump(c), encoding="utf-8")
        try:
            load_config(bad)
            raise SystemExit(f"validation MISSED: {want}")
        except ValueError as e:
            assert want in str(e), (want, str(e))
    # quoted numbers must be coerced, not crash
    c = copy.deepcopy(base)
    c["risk"]["base_risk_pct"] = "5"
    bad.write_text(yaml.safe_dump(c), encoding="utf-8")
    cfg = load_config(bad)
    assert cfg.risk.base_risk_pct == 5.0 and isinstance(cfg.risk.base_risk_pct, float)
    bad.unlink()
    print("config validation ok")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
