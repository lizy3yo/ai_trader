@echo off
REM ============================================================
REM  Evidence-based portfolio (round 6, 2026-11-08) — DEMO FIRST.
REM    1. S&P 500 dip-buyer   (passed final exam: 65 trades, t 3.05)
REM    2. Nikkei dip-buyer    (passed final exam, but only 11 trades)
REM    3. GOLD trend_donchian (beats random entries; no clean final exam)
REM  Each runs in its own window with its own journal and magic number.
REM  The three have ~0 correlation, which is why they are run together.
REM
REM  BEFORE RUNNING: MetaTrader 5 open and logged in, and
REM  Tools > Options > Expert Advisors > "Allow algorithmic trading".
REM  run_all.bat (older set incl. fast_channel / ses_orb) is unchanged.
REM ============================================================

cd /d "%~dp0"

start "Portfolio - US500 RSI2 dip (D1)" cmd /k python main.py live --config config_us500_rsi2.yaml
timeout /t 3 /nobreak >nul

start "Portfolio - JP225 pullback (D1)" cmd /k python main.py live --config config_jp225_pullback.yaml
timeout /t 3 /nobreak >nul

start "Portfolio - GOLD trend_donchian (H1)" cmd /k python main.py live

echo.
echo Three windows opened. Check progress with:  python main.py report --config ^<config^>
echo.
pause
