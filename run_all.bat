@echo off
REM ============================================================
REM  Gold Bot launcher — starts all three strategies, each in
REM  its own window with its own journal and magic number.
REM
REM  BEFORE RUNNING:
REM    1. MetaTrader 5 must be OPEN and LOGGED IN
REM    2. In MT5: Tools > Options > Expert Advisors >
REM       tick "Allow algorithmic trading"
REM    3. Your account equity must be big enough for the
REM       strategy to size a trade (see README "Account size")
REM
REM  To stop: press Ctrl+C in each window, or just close it.
REM  Open positions keep their SL/TP on the broker's server.
REM ============================================================

cd /d "%~dp0"

echo Starting Gold Bot (3 strategies)...
echo.

start "GoldBot - trend_donchian (H1)" cmd /k python main.py live
timeout /t 3 /nobreak >nul

start "GoldBot - fast_channel (M30)" cmd /k python main.py live --config config_fast.yaml
timeout /t 3 /nobreak >nul

start "GoldBot - ses_orb (M15)" cmd /k python main.py live --config config_orb.yaml

echo.
echo Three windows opened — one per strategy.
echo Check progress any time with:  python main.py report
echo.
pause
