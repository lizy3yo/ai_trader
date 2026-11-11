# Round 6 protocol (gold focus + stock-index extension, 2026-11-08)

Read first: `results/FACTORY_PROTOCOL.md` (factory, rules, file prefixes), `results/ROUND5_SUMMARY.md`
(what passed / failed, backtester fixes), `results/RESEARCH_PROTOCOL.md` (data, cost model,
strategy contract). Everything there still applies, with these changes:

- The factory now logs to `results/registry_v2/` by default (the fixed backtester). Round-5 gold
  results live there too: before building an idea, check whether it was already tested
  (`python tools/leaderboard.py`, or read `results/registry_v2/*.csv` filtered on market GOLD).
- Markets are auto-discovered from `research_*.yaml`. Ten stock indices are being added
  (US30Cash, US2000Cash, GER40Cash, UK100Cash, JP225Cash, EU50Cash, FRA40Cash, AUS200Cash,
  HK50Cash, CA60Cash; H1/H4/D1 data). A study with `"markets": "all"` now includes them —
  gold-focused agents should list markets explicitly.
- The random baseline is now fair for state exits (it never enters where the strategy's own
  exit_signal would fire at once). Keep using `"exit_from_strategy": true` with exit_signal.
- Spread floors scale with price (`spread_ref_price`); Friday D1/H4 signals are traded at the
  next open; `session.flat_at` closes inside the bar that contains it. Indices/metals often have
  no bar at 23:30 — use flat_at before the market's daily close (e.g. 22:45).
- Use `--workers 4` (three agents share 12 cores).
- Pass mark unchanged: train+val t >= 3.0 plus the other checks. The lead runs final exams.
- Never touch the locked TEST split. Never run tools/final_exam.py.

Gold facts to build on (round 4–5): long-only H1 Donchian (360 bars, EMA400, SL3/TP5 ATR,
09:00–23:00) beats random entries but scores t 2.8–3.0 (just at the bar) and swap costs it
~0.05R/trade; shorts never paid (even with paid swap); Asian drift sits in the untradeable first
minute after the 01:00 reopen; London ORB is train-negative under real costs; NY-open break of
the London-morning range (M15, long) is a near miss (t 1.6); FOMC-pre-announcement (H1, t 2.29) and
pre-holiday (D1, t 2.35) on gold were near misses; gold H4 3-bar momentum burst passed train but
failed the quiz. Tick volume (`tick_volume` column) has NOT been used by any gold strategy yet.
