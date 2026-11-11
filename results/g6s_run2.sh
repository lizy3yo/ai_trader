#!/bin/sh
# round 6 g6s_ pass 2: FOMC exit anchored to New York time (mechanism check), H4 port of the H1 Donchian (real H4 data only)
cd "$(dirname "$0")/.."
for s in g6s_ev_et_h1 g6s_dc_h4port; do
  echo "=== $s"
  python tools/factory.py --study grids/$s.json --workers 4 2>&1 | grep -v WARNING
done
