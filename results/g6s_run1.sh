#!/bin/sh
# round 6 g6s_ studies, pass 1 (pre-registered)
cd "$(dirname "$0")/.."
for s in g6s_dc_be g6s_dc_trail g6s_dc_chx g6s_dc_filt g6s_dc_ev g6s_ev_union_h1 g6s_vb_h1 g6s_vb_h4 g6s_pb_h1 g6s_short_d1 g6s_short_h4; do
  echo "=== $s"
  python tools/factory.py --study grids/$s.json --workers 4 2>&1 | grep -v WARNING
done
