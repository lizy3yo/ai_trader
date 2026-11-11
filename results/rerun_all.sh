#!/bin/bash
# Re-run every round-5 study on the fixed backtester into results/registry_v2
cd /c/00-SideProject/AI_Trading
while read f; do
  echo "=== $f $(date +%H:%M:%S)"
  python tools/factory.py --study "$f" --workers 11 --registry results/registry_v2 2>&1 | grep -v WARNING | grep -E "^\[factory\] (FINALISTS|.*settings)|^  [A-Z]"
done < results/rerun_studies.txt
echo "=== ALL DONE $(date +%H:%M:%S)"
