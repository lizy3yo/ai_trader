#!/bin/bash
cd /c/00-SideProject/AI_Trading
while read f; do
  [ -z "$f" ] && continue
  echo "=== $f $(date +%H:%M:%S)"
  python tools/factory.py --study "$f" --workers 11 2>&1 | grep -v WARNING | grep -E "^\[factory\] (FINALISTS|.*settings)|^  [A-Z]" | head -5
done < results/eu_studies.txt
echo "=== ALL DONE $(date +%H:%M:%S)"
