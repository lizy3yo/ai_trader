#!/bin/bash
cd /c/00-SideProject/AI_Trading
while read f; do
  [ -z "$f" ] && continue
  echo "=== $f $(date +%H:%M:%S)"
  python tools/factory.py --study "$f" --workers 11 2>&1 | grep -v WARNING | grep -E "^\[factory\] FINALISTS|^  [A-Z]" | head -4
done < results/fw_studies.txt
echo "=== ALL DONE $(date +%H:%M:%S)"
