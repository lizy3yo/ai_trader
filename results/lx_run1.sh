cd /c/00-SideProject/AI_Trading
for s in lx_pairs_rev_metals lx_pairs_rev_indices lx_pairs_rev_crypto lx_pairs_mom_metals lx_pairs_mom_indices lx_pairs_mom_crypto lx_xl_usd lx_xl_crypto lx_xl_risk; do
  echo "=== $s"; python tools/factory.py --study grids/$s.json --workers 3 2>&1 | grep -v WARNING
done
