import json
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rows = json.load(open(path, encoding="utf-8-sig"))
print(len(rows), "rows survived train filter\n")
for r in rows[:n]:
    t, v = r["train"], r["val"]
    print(f"score {r['score']:>10} | {r['timeframe']} {json.dumps(r['params'])}")
    print(f"   train: ret {t['return_pct']:+6.1f}% pf {t['pf']:5.2f} win {t['win_rate']:4.1f}% "
          f"tr {t['trades']:3d} posM {t['pos_months_pct']:3.0f}% worstM {t['worst_month']:+5.0f} dd {t['max_dd']:4.0f}")
    print(f"   val  : ret {v['return_pct']:+6.1f}% pf {v['pf']:5.2f} win {v['win_rate']:4.1f}% "
          f"tr {v['trades']:3d} posM {v['pos_months_pct']:3.0f}% worstM {v['worst_month']:+5.0f} dd {v['max_dd']:4.0f}")
