"""Where are the mismatched blocks of E-69 rehearsal try 3? Per reload: hit, n_pre, and each bad block's group, index
and token range against the hit and against n_pre."""
import json
import sys
from collections import Counter
from pathlib import Path

R = Path("experiments/gemma4_kv_offload/results/e69-reh-20261113T085417Z")
arm = sys.argv[1] if len(sys.argv) > 1 else "C-C"
r = json.loads((R / arm / "run_e69.json").read_text())
groups = r["kv"]["groups"]
bs = {g: x["block_size"] for g, x in enumerate(groups)}
kinds = {g: (x["type"], x["layers"], x.get("sliding_window")) for g, x in enumerate(groups)}
print("groups", kinds)
pos = Counter()
for x in r["reload"]["results"]:
    rel = []
    for g, i in x["bad"]:
        end = (i + 1) * bs[g]
        rel.append((g, i, end - x["hit"], end - x["n_pre"]))
        pos[(g in {gg for gg, k in kinds.items() if k[2]}, end - x["hit"])] += 1
    print(f"{x['req']} hit {x['hit']} n_pre {x['n_pre']} checked {x['checked']} (glob {x['global']} slid "
          f"{x['sliding']}) mismatch {x['mismatch']} same_id {x['same_id']} bad(g,i,end-hit,end-npre) {rel}")
print("by (sliding, end-hit) of the first 8 bad blocks:", sorted(pos.items(), key=lambda kv: -kv[1])[:15])
print("events w:", r["events"].get("w", [])[:8])
print("w:", r.get("w"))
