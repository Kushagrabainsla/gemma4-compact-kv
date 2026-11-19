"""Per reload of E-69 rehearsal try 3: mismatches against the fully stored sliding blocks inside the previous
recomputed range [previous hit, previous n_pre); a reload with more mismatches than such blocks has a block outside."""
import json
from pathlib import Path

R = Path("experiments/gemma4_kv_offload/results/e69-reh-20261113T085417Z")
for arm in ("S-C", "C-C", "C-CX"):
    r = json.loads((R / arm / "run_e69.json").read_text())
    bs = {g: x["block_size"] for g, x in enumerate(r["kv"]["groups"])}
    sliding = {g for g, x in enumerate(r["kv"]["groups"]) if x.get("sliding_window")}
    prev: dict = {}
    over = complete_lists = 0
    for x in r["reload"]["results"]:
        p = prev.get(x["req"])
        n_in = 0
        if p:
            lo, hi = p
            n_in = sum(1 for g in sliding for i in range(hi // bs[g] + 1)
                       if i * bs[g] >= lo and (i + 1) * bs[g] <= min(hi, x["hit"]))
        planted = 1 if arm == "C-CX" else 0
        if x["mismatch"] > n_in + planted:
            over += 1
        complete_lists += x["mismatch"] <= 8
        prev[x["req"]] = (x["hit"], x["n_pre"])
    print(f"{arm}: reloads {len(r['reload']['results'])}, with more mismatches than recomputed blocks (+1 planted in "
          f"C-CX): {over}; reloads with a complete bad list: {complete_lists}")
