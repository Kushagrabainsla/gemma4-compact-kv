"""Test of the recompute hypothesis on E-69 rehearsal try 3: is every mismatched block inside the token range that was
recomputed after the previous reload of the same request, [previous hit, previous n_pre)? Uses the full `bad` lists
only when they are not truncated (at most 8 are stored), else counts with the mismatch totals."""
import json
from pathlib import Path

R = Path("experiments/gemma4_kv_offload/results/e69-reh-20261113T085417Z")
for arm in ("S-C", "C-C", "C-C0", "C-CX"):
    r = json.loads((R / arm / "run_e69.json").read_text())
    bs = {g: x["block_size"] for g, x in enumerate(r["kv"]["groups"])}
    sliding = {g for g, x in enumerate(r["kv"]["groups"]) if x.get("sliding_window")}
    prev: dict = {}
    inside = outside = first_bad = 0
    expect_total = 0
    mism_total = 0
    for x in r["reload"]["results"]:
        p = prev.get(x["req"])
        mism_total += x["mismatch"]
        if p is None:
            first_bad += x["mismatch"]
        else:
            lo, hi = p
            # blocks of a sliding group fully inside [lo, hi) and fully below the current hit
            for g in sliding:
                for i in range(lo // bs[g], hi // bs[g] + 1):
                    if i * bs[g] >= lo and (i + 1) * bs[g] <= hi and (i + 1) * bs[g] <= x["hit"]:
                        expect_total += 1
            for g, i in x["bad"]:
                s, e = i * bs[g], (i + 1) * bs[g]
                if p and s >= lo and e <= hi:
                    inside += 1
                else:
                    outside += 1
        prev[x["req"]] = (x["hit"], x["n_pre"])
    print(f"{arm}: mismatched {mism_total} | in first reloads {first_bad} | listed bad inside the recomputed range "
          f"{inside}, outside {outside} | blocks expected in recomputed ranges {expect_total}")
