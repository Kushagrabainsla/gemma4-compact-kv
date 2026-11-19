import json, math, statistics
R = "experiments/gemma4_kv_rowscale/results/e74-mac-20261118/"
m = json.load(open(R + "metrics_e54.json"))
old = json.load(open("experiments/gemma4_kv_scaletie/results/e54-h100-20261102/e54/metrics.json"))["prompts"]
cells = m["cells"]
print("ncells", len(cells), "status", m.get("status"), "errors", m["errors"], "sha", m["sha_checked"], m["sha_mismatch"])
pids = sorted({k.split("/")[0] for k in cells}, key=int)
layers = sorted({k.split("/")[1] for k in cells}, key=int)
print(pids, layers)
rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))
def pool(getter):
    return rms([statistics.median([getter(cells[f"{p}/{l}"]) for p in pids]) for l in layers])
def poolj(arm, j):
    return pool(lambda c: c["errj"][arm][j])
for X in ("ROW1", "ROW2"):
    ra = [poolj(X, j) / poolj("PTH", j) for j in range(8)]
    print(X, "R_A", [round(r, 4) for r in ra], "med", statistics.median(ra), "max", max(ra))
    pk = pool(lambda c: c["errk"][X]) / pool(lambda c: c["errk"]["PTH"])
    print(X, "R_A_kem", pk)
    g = {l: statistics.median([cells[f"{p}/{l}"]["err"][X] / cells[f"{p}/{l}"]["err"]["PTH"] for p in pids]) for l in layers}
    print(X, "g_A", {l: round(v, 3) for l, v in g.items()})
    rp = {p: rms([cells[f"{p}/{l}"]["err"][X] for l in layers]) / rms([cells[f"{p}/{l}"]["err"]["PTH"] for l in layers]) for p in pids}
    print(X, "R_A(p)", {p: round(v, 3) for p, v in rp.items()})
for a in ("PTH", "ROW1", "ROW2", "STOCK", "TIED", "CALSTD_JOINT", "ROW1_X0", "PTH_X0", "PTH_K", "PTH_V", "ROW1_KROT", "ROW1_KUN", "ROW1_V", "ROW1_Q32", "ROW1_ID"):
    print(a, round(pool(lambda c: c["err"][a]), 5))
for a in ("PTH", "ROW1", "ROW2"):
    print(a, "P_j", [round(poolj(a, j), 4) for j in range(8)])
# per layer medians
for l in layers:
    print(l, {a: round(statistics.median([cells[f"{p}/{l}"]["err"][a] for p in pids]), 4) for a in ("STOCK", "TIED", "PTH", "ROW1", "PTH_X0", "PTH_K", "PTH_V")})
# V8 without A2: cells outside
bad = []
for l in layers:
    for p in pids:
        a = cells[f"{p}/{l}"]["err"]["ROW1_ID"]; b = old[p][l]["err"]["BF16C"]
        if abs(a / b - 1) > 0.01:
            bad.append((p, l, a, b, abs(a - b)))
print("V8 cells outside 1%:", len(bad))
for x in bad: print(x)
print("max abs diff all cells", max(abs(cells[f"{p}/{l}"]["err"]["ROW1_ID"] - old[p][l]["err"]["BF16C"]) for l in layers for p in pids))
print("max bf16c L29", [old[p]["29"]["err"]["BF16C"] for p in pids])
print("sub", {k: round(v, 5) for k, v in json.load(open(R + "summary.json"))["stats"]["sub_mean"].items()})
print("rowmax in Krot max", json.load(open(R + "summary.json"))["stats"]["row_max_in_K_rot_max"])
print("top_share_min", json.load(open(R + "summary.json"))["stats"]["top_share_min"])
print("e5m2", json.load(open(R + "summary.json"))["stats"]["e5m2"], "tie", json.load(open(R + "summary.json"))["stats"]["tie"])
print("w values", {l: cells[f"0/{l}"]["w_value"] for l in layers})
