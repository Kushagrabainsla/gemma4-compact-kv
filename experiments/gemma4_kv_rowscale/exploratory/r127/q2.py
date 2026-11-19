import json
R = "experiments/gemma4_kv_rowscale/results/e74-mac-20261118/"
a = json.load(open(R + "summary_run1_before_A2.json"))
b = json.load(open(R + "summary.json"))
print("run1 checks", {k: v for k, v in a["checks"].items() if k.startswith("V")})
print("A2 checks", {k: v for k, v in b["checks"].items() if k.startswith("V")})
print("stats identical", a["stats"] == b["stats"])
print("stats e53 identical", a["stats_e53"] == b["stats_e53"])
print("run1 has stats?", bool(a["stats"]), a["stats"]["ROW1"]["R_A_med"])
print("same_v_scale_min", b["stats"]["same_v_scale_min"])
print("e53 ROW1 R_A med", b["stats_e53"]["ROW1"]["R_A_med"], b["stats_e53"]["ROW1"]["R_A_min"], b["stats_e53"]["ROW1"]["R_A_max"], "vsSTOCK", b["stats_e53"]["ROW1"]["vs_STOCK"], "vsTIED", b["stats_e53"]["ROW1"]["vs_TIED"])
print("e53 P", {k: round(v, 4) for k, v in b["stats_e53"]["P"].items() if k in ("PTH", "ROW1", "STOCK", "TIED", "CALSTD_JOINT")})
print("pth vs tied", b["stats"]["pth"]["vs_TIED"], "vs joint", b["stats"]["pth"]["vs_CALSTD_JOINT"])
import statistics
for x in ("PTH", "ROW1", "ROW2"):
    pj = b["stats"]["P_factors"][x]
    print(x, "median P_j", statistics.median(pj), "j-wise vs TIED", [round(v / b["stats"]["P"]["TIED"], 2) for v in pj], "vs STOCK median", statistics.median(pj) / b["stats"]["P"]["STOCK"])
print("classes", b["stats"]["classes"])
print("g_STOCK ROW1", b["stats"]["ROW1"]["g_STOCK"])
print("vs_STOCK_factors ROW1", b["stats"]["ROW1"]["vs_STOCK_factors"])
print("code sha", b["code_sha256"])
