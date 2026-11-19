"""E-57 frozen analysis (README Sections 5 to 7 as amended by A1): the validity checks V1 to V9, the build check EQ
(COMPACT against EMULV), the accuracy rule H1 (COMPACT against STOCK, the E-56 interval and margin), the capacity rule
CAP (the steady decode footprints), the throughput rule TPUT, the overhead rule OVH, and the reported rows (answer
changes, the reference NLL, GROSS, NAIVE, the peak and admission ratios, the hook and cache costs, the schedule
fingerprints).

Usage: analyze_e57.py --res <res> --data data_e56.json [--json]   (<res>/stock and <res>/compact from run_e57.py)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
sys.path += [str(HERE.parent / "gemma4_kv_tasks"), str(HERE.parent / "gemma4_kv_model")]
import analyze_e56 as an56  # noqa: E402
import compact_layout as cl  # noqa: E402

GAP_MAX = 0.02                 # V4: the attention-level gap of OFF and VIEWS (E-55 and E-56 records: 0.0025 or less)
GROSS_GSM_DROP, GROSS_NEEDLE_DROP = 0.20, 0.25          # V7: the liveness of the hooks
M_ACC = 0.03                   # H1: the E-56 margin
NEAR_ANSWERS = 3               # EQ NEAR (A1, R-82 F1): at most 3 final answers (E-56: 2 for BF16C, 3 for BF16R), and
NEAR_NLL = an56.M_NLL          # the 95% interval of the NLL change inside +/-0.002 (the E-56 GO3 margin)
CAP_GO = {32_768: 1.25, 65_536: 1.30}                     # CAP: the footprint ratio lines (B3 predicts 1.338, 1.403)
CAP_KILL = 1.15                                           # CAP: at the longest length
TPUT_GAIN, TPUT_FLOOR, TPUT_NONE = 1.15, 0.97, 1.05
OVH_MAX = 1.05
MIN_WINDOW = 32
DECISION_ARMS = {"stock": ("STOCK", "EMULV", "DET"), "compact": ("COMPACT",)}
V_KEYS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9")
LOG_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


def load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def arrays(d: Path, arm: str) -> dict:
    p = d / "arrays" / f"{arm}.npz"
    return dict(np.load(p)) if p.exists() else {}


def subset(data: dict, run: dict) -> dict:
    """The items of the run (a rehearsal runs the first items only)."""
    g, n = (run or {}).get("items", [len(data["gsm8k"]), len(data["needle"])])
    return dict(data, gsm8k=data["gsm8k"][:g], needle=data["needle"][:n])


def clean(r: dict | None) -> bool:
    """A clean row (A1, R-82 B2): no preemption and a decode window of MIN_WINDOW steps or more; a deferred admission
    (a failed allocation while the previous request frees its prefill blocks) is reported, not rejected."""
    return bool(r) and r.get("preemptions") == 0 and r.get("window_steps", 0) >= MIN_WINDOW and bool(r.get("step_ms"))


def accepted(rows: list[dict]) -> dict | None:
    """The accepted C2 row: the clean row with the largest batch."""
    ok = [r for r in rows or [] if clean(r)]
    return max(ok, key=lambda r: r["batch"]) if ok else None


def limit_shown(rows: list[dict], acc: dict | None, cap: int) -> str:
    """How the run showed that the accepted batch is the largest: the next batch failed, the probe was refused, the
    max_num_seqs cap, or not shown (a time cut)."""
    if acc is None:
        return "none"
    if any(r.get("batch") == acc["batch"] + 1 and not clean(r) for r in rows or []):
        return "next batch failed"
    if acc["batch"] >= cap:              # A2: at the cap the probe is refused for max_num_seqs, not for the pool
        return "max_num_seqs"
    return "probe refused" if acc.get("probe_admitted") is False else "not shown"


def log_concurrency(d: Path) -> dict | None:
    """The engine's startup line "Maximum concurrency for N tokens per request: X" (the admission bound)."""
    p = d / "main_log.txt"
    m = LOG_CONC.search(p.read_text(errors="replace")) if p.exists() else None
    return {"tokens": int(m.group(1).replace(",", "")), "x": float(m.group(2))} if m else None


def step_ratio(a: dict | None, b: dict | None) -> float | None:
    """step_ms(a) / step_ms(b) for two clean rows at the same batch, else None."""
    if clean(a) and clean(b) and a["batch"] == b["batch"]:
        return a["step_ms"] / b["step_ms"]
    return None


def ids_equal(a: dict, b: dict) -> tuple[int, int]:
    """(items with identical output token ids, items compared) over the GSM8K and needle answers."""
    same = n = 0
    for task in ("gsm8k", "needle"):
        for i, o in (a or {}).get(task, {}).items():
            p = (b or {}).get(task, {}).get(i)
            if p is not None:
                n += 1
                same += o["ids"] == p["ids"]
    return same, n


def arrays_equal(a: dict, b: dict) -> tuple[int, int]:
    keys = sorted(set(a) & set(b))
    return sum(bool(np.array_equal(a[k], b[k])) for k in keys), len(keys)


def nll_change(arr_a: dict, arr_b: dict, data: dict) -> tuple[float, tuple[float, float], int]:
    """The mean change of the reference NLL per token (b against a), with the E-56 ratio interval."""
    ra, rb = an56.ref_nll(arr_a, data), an56.ref_nll(arr_b, data)
    ids = [g["id"] for g in data["gsm8k"] if g["id"] in ra and g["id"] in rb]
    if len(ids) < 2:
        return float("nan"), (float("nan"), float("nan")), len(ids)
    s = np.array([-(rb[i][0] - ra[i][0]) for i in ids])
    n = np.array([ra[i][1] for i in ids], dtype=np.float64)
    d, ci = an56.ratio_interval(s, n)
    return d, ci, len(ids)


def final_changes(ga: dict, gb: dict) -> int:
    return sum(1 for i, x in ga.items() if i in gb and not an56.same_answer(x["answer"], gb[i]["answer"]))


def summarize(res: Path, data_path: Path) -> dict:
    data = json.loads(data_path.read_text())
    S, C = res / "stock", res / "compact"
    rs, rc = load(S / "run_main.json", {}), load(C / "run_main.json", {})
    sub = subset(data, rs or rc)
    reh = bool(rs.get("rehearsal") or rc.get("rehearsal"))
    outs = {a: load(S / f"outputs_{a}.json") for a in ("STOCK", "EMULV", "DET")}
    outs.update({a: load(C / f"outputs_{a}.json") for a in ("COMPACT", "GROSS", "NAIVE")})
    arr = {a: arrays(S, a) for a in ("STOCK", "EMULV", "DET")}
    arr.update({a: arrays(C, a) for a in ("COMPACT", "GROSS", "NAIVE")})
    graded = {a: an56.grade(sub, o) if o else {} for a, o in outs.items()}
    v = {k: [] for k in V_KEYS}

    # V1: the two layouts as built.
    def glob_group(r):
        return next((g for g in (r.get("kv") or {}).get("groups", []) if g["type"] == "FullAttentionSpec"), None)
    gs, gc = glob_group(rs), glob_group(rc)
    slide = next((g for g in (rc.get("kv") or {}).get("groups", []) if g["type"] == "SlidingWindowSpec"), None)
    if gs is None or gs.get("state_content_bytes") is not None or gs.get("page_size_padded") is not None:
        v["V1"].append("stock: the global group is not the stock spec")
    if (rs.get("tensors") or {}).get("global", {}).get("shape", [0])[-1] != 2 * cl.HEAD:
        v["V1"].append("stock: the global tensor rows are not 1,024 values")
    spec = rc.get("spec") or {}
    n_hooked = len((rc.get("hooks") or {}).get("hooked", []))
    if not spec.get("applied") or len(spec.get("layers", [])) != n_hooked or n_hooked == 0:
        v["V1"].append("compact: the spec patch did not cover every hooked global layer")
    if gc is None or gc.get("state_content_bytes") != cl.ROW * 2 or gc.get("block_size") != spec.get("block"):
        v["V1"].append("compact: the global group is not the compact spec")
    elif slide is None or gc.get("page_size_padded") != slide.get("page_bytes"):
        v["V1"].append("compact: the global page is not padded to the sliding page")
    shape = (rc.get("tensors") or {}).get("global", {}).get("shape", [0, 0, 0, 0])
    if shape[-1] != cl.ROW or shape[2] != spec.get("block"):
        v["V1"].append(f"compact: the global tensor is {shape}, not [B, H, {spec.get('block')}, 640]")
    for name, r in (("stock", rs), ("compact", rc)):         # A1 (R-82 F8): includes a KV-shared global layer
        for p in (r.get("hooks") or {}).get("problems", []):
            v["V1"].append(f"{name}: {p}")

    # V2: the V-view fix ran in every compact arm and never in the stock engine.
    for a, info in (rc.get("arms") or {}).items():
        c = info.get("counters", {})
        if c.get("fix_reads", 0) == 0 or c.get("fix_writes", 0) == 0:
            v["V2"].append(f"{a}: no V-view fix in the compact engine")
    for a, info in (rs.get("arms") or {}).items():
        c = info.get("counters", {})
        if c.get("fix_reads", 0) or c.get("fix_writes", 0):
            v["V2"].append(f"{a}: a V-view fix in the stock engine")

    # V3: equal pools.
    ks, kc = rs.get("kv") or {}, rc.get("kv") or {}
    if not ks or ks.get("num_blocks") != kc.get("num_blocks") or ks.get("bytes_per_block") != kc.get("bytes_per_block"):
        v["V3"].append(f"pools differ: {ks.get('num_blocks')} and {kc.get('num_blocks')} blocks")

    # V4: the attention-level check of the pre-flight.
    for name, r, modes in (("stock", rs, ("OFF", "VIEWS")), ("compact", rc, ("VIEWS",))):
        pf = r.get("preflight") or {}
        for m in modes:
            gaps = pf.get("gaps", {}).get(m, {})
            if len(gaps) != n_hooked or n_hooked == 0:
                v["V4"].append(f"{name} {m}: gaps for {len(gaps)} of {n_hooked} layers")
            elif max(gaps.values()) > GAP_MAX:
                v["V4"].append(f"{name} {m}: a gap of {max(gaps.values()):.4f} (bound {GAP_MAX})")
        if m == "VIEWS" and pf.get("seps", {}).get("VIEWS") and max(pf["seps"]["VIEWS"].values()) > GAP_MAX:
            v["V4"].append(f"{name}: VIEWS separates from the stock reference")
        if pf.get("hook_errors") or (r.get("evidence") or {}).get("hook_errors"):
            v["V4"].append(f"{name}: hook errors")
        if not all((pf.get("finite") or {}).values()):
            v["V4"].append(f"{name}: a non-finite probe score")

    # V5: the health of STOCK (the E-56 V8 bounds).
    health = rs.get("health") or {}
    if health.get("problems"):
        v["V5"] += health["problems"]
    elif not health:
        v["V5"].append("no health record")

    # V6: the decision arms and the capacity rows (C1 with a steady footprint, an accepted C2 row).
    for name, r in (("stock", rs), ("compact", rc)):
        for a in DECISION_ARMS[name]:
            if a not in (r.get("arms") or {}):
                v["V6"].append(f"{name}: {a} did not finish")
        cap = (r.get("capacity") or {}).get("rows", {})
        for L in (r.get("capacity") or {}).get("lengths", []):
            row = cap.get(str(L), {})
            if not (row.get("footprint") or {}).get("blocks") or accepted(row.get("c2")) is None:
                v["V6"].append(f"{name}: no steady footprint or accepted batch at {L}")
        if not cap:
            v["V6"].append(f"{name}: no capacity phase")
        if r.get("status") != "ok":
            v["V6"].append(f"{name}: status {r.get('status')}")

    # V7: GROSS breaks the answers (if it ran).
    gross = None
    if graded.get("GROSS"):
        ids = list(graded["GROSS"])
        acc = lambda g, kind: np.mean([g[i]["correct"] for i in ids if i in g and i.startswith(kind)] or [np.nan])  # noqa: E731
        gross = {k: (float(acc(graded["COMPACT"], p)), float(acc(graded["GROSS"], p))) for k, p in
                 (("gsm8k", "G"), ("needle", "N"))}
        if not (gross["gsm8k"][0] - gross["gsm8k"][1] >= GROSS_GSM_DROP
                or gross["needle"][0] - gross["needle"][1] >= GROSS_NEEDLE_DROP):
            v["V7"].append(f"GROSS did not break the answers: {gross}")

    # V8: no failed allocation or preemption in the decision arms (the same schedule in both engines).
    for name, r in (("stock", rs), ("compact", rc)):
        for a in DECISION_ARMS[name]:
            c = ((r.get("arms") or {}).get(a) or {}).get("counters", {})
            if c.get("alloc_fail", 0) or c.get("preemptions", 0):
                v["V8"].append(f"{a}: {c.get('alloc_fail')} failed allocations, {c.get('preemptions')} preemptions")

    # V9: determinism (DET against STOCK on the DET items), recorded; it sets what EQ can expect.
    det_ids, det_n = ids_equal(outs.get("DET"), outs.get("STOCK"))
    det_arr, det_k = arrays_equal(arr.get("DET", {}), {k: x for k, x in arr.get("STOCK", {}).items()})
    det_ok = det_n > 0 and det_ids == det_n and det_arr == det_k
    if not det_ok:
        v["V9"].append(f"DET differs from STOCK: ids {det_ids}/{det_n}, arrays {det_arr}/{det_k}")

    # EQ: COMPACT against EMULV.
    eq_ids, eq_n = ids_equal(outs.get("COMPACT"), outs.get("EMULV"))
    eq_arr, eq_k = arrays_equal(arr.get("COMPACT", {}), arr.get("EMULV", {}))
    p_s, p_c = S / "arrays" / "probe_VIEWS.npy", C / "arrays" / "probe_VIEWS.npy"
    probe_eq = p_s.exists() and p_c.exists() and bool(np.array_equal(np.load(p_s), np.load(p_c)))
    eq_final = final_changes(graded.get("EMULV", {}), graded.get("COMPACT", {}))
    eq_nll, eq_ci, _ = nll_change(arr.get("EMULV", {}), arr.get("COMPACT", {}), sub)
    if eq_n and eq_ids == eq_n and eq_k and eq_arr == eq_k and probe_eq:
        eq = "EXACT"
    elif (eq_n and eq_final <= NEAR_ANSWERS and all(math.isfinite(x) for x in eq_ci)
          and eq_ci[0] >= -NEAR_NLL and eq_ci[1] <= NEAR_NLL):
        eq = "NEAR"
    else:
        eq = "FAIL"
    # The schedule fingerprints (R-82 F4): a match per phase tells a kernel difference from a schedule difference.
    sch_e = ((rs.get("arms") or {}).get("EMULV") or {}).get("schedule") or {}
    sch_c = ((rc.get("arms") or {}).get("COMPACT") or {}).get("schedule") or {}
    eq_sched = {}
    for p in ("score", "gsm8k", "needle"):
        a, b = (sch_e.get(p) or {}).get("sha"), (sch_c.get(p) or {}).get("sha")
        if a is not None or b is not None:
            eq_sched[p] = a is not None and a == b

    # H1 and the reported answer rows: COMPACT (and EMULV) against STOCK.
    items = [i["id"] for i in sub["gsm8k"] + sub["needle"]]
    st = {}
    for a in ("COMPACT", "EMULV", "NAIVE"):
        if all(i in graded.get(a, {}) and i in graded["STOCK"] for i in items) and graded.get(a):
            x = [graded["STOCK"][i]["correct"] for i in items]
            y = [graded[a][i]["correct"] for i in items]
            d, ci = an56.paired_diff(x, y)
            st[a] = {"acc": float(np.mean(y)), "acc_stock": float(np.mean(x)), "acc_change": d, "acc_ci": ci,
                     "final_changes": final_changes(graded["STOCK"], graded[a]),
                     "text_changes": sum(1 for i in items if graded[a][i]["ids"] != graded["STOCK"][i]["ids"]),
                     "nll_change": nll_change(arr["STOCK"], arr[a], sub)[:2]}
    if "COMPACT" in st:
        lo, hi = st["COMPACT"]["acc_ci"]
        h1 = "KILL1" if hi < 0 else ("GO1" if lo >= -M_ACC and hi <= M_ACC else "INCONCLUSIVE1")
    else:
        h1 = "NA1"

    # CAP, TPUT, OVH (A1): CAP is the ratio of the steady decode footprints (C1). Reported: the prefill peaks, the
    # concurrency the pool holds under the admission rule (1 + (free - peak) / steady), the startup log line, the
    # batches with how the limit showed, the hook cost (EMULV row / STOCK) and the cache cost (C3 / EMULV row).
    cap = {}
    seqs = int((rs.get("engine_kwargs") or {}).get("max_num_seqs", 16))
    log_s, log_c = log_concurrency(S), log_concurrency(C)
    for L in sorted({int(x) for x in ((rs.get("capacity") or {}).get("lengths") or [])}):
        row_s = (rs.get("capacity") or {}).get("rows", {}).get(str(L), {})
        row_c = (rc.get("capacity") or {}).get("rows", {}).get(str(L), {})
        fp_s, fp_c = row_s.get("footprint") or {}, row_c.get("footprint") or {}
        fs, fc = fp_s.get("blocks"), fp_c.get("blocks")
        ps, pc = fp_s.get("blocks_peak"), fp_c.get("blocks_peak")
        conc = [1.0 + (f["free_idle"] - f["blocks_peak"]) / f["blocks"] if f.get("blocks") and f.get("blocks_peak")
                and f.get("free_idle") else None for f in (fp_s, fp_c)]
        a_s, a_c = accepted(row_s.get("c2")), accepted(row_c.get("c2"))
        c3, rep, emu = row_c.get("c3"), row_s.get("c2_repeat"), row_s.get("emulv")
        aa_r = step_ratio(rep, a_s)
        aa = aa_r - 1.0 if aa_r is not None else None
        cap[L] = {"footprint": (fs, fc), "ratio": fs / fc if fs and fc else None,
                  "peak": (ps, pc), "ratio_peak": ps / pc if ps and pc else None,
                  "concurrency": tuple(conc), "ratio_concurrency": conc[1] / conc[0] if all(conc) else None,
                  "batch": (a_s or {}).get("batch"), "batch_compact": (a_c or {}).get("batch"),
                  "limit": (limit_shown(row_s.get("c2"), a_s, seqs), limit_shown(row_c.get("c2"), a_c, seqs)),
                  "tps": ((a_s or {}).get("tokens_per_s"), (a_c or {}).get("tokens_per_s")),
                  "tput": (a_c["tokens_per_s"] / a_s["tokens_per_s"]) if a_s and a_c else None,
                  "ovh": step_ratio(c3, a_s), "hook": step_ratio(emu, a_s), "cache": step_ratio(c3, emu),
                  "aa": aa, "ovh_line": max(OVH_MAX, 1.0 + 3.0 * abs(aa)) if aa is not None else OVH_MAX,
                  "alloc_fail": ((a_s or {}).get("alloc_fail"), (a_c or {}).get("alloc_fail"))}
    lens = sorted(cap)
    ratios = [cap[L]["ratio"] for L in lens]
    if not lens or any(r is None for r in ratios):
        capw = "NA-CAP"
    elif all(cap[L]["ratio"] >= CAP_GO.get(L, CAP_GO[max(CAP_GO)]) for L in lens):
        capw = "GO-CAP"
    elif cap[lens[-1]]["ratio"] < CAP_KILL:
        capw = "KILL-CAP"
    else:
        capw = "PARTIAL-CAP"
    tp = [cap[L]["tput"] for L in lens]
    if not lens or any(t is None for t in tp):
        tpw = "NA-TPUT"
    elif max(tp) >= TPUT_GAIN and min(tp) >= TPUT_FLOOR:
        tpw = "TPUT-GAIN"
    elif max(tp) < TPUT_NONE:
        tpw = "TPUT-NONE"
    else:
        tpw = "TPUT-PARTIAL"
    ov = [cap[L]["ovh"] for L in lens]
    ovw = ("NA-OVH" if not lens or any(o is None for o in ov)
           else ("OVH-OK" if all(cap[L]["ovh"] <= cap[L]["ovh_line"] for L in lens) else "OVH-HIGH"))

    # The decision (A1, R-82 F9: one word for each case; the order below decides).
    bad = {k: x for k, x in v.items() if x and k != "V9"}
    if reh:
        word = "REHEARSAL"
    elif bad:
        word = "INVALID"
    elif eq == "FAIL":
        word = "BUILD-FAIL"
    elif h1 == "KILL1" or capw == "KILL-CAP":
        word = "BUILD-KILL"
    elif h1 == "GO1" and capw == "GO-CAP" and tpw in ("TPUT-GAIN", "TPUT-PARTIAL") and ovw == "OVH-OK":
        word = "BUILD-GO"
    else:
        word = "BUILD-PIVOT"
    return {"word": word, "eq": eq, "eq_detail": {"ids": [eq_ids, eq_n], "arrays": [eq_arr, eq_k], "probe": probe_eq,
                                                  "final_changes": eq_final, "nll_change": eq_nll, "nll_ci": eq_ci,
                                                  "schedule_match": eq_sched},
            "det": {"ids": [det_ids, det_n], "arrays": [det_arr, det_k], "ok": det_ok},
            "h1": h1, "answers": st, "cap": {str(L): x for L, x in cap.items()}, "cap_word": capw, "tput_word": tpw,
            "ovh_word": ovw, "gross": gross, "validity": v, "rehearsal": reh,
            "spec": spec, "kv": {"stock": ks, "compact": kc},
            "log_concurrency": {"stock": log_s, "compact": log_c,
                                "ratio": log_c["x"] / log_s["x"] if log_s and log_c and log_s["x"] else None},
            "predicted_ratio": {str(L): cl.blocks_per_request(L, 32, 16, 5) / cl.blocks_per_request(L, 48, 16, 5)
                                for L in lens}}


def line(s: dict) -> str:
    c = s["answers"].get("COMPACT", {})
    ci = c.get("acc_ci", (float("nan"), float("nan")))
    caps = " ".join(f"R{int(L) // 1024}K {x['ratio']:.3f}" if x["ratio"] else f"R{int(L) // 1024}K NA"
                    for L, x in s["cap"].items())
    tps = " ".join(f"T{int(L) // 1024}K {x['tput']:.3f} ({x['batch']}->{x['batch_compact']})" if x["tput"]
                   else f"T{int(L) // 1024}K NA" for L, x in s["cap"].items())
    ovs = " ".join(f"O{int(L) // 1024}K {x['ovh']:.3f}" if x["ovh"] else f"O{int(L) // 1024}K NA"
                   for L, x in s["cap"].items())
    bad = [k for k, x in s["validity"].items() if x and k != "V9"]       # A2: V9 is recorded, not a failed check
    return (f"E-57: {s['word']} | EQ {s['eq']} | {s['h1']} change {c.get('acc_change', float('nan')):+.4f} "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {s['cap_word']} {caps} | {s['tput_word']} {tps} | {s['ovh_word']} {ovs}"
            f" | failed checks: {', '.join(bad) or 'none'} | V9 recorded: {'differs' if s['validity'].get('V9') else 'same'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE.parent / "gemma4_kv_tasks" / "data_e56.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = summarize(args.res, args.data)
    (args.res / "decision.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    if args.json:
        print(json.dumps(s, indent=1, default=str))
    for k, x in s["validity"].items():
        for p in x:
            print(f"{k}: {p}")
    print(line(s))


if __name__ == "__main__":
    main()
