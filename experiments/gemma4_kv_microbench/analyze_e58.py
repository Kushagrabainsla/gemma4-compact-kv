"""E-58 frozen analysis (README Sections 5 to 7): the validity checks V1 to V5, the cache words, the hook word, the
closure rows against E-57, and the decision word.

Usage: analyze_e58.py --res <dir with bench.json> [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DECIDE = "D32"                    # the shape with the E-57 split (32K: hooks +2.55 ms, cache +2.09 ms per step)
E57 = {"hook": 2.55, "cache": 2.09}
T_CACHE = 0.5                     # ms per 10-layer step: 25% of the E-57 cache cost (R-83 Section 8)
T_HOOK_CPU, T_HOOK_GPU = 1.0, 0.5 # A1 (R-84 B1): four disjoint hook words from the CPU and the graph GPU time
T_EXPLAINS = 2.0                  # A1: the flag "explains E-57" (the hook CPU time near the E-57 +2.55 ms)
T_KILL = 1.5                      # ms per step: a compact-read cost above most of the E-57 cache cost, at both shapes
V1_TOL = 2e-3                     # A1 (R-84 B3): bit-identical, or at most 0.002 (about 2% of the output scale)
V4_SPREAD = 0.03                  # A1 (R-84 B3): the round spread (max / min - 1) of a read arm at the decision shape
V3_LOOP = 0.5                     # A1 (R-84 F3): every CPU issue loop at most half of the GPU sleep
VLLM = "0.33.0rc1.dev229+g3966924d5"
LAYOUTS = ("S32", "S48", "C32", "C48")


def med(xs):
    xs = [x for x in xs or [] if x is not None]
    return statistics.median(xs) if xs else None


def diff(a, b):
    return a - b if a is not None and b is not None else None


def shape_summary(sh: dict) -> dict:
    """The per-step (10-layer) differences of one shape, in ms."""
    r = {lo: med(sh.get("R", {}).get(lo)) for lo in LAYOUTS}
    spread = {lo: (max(v) / min(v) - 1.0) if v and min(v) > 0 else None for lo, v in sh.get("R", {}).items()}
    w, p, h, x = sh.get("W", {}), sh.get("P", {}), sh.get("H", {}), sh.get("X", {})
    pc = {k: v.get("cpu") for k, v in p.items()}
    hv, hn = h.get("VIEWS", {}), h.get("NONE", {})
    out = {"read": r, "spread": spread,
           "views": diff(r["C32"], r["S32"]), "block_compact": diff(r["C48"], r["C32"]),
           "block_stock": diff(r["S48"], r["S32"]), "read_total": diff(r["C48"], r["S32"]),
           "write": diff(w.get("C48"), w.get("S32")),
           "wrapper_cache": diff(pc.get("C48 wrapped"), pc.get("S32 wrapped")),
           "wrapper_total": diff(pc.get("C48 wrapped"), pc.get("C48 raw")),
           "wrapper_stock": diff(pc.get("S32 wrapped"), pc.get("S32 raw")),
           "hook_cpu": diff(hv.get("cpu"), hn.get("cpu")),
           "hook_eager_event": diff(hv.get("gpu"), hn.get("gpu")),       # CPU-bound in eager, not a GPU time (F2)
           "hook_graph": diff(hv.get("graph"), hn.get("graph")),
           "hook_dispatch_cpu": diff(hv.get("cpu"), h.get("EAGER_FN", {}).get("cpu")),
           "compiled": {k: h.get("COMPILED_FN", {}).get(k) for k in ("cpu", "gpu", "graph")},
           "kernels": {k: v.get("kernels") for k, v in list(h.items()) + [("X " + a, b) for a, b in x.items()]},
           "closure": {}}
    out["kernel_total"] = (out["read_total"] + out["write"]
                           if out["read_total"] is not None and out["write"] is not None else None)
    for name, a, b in (("hook", "EMULV", "STOCK"), ("cache", "COMPACT", "EMULV")):
        out["closure"][name] = {m: diff(x.get(a, {}).get(m), x.get(b, {}).get(m)) for m in ("cpu", "gpu", "wall",
                                                                                           "graph")}
    out["bandwidth_GBps"] = {lo: (sh.get("bytes_distinct", {}).get(lo, 0) / (r[lo] * 1e6)) if r[lo] else None
                             for lo in LAYOUTS}
    return out


def summarize(res: Path) -> dict:
    rec = json.loads((res / "bench.json").read_text())
    meta, shapes = rec.get("meta", {}), rec.get("shapes", {})
    reh, stub = bool(meta.get("rehearsal")), bool(meta.get("stub"))
    s = {name: shape_summary(sh) for name, sh in shapes.items()}
    v = {k: [] for k in ("V1", "V2", "V3", "V4", "V5")}
    for name, sh in shapes.items():
        for k, x in (sh.get("v1") or {}).items():
            if not x.get("identical") and (x.get("max_abs") is None or x["max_abs"] > V1_TOL):
                v["V1"].append(f"{name} {k}: max difference {x.get('max_abs')}")
        if name == DECIDE and len(sh.get("v1") or {}) < 6:
            v["V1"].append(f"{name}: {len(sh.get('v1') or {})} of 6 output checks")
        bound = V3_LOOP * sh["sleep_ms"] if sh.get("sleep_ms") else None
        for group in ("P", "H", "X"):
            for k, x in (sh.get(group) or {}).items():
                if x.get("busy") is False:
                    v["V3"].append(f"{name} {group} {k}: the GPU was idle before the issue loop ended")
                if bound is not None and x.get("loop_max_ms") is not None and x["loop_max_ms"] > bound:
                    v["V3"].append(f"{name} {group} {k}: a CPU loop of {x['loop_max_ms']:.1f} ms (bound {bound:.1f})")
        if name == DECIDE:
            for lo, sp in s[name]["spread"].items():
                if sp is None or sp > V4_SPREAD:
                    v["V4"].append(f"{name} R {lo}: round spread {sp}")
            need = ["v1"] + [f"R {lo} round 0" for lo in LAYOUTS] + ["W S32", "W C48", "P S32 wrapped",
                                                                   "P C48 wrapped", "H NONE", "H VIEWS"]
            for e in sh.get("errors", []):
                if e["arm"] in need or e["arm"].startswith("X "):
                    v["V5"].append(f"{name} {e['arm']}: {e['error'][:120]}")
    if DECIDE not in shapes:
        v["V5"].append(f"no {DECIDE} shape")
    if rec.get("status") != "ok":
        v["V5"].append(f"status {rec.get('status')}: {rec.get('error', '')[:120]}")
    if not stub and not reh:
        if "A100" not in str(meta.get("gpu")):
            v["V2"].append(f"GPU {meta.get('gpu')}")
        if meta.get("vllm") != VLLM:
            v["V2"].append(f"vLLM {meta.get('vllm')}")

    d = s.get(DECIDE, {})
    cache = []
    if d:
        for word, key in (("CACHE-VIEWS", "views"), ("CACHE-BLOCK", "block_compact"),
                          ("CACHE-WRAPPER", "wrapper_cache")):
            if d.get(key) is not None and d[key] >= T_CACHE:
                cache.append(word)
    cache = cache or (["CACHE-NONE"] if d else ["NA"])
    faster = bool(d.get("kernel_total") is not None and d["kernel_total"] <= -T_CACHE)
    hc, hg = d.get("hook_cpu"), d.get("hook_graph")
    if hc is None or hg is None:
        hook = "HOOK-NA"
    else:                                               # A1 (R-84 B1): four words that exclude each other
        hook = {(True, False): "HOOK-CPU", (False, True): "HOOK-GPU", (True, True): "HOOK-MIXED",
                (False, False): "HOOK-SMALL"}[(hc >= T_HOOK_CPU, hg >= T_HOOK_GPU)]
    explains = bool(hc is not None and hc >= T_EXPLAINS)
    kill = all(s.get(n, {}).get("views") is not None and s[n]["views"] >= T_KILL for n in ("D32", "D64"))
    bad = {k: x for k, x in v.items() if x}
    if reh or stub:
        word = "REHEARSAL"
    elif bad:
        word = "INVALID"
    elif kill:
        word = "KILL"
    elif cache == ["CACHE-NONE"] and hook == "HOOK-SMALL":
        word = "PIVOT"
    elif cache == ["NA"] or hook == "HOOK-NA":
        word = "INVALID"
    else:
        word = "GO"
    return {"word": word, "cache_words": cache, "cache_faster": faster, "hook_word": hook, "kill": kill,
            "hook_explains_e57": explains, "cache_cost_open": cache == ["CACHE-NONE"],
            "scratch_guard": rec.get("scratch_guard"), "validity": v, "shapes": s, "meta": meta, "e57": E57}


def f(x, nd=2):
    return "NA" if x is None else f"{x:+.{nd}f}"


def line(s: dict) -> str:
    d = s["shapes"].get(DECIDE, {})
    c, k = d.get("closure", {}).get("hook", {}), d.get("closure", {}).get("cache", {})
    o = s["shapes"].get("D64", {})
    bad = [k2 for k2, x in s["validity"].items() if x]
    g = s.get("scratch_guard") or {}
    guard = (f"scratch guard: {g.get('guard_values_changed')} values past the engine-size scratch changed"
             if "guard_values_changed" in g else f"scratch guard: {g.get('error', 'not run')}")
    return (f"E-58: {s['word']}{' (the E-57 cache cost stays open)' if s['cache_cost_open'] else ''} | {DECIDE} "
            f"cache: {' '.join(s['cache_words'])}"
            f"{' (compact kernels faster)' if s['cache_faster'] else ''} (views {f(d.get('views'))} ms, block "
            f"{f(d.get('block_compact'))}, wrapper {f(d.get('wrapper_cache'))}, kernels {f(d.get('kernel_total'))}) | "
            f"hooks: {s['hook_word']}{' (explains E-57)' if s['hook_explains_e57'] else ''} (CPU "
            f"{f(d.get('hook_cpu'))} ms, GPU graph {f(d.get('hook_graph'))}) | X hook: wall {f(c.get('wall'))} CPU "
            f"{f(c.get('cpu'))} graph {f(c.get('graph'))}; X cache: wall {f(k.get('wall'))} GPU {f(k.get('gpu'))} "
            f"CPU {f(k.get('cpu'))} graph {f(k.get('graph'))} (E-57: hook +{E57['hook']}, cache +{E57['cache']}) | "
            f"D64: views {f(o.get('views'))}, block {f(o.get('block_compact'))}, kernels {f(o.get('kernel_total'))} "
            f"| {guard} | failed checks: {', '.join(bad) or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = summarize(args.res)
    (args.res / "decision.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    if args.json:
        print(json.dumps(s, indent=1, default=str))
    for key, x in s["validity"].items():
        for p in x:
            print(f"{key}: {p}")
    print(line(s))


if __name__ == "__main__":
    main()
