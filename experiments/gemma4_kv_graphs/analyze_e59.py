"""E-59 frozen analysis (README Sections 5 to 7): the row medians, OG and its band, the secondary numbers and words,
the validity checks V1 to V5, and the decision word.

Usage: analyze_e59.py --res <session results dir with G1 ... E3 folders> [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ENGINES = ("G1", "G2", "G3", "G4", "E1", "E2", "E3")
GRAPH = ("G1", "G2", "G3", "G4")
PLAN = {"G1": ("stock", True, [("STOCK", 3), ("STOCK", 3)]), "G2": ("stock", True, [("EMULV", 3), ("EMULV", 3)]),
        "G3": ("compact", True, [("COMPACT", 3), ("COMPACT", 3), ("COMPACT", 4)]),
        "G4": ("stock", True, [("STOCK", 3)]),
        "E1": ("stock", False, [("STOCK", 3), ("EMULV", 3), ("STOCK", 3)]),
        "E2": ("compact", False, [("COMPACT", 3), ("COMPACT", 3)]),
        "E3": ("stock", False, [("STOCK", 3), ("EMULV", 3), ("STOCK", 3)])}   # A3: E3 mirrors E1
MDE, KILL, V5_BAND, REPLAY_MIN = 0.02, 0.05, 0.03, 0.9
VLLM = "0.33.0rc1.dev229+g3966924d5"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
HOOKED = 10                                   # the global layers of the 31B (the main run)


def load(res: Path) -> dict:
    out = {}
    for e in ENGINES:
        p = res / e / "run_e59.json"
        out[e] = json.loads(p.read_text()) if p.exists() else None
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ratio(a, b):
    return a / b - 1.0 if a is not None and b else None


def clean(row: dict, rec: dict) -> bool:
    return (row.get("end") == "window" and row.get("preemptions") == 0 and row.get("max_decoding", 0) >= row["batch"]
            and row.get("window_steps", 0) >= rec.get("min_window", 100))


def rows_of(rec: dict | None, arm: str, batch: int = 3) -> list[dict]:
    return [r for r in (rec or {}).get("rows", []) if r.get("arm") == arm and r.get("batch") == batch]


def medians(rec, arm, batch=3):
    return [r.get("median_ms") for r in rows_of(rec, arm, batch) if clean(r, rec)]


def spread(xs: list) -> float | None:
    return abs(xs[1] / xs[0] - 1.0) if len(xs) >= 2 and xs[0] else None


def same_tokens(a: dict | None, b: dict | None, need: int) -> tuple[bool, str]:
    """V1: the tokens of each prompt agree over the common part, and the common part has `need` tokens or more."""
    if not a or not b:
        return False, "a row is missing"
    ta, tb = a.get("tokens") or {}, b.get("tokens") or {}
    keys = sorted(set(ta) | set(tb))
    if not keys:
        return False, "no tokens"
    for k in keys:
        x, y = ta.get(k) or [], tb.get(k) or []
        n = min(len(x), len(y))
        if n < need:
            return False, f"prompt {k}: {n} common tokens (need {need})"
        if x[:n] != y[:n]:
            first = next(i for i in range(n) if x[i] != y[i])
            return False, f"prompt {k}: tokens differ from position {first}"
    return True, f"{len(keys)} prompts equal"


def summarize(res: Path) -> dict:
    recs = load(res)
    present = [r for r in recs.values() if r]
    reh = any(r.get("rehearsal") for r in present)
    stub = any(r.get("stub") for r in present)
    v = {k: [] for k in ("V1", "V2", "V3", "V4", "V5")}

    # V4: the graph block ran all its rows, clean; the hooks; the compact spec.
    for e in GRAPH:
        rec = recs[e]
        if not rec:
            v["V4"].append(f"{e}: no record")
            continue
        if rec.get("status") != "ok":
            v["V4"].append(f"{e}: status {rec.get('status')} {rec.get('preflight') or rec.get('errors')}")
        got = [(r.get("arm"), r.get("batch")) for r in rec.get("rows", []) if r.get("batch") == 3]
        if got != [x for x in PLAN[e][2] if x[1] == 3]:   # A1 (R-86 B2): the batch-3 rows decide; batch 4 feeds TG
            v["V4"].append(f"{e}: rows {got}, planned {PLAN[e][2]}")
        for r in rec.get("rows", []):
            if r.get("batch") == 3 and not clean(r, rec):
                v["V4"].append(f"{e} row {r.get('index')}: not clean (end {r.get('end')}, preemptions "
                               f"{r.get('preemptions')}, window {r.get('window_steps')})")
    for e, rec in recs.items():
        if not rec:
            continue
        hooks = rec.get("hooks") or {}
        n = len(hooks.get("hooked") or [])
        if hooks.get("problems") or n == 0 or (not reh and not stub and n != HOOKED):
            v["V4"].append(f"{e}: hooks {n} layers, problems {hooks.get('problems')}")
        if PLAN[e][0] == "compact" and not (rec.get("spec") or {}).get("applied"):
            v["V4"].append(f"{e}: the compact spec did not apply")
        if rec.get("hook_errors"):
            v["V4"].append(f"{e}: hook errors {rec['hook_errors'][:2]}")

    # V2: the setup, the graph settings and the replays.
    for e, rec in recs.items():
        if not rec:
            continue
        comp = rec.get("compilation") or {}
        graphs = PLAN[e][1]
        if bool(rec.get("graphs")) != graphs or rec.get("layout") != PLAN[e][0]:
            v["V2"].append(f"{e}: layout {rec.get('layout')} graphs {rec.get('graphs')}, planned {PLAN[e][:2]}")
        if graphs and (comp.get("cudagraph_mode") != "FULL_DECODE_ONLY" or comp.get("mode") != "NONE"
                       or not {3, 4, 8} <= set(comp.get("capture_sizes", []))):
            v["V2"].append(f"{e}: graph settings {comp}")
        if not graphs and comp.get("cudagraph_mode") != "NONE":
            v["V2"].append(f"{e}: eager settings {comp}")
        for r in rec.get("rows", []):
            w, rp = r.get("window_steps", 0), r.get("replays", 0)
            if graphs and w and rp < REPLAY_MIN * w:         # A1 (R-86 B2): replays run in the async worker
                v["V2"].append(f"{e} row {r.get('index')}: {rp} graph replays in {w} window steps")
            if not graphs and rp:
                v["V2"].append(f"{e} row {r.get('index')}: {rp} graph replays in an eager row")
        if not reh and not stub:
            if "A100" not in str(rec.get("gpu")):
                v["V2"].append(f"{e}: GPU {rec.get('gpu')}")
            if (rec.get("engine") or {}).get("vllm") != VLLM:
                v["V2"].append(f"{e}: vLLM {(rec.get('engine') or {}).get('vllm')}")
            if REVISION not in str(rec.get("snapshot", "")):
                v["V2"].append(f"{e}: the snapshot {rec.get('snapshot')} is not revision {REVISION[:8]}")

    # V3: the scratch rows (recorded per engine; the engine stops before its rows if they fail).
    for e, rec in recs.items():
        if rec and rec.get("status") == "preflight" and any("scratch" in p for p in rec.get("preflight") or []):
            v["V3"].append(f"{e}: {[p for p in rec['preflight'] if 'scratch' in p][:2]}")
        if rec and not rec.get("scratch"):
            v["V3"].append(f"{e}: no scratch record")

    # V1: the hooks inside the graphs (and in eager).
    g3, g2 = rows_of(recs["G3"], "COMPACT"), rows_of(recs["G2"], "EMULV")
    need = (recs["G3"] or {}).get("min_common", 96)
    ok, why = same_tokens(g3[0] if g3 else None, g2[0] if g2 else None, need)
    v1 = {"graph": why}
    if not ok:
        v["V1"].append(f"graph: COMPACT against EMULV: {why}")
    e2, e1 = rows_of(recs["E2"], "COMPACT"), rows_of(recs["E1"], "EMULV")
    if e2 and e1:
        ok, why = same_tokens(e2[0], e1[0], need)
        v1["eager"] = why
        if not ok:
            v["V1"].append(f"eager: COMPACT against EMULV: {why}")
    else:
        v1["eager"] = "not run"

    # The numbers (ms per decode step; row medians).
    g1s, g4s = medians(recs["G1"], "STOCK"), medians(recs["G4"], "STOCK")
    S_g = med(g1s + g4s)
    E_g = med(medians(recs["G2"], "EMULV"))
    c3 = medians(recs["G3"], "COMPACT")
    C_g = med(c3)
    c4 = medians(recs["G3"], "COMPACT", 4)
    OG, HG, KG = ratio(C_g, S_g), ratio(E_g, S_g), ratio(C_g, E_g)
    a_g = max([x for x in (spread(g1s), spread(medians(recs["G2"], "EMULV")), spread(c3)) if x is not None],
              default=None)
    p_g = abs(ratio(med(g4s), med(g1s))) if g4s and g1s else None
    B_g = max([MDE] + [x for x in (a_g, p_g) if x is not None])
    if a_g is None or p_g is None:
        v["V5"].append(f"the noise is not measured (a_g {a_g}, p_g {p_g})")
    elif B_g > V5_BAND:
        v["V5"].append(f"band {B_g:.4f} > {V5_BAND}")

    e1s, e1m = medians(recs["E1"], "STOCK"), medians(recs["E1"], "EMULV")
    e3s, e3m = medians(recs["E3"], "STOCK"), medians(recs["E3"], "EMULV")
    hes = [x for x in (ratio(med(e1m), med(e1s)), ratio(med(e3m), med(e3s))) if x is not None]
    HE = sum(hes) / len(hes) if hes else None
    em = [x for x in (med(e1m), med(e3m)) if x is not None]
    KE = ratio(med(medians(recs["E2"], "COMPACT")), sum(em) / len(em)) if em else None
    p_e = abs(ratio(med(e3s), med(e1s))) if e3s and e1s else None
    a_e = max([x for x in (spread(e1s), spread(medians(recs["E2"], "COMPACT"))) if x is not None], default=None)
    B_e = max([MDE] + [x for x in (a_e, p_e) if x is not None])
    st_e = [x for x in (med(e1s), med(e3s)) if x is not None]
    GS = ratio(S_g, sum(st_e) / len(st_e)) if st_e else None
    TG = (4 / med(c4)) / (3 / S_g) if c4 and S_g else None

    if HE is None or HG is None:
        hook = "NA"
    elif HE < B_e:
        hook = "HOOK-NOREPRO"
    elif HG < B_g:
        hook = "HOOK-FIXED"
    elif HG <= HE / 2:
        hook = "HOOK-REDUCED"
    else:
        hook = "HOOK-STAYS"
    if KE is None or KG is None:
        cache = "NA"
    elif KE < B_e:
        cache = "CACHE-NOREPRO"
    elif KG < B_g:
        cache = "CACHE-EAGER"
    else:
        cache = "CACHE-REAL"
    process = bool((p_e is not None and p_e >= MDE) or (p_g is not None and p_g >= MDE))
    flags = []                                             # A1 (R-86 F3): report words
    if OG is not None and OG <= -B_g:
        flags.append("FASTER")
    if KG is not None and KG <= -B_g:
        flags.append("CACHE-GAIN")
    s_all = g1s + g4s
    weak = bool(c3 and s_all and min(c3) / max(s_all) - 1.0 < 0.0)

    bad = {k: x for k, x in v.items() if x}
    if reh or stub:
        word = "REHEARSAL"
    elif bad or OG is None:
        word = "INVALID"
    elif OG < B_g:
        word = "GO"
    elif OG >= KILL:
        word = "KILL"
    else:
        word = "PIVOT-WEAK" if weak else "PIVOT"

    def r4(x):
        return None if x is None else round(x, 5)

    numbers = {"S_g": r4(S_g), "E_g": r4(E_g), "C_g": r4(C_g), "C4_g": r4(med(c4)), "OG": r4(OG), "HG": r4(HG),
               "KG": r4(KG), "a_g": r4(a_g), "p_g": r4(p_g), "B_g": r4(B_g), "HE": r4(HE), "KE": r4(KE),
               "p_e": r4(p_e), "a_e": r4(a_e), "B_e": r4(B_e), "GS": r4(GS), "TG": r4(TG)}
    report = {}
    for e, rec in recs.items():
        if rec:
            report[e] = [{k: r.get(k) for k in ("arm", "batch", "median_ms", "mean_ms", "slow_share", "cpu_ms",
                                                "replays", "window_steps", "end", "seconds")}
                         for r in rec.get("rows", [])]
    profiles = {f"{e} row {r.get('index')}": r["profile"] for e, rec in recs.items() if rec
                for r in rec.get("rows", []) if r.get("profile")}
    return {"word": word, "hook_word": hook, "cache_word": cache, "process_flag": process, "flags": flags,
            "numbers": numbers,
            "validity": v, "v1": v1, "rows": report, "profiles": profiles,
            "status": {e: (rec or {}).get("status") for e, rec in recs.items()}, "rehearsal": reh, "stub": stub}


def pct(x):
    return "NA" if x is None else f"{100 * x:+.2f}%"


def line(s: dict) -> str:
    n = s["numbers"]
    bad = [k for k, x in s["validity"].items() if x]
    tg = "NA" if n["TG"] is None else f"{n['TG']:.3f}"
    fl = f" ({' '.join(s['flags'])})" if s.get("flags") else ""
    return (f"E-59: {s['word']}{fl} | OG {pct(n['OG'])} (band {pct(n['B_g'])}; S_g {n['S_g']} ms, C_g {n['C_g']} ms) | "
            f"graph HG {pct(n['HG'])} KG {pct(n['KG'])} process and drift p_g {pct(n['p_g'])} | eager HE {pct(n['HE'])} KE "
            f"{pct(n['KE'])} p_e {pct(n['p_e'])} | {s['hook_word']} {s['cache_word']}"
            f"{' (process effect >= 2%)' if s['process_flag'] else ''} | GS {pct(n['GS'])} | TG {tg} | "
            f"failed checks: {', '.join(bad) or 'none'}")


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
