"""E-68 part 2 (README_E68.md Sections 4.4 and 7): the E-67 decode-node grid with the CPU offload tier, timed with the
W-M step table measured on the A100 (tag `meas`, from the GPU `fit.json`), next to the two E-65 bracket tables.

The E-67 code runs unchanged (`e67.run_cell`, `e67.labels_for`): this file only adds the measured table as a third
tag and writes the E-68 word. The `none` labels (no offload) under the measured table come from `e65_model.py` with
the same `fit.json` (run_e68.sh).

Usage: e68.py <GPU fit.json> <mode: prompt200 | all200> <out json>   (nice 10, one core, about 50 minutes per mode)
       e68.py --word <e68_prompt200.json> <e68_all200.json> <e65 part 2 json> <GPU fit.json>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import analyze_e65 as an  # noqa: E402
import e64  # noqa: E402
import e66  # noqa: E402
import e67  # noqa: E402

TAGS = ("r1", "r0625", "meas")
BORDER_SEEDS = 4                                 # A1 (R-107 B3): MEAS-BORDERLINE at 4 of 5 seeds GO
OK_GPU_WORDS = ("VALID", "REHEARSAL")            # REHEARSAL only for the stub and the rehearsal (never decides)


A2_TOL = 0.01                                    # A2 as changed by R-108: half the 2% MDE (MEASUREMENT.md Section 1)
A2_SAVING_SHARE = 0.25                           # R-108: an H error must also stay below 25% of the modelled saving
TIE = (3, 32_768)


def rng(xs: list[float]) -> float:
    return max(xs) - min(xs)


def a2(fit_res: dict, bench: dict) -> dict:
    """A2 (as changed by R-108): the bound of an SL or H round spread on the decision quantity, the step ratio
    rho = t_compact / t_stock at that point (all times in ms):
    - SL is timed once and enters both layouts (10 layers each); T_OTHER takes -10 SL(tie). So an SL error e moves
      only the denominator: |d rho| <= (|t_s - t_c| / t_s) x 10 (range SL(point) + range SL(tie)) / t_s.
    - H enters only the compact step (2 layers): |d rho| <= 2 range H / t_s, and 2 range H must also stay below 25% of
      the modelled saving |t_s - t_c|.
    A spread is waived if its bound is below 1%. The G spread rule (3%) is not changed. Returns a copy of the analysis
    with the recomputed word and a report-only drift line of the sequential points; the frozen `fit.json` is kept."""
    out = json.loads(json.dumps(fit_res))
    pts = {(p["batch"], p["length"]): p for p in bench["points"]}
    out["a2_drift_report"] = [
        {"B": p["batch"], "L": p["length"],
         **{f"G_{lo}_last_over_first": round(p["G"][lo][-1] / p["G"][lo][0], 4) for lo in ("S32", "C48")}}
        for p in bench["points"] if p.get("sequential")]
    if "V3" not in out["failed"]:
        out["a2"] = {"applied": False}
        return out
    rows = {(r["B"], r["L"]): r for r in out["table"]}
    tie_sl = rng(pts[TIE]["SL"]) if TIE in pts else 0.0
    keep, waived = [], []
    for item in out["checks"]["V3_failed_points"]:
        if len(item) == 2:                                     # a G point: the frozen rule stands
            keep.append(item)
            continue
        b, ln, k = item
        r = rows[(b, ln)]
        ts, saving = r["t_stock_ms"], abs(r["t_stock_ms"] - r["t_compact_ms"])
        if k == "SL":
            bound = saving / ts * 10 * (rng(pts[(b, ln)]["SL"]) + tie_sl) / ts
            ok = bound < A2_TOL
        else:
            e = 2 * rng(pts[(b, ln)]["H"])
            bound = e / ts
            ok = bound < A2_TOL and e < A2_SAVING_SHARE * saving
        (waived if ok else keep).append([b, ln, k, round(bound, 6)])
    out["a2"] = {"applied": True, "kept": keep, "waived": waived, "tie_sl_range_ms": tie_sl}
    if not keep:
        out["failed"] = [f for f in out["failed"] if f != "V3"]
        out["checks"]["V3"] = True
        if out["word"] == "INVALID" and not out["failed"]:
            out["word"] = "VALID"
    return out


A3_FIRST_ROUND_TOL = 0.01                        # A3: rounds 2 and 3 agree within 1%
A3_RATIO_TOL, A3_ABS_TOL, A3_MAX_WAIVED = 0.03, 0.10, 3          # R-109 items 1 and 2
E58_RATIO = an.E58_READ_MS["C48"] / an.E58_READ_MS["S32"]      # 0.9027


def a3(res: dict, bench: dict) -> dict:
    """A3 (after S6 attempt 1 on an A100 PCIe; the user: "Yes, amend and rerun"). Applied after A2.
    - V4 by ratio (as changed by R-109): the tie read ratio C48 / S32 (medians) within 3% of the E-58 ratio, and each
      read within 10% of its E-58 value (the bench `tie_ok` with E68_TIE_RULE=ratio).
    - V3 for G: a point that fails the 3% spread is waived if it is the tie or an interleaved point (not a sequential
      one), and in each failing layout round 1 is the slowest and rounds 2 and 3 agree within 1% (a warm-up effect;
      the model uses the median of the 3 rounds). At most 3 points are waived; more makes the run INVALID.
    The result is labelled exploratory (the rules changed after the attempt-1 tie data)."""
    out = json.loads(json.dumps(res))
    pts = {(p["batch"], p["length"]): p for p in bench["points"]}
    info: dict = {"exploratory": "A3: cross-host (A100 PCIe); V4 and the G spread rule changed after the attempt-1 tie",
                  "tie_rule_in_bench": bench.get("meta", {}).get("tie_rule")}
    if TIE in pts:
        gs, gc = (st_median(pts[TIE]["G"][lo]) for lo in ("S32", "C48"))
        abs_ok = (abs(2 * gs / an.E58_READ_MS["S32"] - 1) <= A3_ABS_TOL
                  and abs(2 * gc / an.E58_READ_MS["C48"] - 1) <= A3_ABS_TOL)
        info.update(tie_ratio=gc / gs, e58_ratio=E58_RATIO,
                    v4_ratio_ok=abs(gc / gs / E58_RATIO - 1) <= A3_RATIO_TOL and abs_ok,
                    v4_abs_values_ms=out["checks"].get("V4_values_ms"))
        if "V4" in out["failed"] and info["v4_ratio_ok"]:
            out["failed"] = [f for f in out["failed"] if f != "V4"]
            out["checks"]["V4"] = True
    if "V3" in out["failed"]:
        left = out.get("a2", {}).get("kept", out["checks"]["V3_failed_points"]) if out.get("a2", {}).get("applied") \
            else out["checks"]["V3_failed_points"]
        keep, waived = [], []
        for item in left:
            if len(item) != 2:
                keep.append(item)
                continue
            p = pts[tuple(item)]
            ok = tuple(item) == TIE or not p.get("sequential", False)
            for lo in ("S32", "C48"):
                r = p["G"][lo]
                if max(r) / min(r) - 1 > an.SPREAD_G:
                    ok = ok and r[0] == max(r) and max(r[1:]) / min(r[1:]) - 1 <= A3_FIRST_ROUND_TOL
            (waived if ok else keep).append(list(item))
        if len(waived) > A3_MAX_WAIVED:                       # R-109 item 2: a cap on the waiver
            keep, waived = keep + waived, []
        info.update(g_waived=waived, kept=keep)
        if not keep:
            out["failed"] = [f for f in out["failed"] if f != "V3"]
            out["checks"]["V3"] = True
    if out["word"] == "INVALID" and not out["failed"]:
        out["word"] = "VALID"
    out["a3"] = info
    return out


def st_median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def amend(fit_res: dict, bench: dict) -> dict:
    """A2, then A3 (README_E68.md Amendments). A3 applies only to a bench that ran with the ratio tie (the Modal A100
    PCIe runs); a run with the frozen absolute tie (the Runpod A100 SXM run of D-120) gets A2 only."""
    out = a2(fit_res, bench)
    return a3(out, bench) if bench.get("meta", {}).get("tie_rule") == "ratio" else out


def load(fit_path: Path) -> tuple[dict, dict, str]:
    fits, pf = e66.load()
    gpu = json.loads(fit_path.read_text())
    if gpu["word"] not in OK_GPU_WORDS:
        raise SystemExit(f"E-68: INVALID GPU part ({gpu['failed']}); part 2 does not run")
    fits["meas"] = gpu["fit"]
    return fits, pf, gpu["word"]


def run_mode(mode: str, fits: dict, pf: dict, out: Path | None = None, names=e64.D_WORK, seeds=e64.SEEDS,
             pools=e64.POOLS, horizon: int = e64.HORIZON, warmup: int = e64.WARMUP) -> dict:
    models = {t: an.Model(fits[t]) for t in TAGS}
    t0 = time.time()
    res: dict = {"mode": mode, "rule": "README_E68.md Section 7", "cells": [],
                 "labels": {t: {l: {} for l in e67.LINKS} for t in TAGS}}
    for name in names:
        rows = [e67.run_cell(mode, name, pool, seed, models, fits, pf, horizon=horizon, warmup=warmup)
                for seed in seeds for pool in pools]
        res["cells"].extend(rows)
        for tag in TAGS:
            for link in e67.LINKS:
                res["labels"][tag][link][f"D-{name}"] = e67.labels_for(rows, tag, link)
        res["seconds"] = time.time() - t0
        if out:
            out.write_text(json.dumps(res, indent=1))
        lab = res["labels"]["meas"]["L20"][f"D-{name}"]
        print(f"E-68 {mode} D-{name}: meas {lab['label']} {lab['summary']['Q_M']} ({res['seconds']:.0f} s)",
              flush=True)
    res["go_meas_L20"] = [k for k, v in res["labels"]["meas"]["L20"].items() if v["label"] == "GO"]
    res["seconds"] = time.time() - t0
    if out:
        out.write_text(json.dumps(res, indent=1))
    return res


def v8(prompt: dict, all_: dict, e67_prompt: dict, e67_all: dict) -> list[str]:
    """V8: the bracket tags of part 2 give the E-67 labels and medians (every workload, link and bracket tag)."""
    bad = []
    for mine, ref in ((prompt, e67_prompt), (all_, e67_all)):
        for tag in ("r1", "r0625"):
            for link in e67.LINKS:
                for k, v in ref["labels"][tag][link].items():
                    m = mine["labels"][tag][link].get(k)
                    if (m is None or m["label"] != v["label"]          # A1 (R-107 F6): floats within 1e-12
                            or any(abs(m["summary"]["Q_M"][s] - v["summary"]["Q_M"][s]) > 1e-12
                                   for s in ("median", "min", "max"))):
                        bad.append(f"{mine['mode']} {tag} {link} {k}")
    return bad


def seeds_go(lab: dict) -> int:
    """The number of seeds whose own label is GO (the per-seed gate of E-65 A1)."""
    return sum(p["label"] == "GO" for p in lab["per_seed"])


def read_ratio(gpu: dict) -> dict:
    """Report only (A1, R-107 F6): the median read ratio G_C / G_S over the points with B x L >= 2^18."""
    big = sorted(r["read_ratio"] for r in gpu.get("table", []) if r["tokens"] >= an.PHI_MIN_TOKENS)
    if not big:
        return {"median": None, "n": 0}
    n = len(big)
    med = big[n // 2] if n % 2 else (big[n // 2 - 1] + big[n // 2]) / 2
    return {"median": med, "min": big[0], "max": big[-1], "n": n}


def word(prompt: dict, all_: dict, none_e65: dict, gpu: dict, v8_failed: list[str] | None = None) -> dict:
    """The frozen E-68 word (Section 7), at the measured table and L20, and the report lines next to it."""
    if gpu["word"] not in OK_GPU_WORDS:
        return {"word": "INVALID", "failed": gpu["failed"]}
    if v8_failed:
        return {"word": "INVALID", "failed": ["V8"], "v8": v8_failed}
    go_p, go_a = prompt["go_meas_L20"], all_["go_meas_L20"]
    meas_a = all_["labels"]["meas"]["L20"]
    near = [k for k, v in meas_a.items() if seeds_go(v) >= BORDER_SEEDS]
    if go_p:
        w = "MEAS-DEFAULT-GO"
    elif go_a:
        w = "MEAS-CONFIG-GO"
    elif near:                                     # A1 (R-107 B3): no GO, but 4 of 5 seeds or more are GO
        w = "MEAS-BORDERLINE"
    else:
        w = "MEAS-NO"
    if gpu["word"] == "REHEARSAL":
        w = f"REHEARSAL | {w}"
    if gpu.get("a3"):
        w = f"{w} (EXPLORATORY, A3: cross-host A100 PCIe)"
    table = {}
    for k in meas_a:
        n = none_e65["labels"].get(k, {})
        table[k] = {"none_meas": [n.get("label"), (n.get("summary") or {}).get("Q_M", {}).get("median")],
                    **{f"{m}_{t}": [r["labels"][t]["L20"][k]["label"],
                                    r["labels"][t]["L20"][k]["summary"]["Q_M"]["median"],
                                    seeds_go(r["labels"][t]["L20"][k])]
                       for m, r in (("prompt200", prompt), ("all200", all_)) for t in TAGS}}
    return {"word": w, "go_prompt200": go_p, "go_all200": go_a, "borderline_all200": near, "Phi": gpu.get("Phi"),
            "read_ratio": read_ratio(gpu), "table": table,
            "link_use_max": max(v["link_use_max"] for r in (prompt, all_) for v in r["labels"]["meas"]["L20"].values())}


def main() -> None:
    if sys.argv[1] == "--word":
        p, a, n, g = (json.loads(Path(x).read_text()) for x in sys.argv[2:6])
        e67p, e67a = (json.loads((HERE / "results/e67" / f"e67_{m}.json").read_text()) for m in ("prompt200", "all200"))
        res = word(p, a, n, g, v8(p, a, e67p, e67a))
        seeds = {k: v["all200_meas"][2] for k, v in res.get("table", {}).items()}
        print(f"E-68: {res['word']} | GO prompt200 {res.get('go_prompt200') or 'none'} | GO all200 "
              f"{res.get('go_all200') or 'none'} | seeds GO all200 {seeds} | read ratio {res.get('read_ratio')} | "
              f"Phi {res.get('Phi')}")
        print(json.dumps(res, indent=1))
        return
    if sys.argv[1] == "--amend":                     # A2 and A3: e68.py --amend <bench dir> <out json>
        d = Path(sys.argv[2])
        res = amend(json.loads((d / "fit.json").read_text()), json.loads((d / "bench.json").read_text()))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
        print(f"E-68 A2+A3: word {res['word']} | failed {res['failed'] or 'none'} | A2 {res['a2']} | "
              f"A3 {res.get('a3', 'not applied (absolute tie)')}")
        return
    if sys.argv[1] == "--ratio":                     # report only, before part 2 (A1, R-107 F6)
        print(f"E-68 read ratio at B x L >= 2^18: {read_ratio(json.loads(Path(sys.argv[2]).read_text()))}")
        return
    fit_path, mode, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    if mode not in ("prompt200", "all200"):
        raise SystemExit(f"E-68: mode {mode} is not in the plan")
    fits, pf, _ = load(fit_path)
    res = run_mode(mode, fits, pf, out)
    print(f"E-68 {mode}: GO at the measured table, L20: {res['go_meas_L20'] or 'none'} | {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
