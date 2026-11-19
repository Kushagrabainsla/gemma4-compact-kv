"""E-65 part 2 (README Sections 4.5 and 7): the E-64 grid (the same workloads, pools, seeds, horizon and engine model)
with the measured time model W-M in place of the assumed arms W-A and W-B. Decides the per-workload labels.

Time of one step (seconds) under W-M:
- decode part (if any request decodes): `analyze_e65.Model.step(layout, B, mean context)`;
- prefill part: for each chunk of n tokens that starts at context c0, n / 2,048 x (a + b x (c0 + n / 2) / 1,024),
  with a and b per layout from `prefill_fit.json` (the E-59 traces).
Report-only arms: M_max (the decode part uses the longest context, not the mean), M_pf_equal (the compact prefill
uses the stock fit), M_other_prof (T_OTHER from the E-59 profile), M_gemm (T_OTHER grows 0.4% per request past 3).

Usage: e65_model.py <fit.json> <prefill_fit.json> <out json>   (run with `nice -n 10`; one core; about 1 hour)
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import analyze_e65 as an  # noqa: E402
import e64  # noqa: E402
import poolsim as ps  # noqa: E402

ARMS = ("M", "M_max", "M_pf_equal", "M_other_prof", "M_gemm")
GATE = ("M", "M_pf_equal", "M_gemm")         # A1 (R-101 F1, F2): GO and KILL need all three
GO, KILL = 1.25, 1.10


def prefill_s(chunks, pf: dict) -> float:
    return sum(n / 2_048 * (pf["a_s"] + pf["b_s_per_1024"] * (c0 + n / 2) / 1_024) for n, c0 in chunks)


def time_fns(layout: str, model: an.Model, pf: dict, t_other_prof: float) -> dict:
    pf_own, pf_stock = pf[layout], pf["stock"]

    def dec(d, use_max=False, t_other=None, gemm=False):
        c = d["decode_ctx"]
        if not c:
            return 0.0
        b = len(c)
        lbar = max(c) if use_max else sum(c) / b
        if gemm:
            t_other = model.t_other * (1 + 0.004 * max(b - 3, 0))
        return model.step(layout, b, lbar, t_other)

    return {
        "M": lambda d: dec(d) + prefill_s(d["prefill_chunks"], pf_own),
        "M_max": lambda d: dec(d, use_max=True) + prefill_s(d["prefill_chunks"], pf_own),
        "M_pf_equal": lambda d: dec(d) + prefill_s(d["prefill_chunks"], pf_stock),
        "M_other_prof": lambda d: dec(d, t_other=t_other_prof) + prefill_s(d["prefill_chunks"], pf_own),
        "M_gemm": lambda d: dec(d, gemm=True) + prefill_s(d["prefill_chunks"], pf_own),
    }


def run_cell(scenario: str, name: str, pool: int, seed: int, model, pf, t_prof, horizon=e64.HORIZON,
             warmup=e64.WARMUP) -> dict:
    row = {"scenario": scenario, "workload": name, "pool": pool, "seed": seed}
    for lay in ("stock", "compact"):
        out = ps.run_stream(pool, lay, e64.gen_factory(name, seed), horizon, warmup,
                            time_fns=time_fns(lay, model, pf, t_prof), decode_only=(scenario == "D"))
        if out["decode_per_step"] <= 0:
            raise RuntimeError(f"V2: no decoding: {scenario} {name} {pool} {seed} {lay}")
        for k in ("decode_per_step", "pool_bound_share", "seq_bound_share", "preemptions"):
            row[f"{k}_{lay}"] = out[k]
        for a in ARMS:
            row[f"dps_{a}_{lay}"] = out[f"decode_per_second_{a}"]
    row["r_step"] = row["decode_per_step_compact"] / row["decode_per_step_stock"]
    for a in ARMS:
        row[f"r_{a}"] = row[f"dps_{a}_compact"] / row[f"dps_{a}_stock"]
    row["included"] = (row["pool_bound_share_stock"] >= e64.MIN_BOUND
                       and row["seq_bound_share_stock"] <= e64.MAX_SEQ_BOUND)
    return row


def label(*qs) -> str:
    """GO if every gate Q is at least 1.25; KILL if every gate Q is below 1.10; else MID (NA if one is missing)."""
    if any(q is None for q in qs):
        return "NA"
    if all(q >= GO for q in qs):
        return "GO"
    return "KILL" if all(q < KILL for q in qs) else "MID"


def workload_label(rows: list[dict], seeds=e64.SEEDS) -> dict:
    per_seed = []
    for s in seeds:
        rs = [r for r in rows if r["seed"] == s]
        per_seed.append({"seed": s, **{f"Q_{a}": e64.med(rs, f"r_{a}") for a in ARMS + ("step",)}})
        per_seed[-1]["label"] = label(*(per_seed[-1][f"Q_{a}"] for a in GATE))
    labs = {p["label"] for p in per_seed}
    final = labs.pop() if len(labs) == 1 else "MID"
    summary = {}
    for k in tuple(f"Q_{a}" for a in ARMS) + ("Q_step",):
        v = [p[k] for p in per_seed if p[k] is not None]
        summary[k] = {"median": st.median(v), "min": min(v), "max": max(v)} if v else None
    return {"label": final, "per_seed": per_seed, "summary": summary,
            "excluded": sum(not r["included"] for r in rows)}


def main() -> None:
    fit_res = json.loads(Path(sys.argv[1]).read_text())
    pf = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])
    if fit_res["word"] not in ("VALID", "REHEARSAL", "BRACKET"):
        sys.exit(f"E-65: INVALID GPU part ({fit_res['failed']}); part 2 does not run")
    model = an.Model(fit_res["fit"])
    t_prof = fit_res["fit"]["t_other_profile_s"]
    t0 = time.time()
    res: dict = {"rule": "README Section 7", "gpu_word": fit_res["word"], "cells": [], "labels": {}}
    for scenario, names in (("U", e64.U_WORK), ("D", e64.D_WORK)):
        for name in names:
            rows = [run_cell(scenario, name, pool, seed, model, pf, t_prof) for seed in e64.SEEDS
                    for pool in e64.POOLS]
            res["cells"].extend(rows)
            res["labels"][f"{scenario}-{name}"] = workload_label(rows)
            res["seconds"] = time.time() - t0
            out.write_text(json.dumps(res, indent=1))
            lab = res["labels"][f"{scenario}-{name}"]
            print(f"{scenario}-{name}: {lab['label']} {lab['summary']['Q_M']} ({res['seconds']:.0f} s)", flush=True)
    res["word"] = " | ".join(f"{k} {v['label']}" for k, v in res["labels"].items())
    if fit_res["word"] in ("REHEARSAL", "BRACKET"):
        res["word"] = f"{fit_res['word']} | " + res["word"]
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print(f"E-65: {res['word']} | {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
