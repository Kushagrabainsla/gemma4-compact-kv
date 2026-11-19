"""E-64: decode-heavy unified node (U) and disaggregated decode node (D). Rule: README_E64.md Section 6 as amended by
A1 (Section 8). Usage: e64.py <out json>   (run with `nice -n 10`; one core; pure Python; saves after each workload)
"""
import itertools
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import poolsim as ps
import sweep

POOLS = [6_727, 8_724, 11_314, 14_672, 19_027]
SEEDS = [0, 1, 2, 3, 4]
HORIZON, WARMUP = 200_000, 40_000                               # A1 B3
MIN_BOUND, MAX_SEQ_BOUND, MIN_POOLS = 0.5, 0.1, 3
G_STOCK, G_COMPACT = 10 * 4 * 1_024 * 2, 10 * 4 * 640 * 2     # global KV bytes per token (10 layers, 4 heads)
S_BYTES = 50 * 16 * 512 * 2                                     # sliding KV bytes per token (50 layers, 16 heads)
T_FIXED, BW, HOOK_COMPACT_B = 0.040, 1.5e12, 0.0004            # A1 F4: +0.4 ms compact hook cost in W-B
U_WORK, D_WORK = ("D1", "D2", "D3", "D4"), ("D1", "D2", "D3", "D4", "M1", "M2")


def draw(rng: random.Random, name: str) -> tuple[int, int]:
    def ln(median, sigma, lo, hi):
        return sweep.lognormal(rng, median, sigma, lo, hi)
    if name == "D1":
        return ln(1_024, 0.8, 256, 131_072), ln(2_048, 0.6, 16, 16_384)
    if name == "D2":
        return ln(8_192, 0.8, 256, 131_072), ln(2_048, 0.6, 16, 16_384)
    if name == "D3":
        return ln(32_768, 0.6, 256, 131_072), ln(2_048, 0.6, 16, 16_384)
    if name == "D4":
        return ln(4_096, 0.8, 256, 131_072), ln(8_192, 0.6, 16, 16_384)
    return sweep.draw(rng, name)                                 # M1, M2 as in E-63


def gen_factory(name: str, seed: int):
    def gen():
        rng = random.Random(f"64-{name}-{seed}")
        return (draw(rng, name) for _ in itertools.count())
    return gen


def step_time(d: dict, gbytes: int, t_prefill: float, bw: float = BW, hook: float = 0.0,
              prefill_only: bool = False) -> float:
    t = 0.0
    if d["decode_ctx"] and not prefill_only:
        t += T_FIXED + hook + sum(gbytes * c + S_BYTES * min(c, 1_024) for c in d["decode_ctx"]) / bw
    t += t_prefill * d["prefill_tokens"] / 2_048
    return t


def time_fns(layout: str) -> dict:
    g = G_COMPACT if layout == "compact" else G_STOCK
    h = HOOK_COMPACT_B if layout == "compact" else 0.0
    return {
        "A": lambda d: step_time(d, g, 0.5),
        "B": lambda d: step_time(d, G_STOCK, 1.0, hook=h),
        "A_bw_lo": lambda d: step_time(d, g, 0.5, bw=0.75e12),
        "A_bw_hi": lambda d: step_time(d, g, 0.5, bw=3e12),
        "B_bw_lo": lambda d: step_time(d, G_STOCK, 1.0, bw=0.75e12, hook=h),
        "B_bw_hi": lambda d: step_time(d, G_STOCK, 1.0, bw=3e12, hook=h),
        "A_prefill": lambda d: step_time(d, g, 0.5, prefill_only=True),
    }


ARMS = ("A", "B", "A_bw_lo", "A_bw_hi", "B_bw_lo", "B_bw_hi")


def run_cell(scenario: str, name: str, pool: int, seed: int) -> dict:
    row = {"scenario": scenario, "workload": name, "pool": pool, "seed": seed}
    for lay in ("stock", "compact"):
        out = ps.run_stream(pool, lay, gen_factory(name, seed), HORIZON, WARMUP, time_fns=time_fns(lay),
                            decode_only=(scenario == "D"))
        if out["decode_per_step"] <= 0:
            raise RuntimeError(f"V2: no decoding: {scenario} {name} {pool} {seed} {lay}")
        for k in ("decode_per_step", "pool_bound_share", "seq_bound_share", "preemptions", "recomputed_tokens",
                  "skipped", "finished"):
            row[f"{k}_{lay}"] = out[k]
        for a in ARMS + ("A_prefill",):
            row[f"dps_{a}_{lay}"] = out[f"decode_per_second_{a}"]
        # The share of W-A time spent on prefill (local re-prefill on a decode node): dps_A / dps_A_prefill.
        row[f"prefill_time_share_{lay}"] = out["decode_per_second_A"] / out["decode_per_second_A_prefill"] \
            if out["decode_per_second_A_prefill"] > 0 else 0.0
    row["r_step"] = row["decode_per_step_compact"] / row["decode_per_step_stock"]
    for a in ARMS:
        row[f"r_{a}"] = row[f"dps_{a}_compact"] / row[f"dps_{a}_stock"]
    row["included"] = (row["pool_bound_share_stock"] >= MIN_BOUND and row["seq_bound_share_stock"] <= MAX_SEQ_BOUND)
    row["recompute_flag"] = max(row["prefill_time_share_stock"], row["prefill_time_share_compact"]) > 0.15 \
        if scenario == "D" else False
    return row


def label(qa, qb) -> str:
    if qa is None or qb is None:
        return "NA"
    if qa >= 1.25 and qb >= 1.25:
        return "GO"
    if qa < 1.10 and qb < 1.10:
        return "KILL"
    return "MID"


def med(rows, key):
    inc = [r[key] for r in rows if r["included"]]
    return st.median(inc) if len(inc) >= MIN_POOLS else None


def workload_label(rows: list[dict]) -> dict:
    per_seed = []
    for s in SEEDS:
        rs = [r for r in rows if r["seed"] == s]
        qa, qb = med(rs, "r_A"), med(rs, "r_B")
        per_seed.append({"seed": s, "Q_A": qa, "Q_B": qb, "label": label(qa, qb),
                         **{f"Q_{a}": med(rs, f"r_{a}") for a in ARMS[2:]}, "Q_step": med(rs, "r_step")})
    labs = {p["label"] for p in per_seed}
    final = labs.pop() if len(labs) == 1 else "MID"
    vals = lambda k: [p[k] for p in per_seed if p[k] is not None]  # noqa: E731
    summary = {k: {"median": st.median(vals(k)), "min": min(vals(k)), "max": max(vals(k))} if vals(k) else None
               for k in ("Q_A", "Q_B", "Q_step") + tuple(f"Q_{a}" for a in ARMS[2:])}
    return {"label": final, "per_seed": per_seed, "summary": summary,
            "recompute_flags": sum(r["recompute_flag"] for r in rows),
            "excluded": sum(not r["included"] for r in rows)}


def checks() -> dict:
    v = {}
    d = {"decode_ctx": [32_768] * 3, "prefill_tokens": 0}
    t_stock, t_compact = step_time(d, G_STOCK, 0.5), step_time(d, G_COMPACT, 0.5)
    v["V7"] = abs(t_stock / 0.05073 - 1) <= 0.30
    v["V7_values_ms"] = [round(1e3 * t_stock, 2), round(1e3 * t_compact, 2), 50.73, 50.52]
    ok = True
    for pool, lay, p in ((8_724, "stock", 32_768), (8_724, "compact", 32_768), (14_672, "compact", 65_536)):
        sim = ps.PoolSim(pool, lay, decode_only=True)
        b = pool // sim.steady(p)
        out = ps.run_saturated(pool, lay, [(p, 256)] * (b + 3), decode_only=True)
        ok &= out["max_decoding"] in (b, b - 1)
    v["V8"] = ok
    # A1 F6: the sliding blocks of steady() against the vLLM formula cdiv(c,16) - (c-1023)//16 + 1 (new slot),
    # within 1 block per sliding group (steady() is the E-57-validated StubEngine rule).
    diffs = []
    for c in (10_000, 32_010, 65_536, 100_003):
        vllm = math.ceil(c / 16) - (c - 1_023) // 16 + 1
        ours = min(math.ceil(c / 16), math.ceil(1_024 / 16) + 1)
        diffs.append(vllm - ours)
    v["V8b"] = all(abs(x) <= 1 for x in diffs)
    v["V8b_diffs"] = diffs
    return v


def main() -> None:
    out = Path(sys.argv[1])
    t0 = time.time()
    res: dict = {"rule": "README_E64.md Sections 6 and 8 (A1)", "checks": checks(), "cells": [], "labels": {}}
    for scenario, names in (("U", U_WORK), ("D", D_WORK)):
        for name in names:
            rows = [run_cell(scenario, name, pool, seed) for seed in SEEDS for pool in POOLS]
            res["cells"].extend(rows)
            res["labels"][f"{scenario}-{name}"] = workload_label(rows)
            res["seconds"] = time.time() - t0
            out.write_text(json.dumps(res, indent=1))
            lab = res["labels"][f"{scenario}-{name}"]
            print(f"{scenario}-{name}: {lab['label']} {lab['summary']['Q_A']} {lab['summary']['Q_B']} "
                  f"({res['seconds']:.0f} s)", flush=True)
    # Pooled, reported only.
    for s in ("U", "D"):
        rows = [r for r in res["cells"] if r["scenario"] == s and r["included"]]
        res[f"pooled_{s}"] = {"Q_A": st.median(r["r_A"] for r in rows), "Q_B": st.median(r["r_B"] for r in rows)}
    gate = res["checks"]["V7"] and res["checks"]["V8"] and res["checks"]["V8b"]
    res["word"] = "INVALID" if not gate else " | ".join(f"{k} {v['label']}" for k, v in res["labels"].items())
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print(f"E-64: {res['word']} | checks {res['checks']} | {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
