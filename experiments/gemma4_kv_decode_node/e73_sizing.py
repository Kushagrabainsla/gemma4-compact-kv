"""E-73 sizing (exploratory, free, before any design is frozen): candidate second operating points for the engine test of
the compact K=V cache. For each D pool (blocks) and workload (prompt median), D3 family, seeds 0 to 4, backlog depth 3,
the E-71 engine window [2,000, 22,000) work steps, the E-67 model (all200, L20, tag `meas`) gives:
- r_step: compact over stock decode tokens per work step (the E-71 primary statistic);
- Q_graph and Q_eager (as `e71_interval.py`);
- running requests per step and preemptions per layout;
- the engine-scaled step time and arm time (the E-71 calibration: engine step / model step, per layout);
- the prefill rate one P engine must give (the E-71 calibration: E-71 C needed about 810 tokens/s at 101.3 decode
  tokens/s; scaled by the model's compact decode rate and the workload's prompt-to-output ratio);
- the start check of `max_model_len` for stock D at this pool (R-118 B1), and the length limit used.
A point is feedable if the needed prefill rate is at most 60% of about 1,800 tokens/s (E-71 P at the start: 9 requests
in about 96 s; heuristic).

Usage: e73_sizing.py <out json> [--mode all200|all69] [--pools 6144,...] [--workloads D3Q,...]   (nice 10, one core)
`all69` is the engine's CPU tier (64 GiB = 68.7 GB, `cpu_bytes_to_use`); `all200` is the E-67 tier (the default).
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e71_predict as ep  # noqa: E402

POOLS = (4_096, 6_144, 8_192)                      # 10, 15 and 20 GiB at 2.5 MiB per 31B block
WORKLOADS = {"D3Q": (8_192, 2_048), "D3H": (16_384, 2_048), "D3": (32_768, 2_048)}
SEEDS, DEPTH, WARMUP, HORIZON = (0, 1, 2, 3, 4), 3, 2_000, 22_000
EAGER_COMMON, EAGER_COMPACT = 0.0015, 0.0022       # E-59 eager costs (as e71_interval.py)
CAL_STEP = {"stock": (69.36 + 71.27) / 2 / 53.886, "compact": 74.16 / 55.501}   # E-71 engine / model step (seed 0)
CAL_PREFILL = 810.0 / 101.25                       # E-71 C: prefill tokens/s per decode token/s (D3H), a record
P_CAPACITY, P_SHARE = 1_800.0, 0.6
SLIDING_BLOCKS = 5 * (-(-(1_023 + 2 * 2_048) // 16) + 1)    # R-118 B1: 5 groups at D budget 2,048 (async)


def max_len(pool: int) -> int:
    return min(144_000, (pool - 1 - SLIDING_BLOCKS) * 32)


def arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    out = Path(sys.argv[1])
    mode = arg("--mode", "all200")
    ep.e67.MODES["all69"] = (False, float(64 * 2**30))   # the engine tier: 64 GiB, all blocks stored
    pools = tuple(int(x) for x in arg("--pools", ",".join(map(str, POOLS))).split(","))
    workloads = {k: WORKLOADS[k] for k in arg("--workloads", ",".join(WORKLOADS)).split(",")}
    ep.e64.draw = ep.draw                               # D3H and D3Q as E-71; D3 is the E-64 generator itself
    orig_gen, orig_run = ep.e64.gen_factory, ep.e67.ps.run_stream
    fits, pf = ep.e66.load()
    fits["meas"] = json.loads(ep.FIT.read_text())["fit"]
    models = {t: ep.an.Model(f) for t, f in fits.items()}
    ep.e67.ps.run_stream = lambda *a, **kw: orig_run(*a, keep_waiting=DEPTH, **kw)
    res: dict = {"note": "exploratory sizing for E-73; window [2,000, 22,000); depth 3; seeds 0 to 4", "mode": mode,
                 "calibration": {"step": CAL_STEP, "prefill_per_decode_D3H": CAL_PREFILL, "p_capacity": P_CAPACITY},
                 "points": []}
    t0 = time.time()
    for pool in pools:
        lim = max_len(pool)
        ep.e64.gen_factory = lambda name, seed, lim=lim: (lambda g=orig_gen(name, seed): (
            (p, m) for p, m in g() if p + m <= lim))
        for name, (p_med, o_med) in workloads.items():
            cells = []
            for seed in SEEDS:
                row = ep.e67.run_cell(mode, name, pool - 1, seed, models, fits, pf, horizon=HORIZON, warmup=WARMUP)
                n = HORIZON - WARMUP
                t = {lay: row[f"window_decoded_{lay}"] / row[f"dps_meas_M_{lay}"] for lay in ("stock", "compact")}
                dec = {lay: row[f"window_decoded_{lay}"] for lay in ("stock", "compact")}
                q_eager = (dec["compact"] / (t["compact"] + n * (EAGER_COMMON + EAGER_COMPACT))) / \
                          (dec["stock"] / (t["stock"] + n * EAGER_COMMON))
                cells.append({"seed": seed, "r_step": row["r_step"], "Q_graph": row["r_meas_M_L20"], "Q_eager": q_eager,
                              "running_stock": dec["stock"] / n, "running_compact": dec["compact"] / n,
                              "preempt_stock": row["preemptions_stock"], "preempt_compact": row["preemptions_compact"],
                              "step_ms_stock": 1e3 * t["stock"] / n, "step_ms_compact": 1e3 * t["compact"] / n,
                              "dps_compact": row["dps_meas_M_compact"],
                              "readmits": {lay: row[f"readmits_{lay}"] for lay in ("stock", "compact")},
                              "hits": {lay: row[f"hits_{lay}"] for lay in ("stock", "compact")}})
            med = lambda k: st.median(c[k] for c in cells)  # noqa: E731
            # the engine decode rate scaled from E-71 C (101.25 tokens/s, model 128.596 at D3H, 6,144), times the
            # prompt tokens per output token of the workload (both lognormal with sigma 0.6, so the medians' ratio)
            engine_dps = 101.25 * med("dps_compact") / 128.596
            need = engine_dps * (p_med / o_med)
            arm_s = {lay: HORIZON * med(f"step_ms_{lay}") / 1e3 * CAL_STEP[lay] for lay in ("stock", "compact")}
            pt = {"pool": pool, "gib": round(pool * 2.5 / 1024, 1), "workload": name, "max_len": lim,
                  "r_step": {"median": med("r_step"), "min": min(c["r_step"] for c in cells),
                             "max": max(c["r_step"] for c in cells), "seed0": cells[0]["r_step"]},
                  "Q_graph_median": med("Q_graph"), "Q_eager_median": med("Q_eager"),
                  "running": {"stock": med("running_stock"), "compact": med("running_compact")},
                  "preemptions_window": {"stock": med("preempt_stock"), "compact": med("preempt_compact")},
                  "engine_step_ms": {lay: med(f"step_ms_{lay}") * CAL_STEP[lay] for lay in ("stock", "compact")},
                  "engine_arm_s": arm_s, "prefill_needed_tok_s": need,
                  "feedable": need <= P_SHARE * P_CAPACITY,
                  "cost_two_arms_usd": (sum(arm_s.values()) + 2 * 250 + 200) * 0.0018476, "cells": cells}
            res["points"].append(pt)
            out.write_text(json.dumps(res, indent=1) + "\n")
            print(f"pool {pool} ({pt['gib']} GiB) {name}: r_step {pt['r_step']['median']:.3f} "
                  f"[{pt['r_step']['min']:.3f}, {pt['r_step']['max']:.3f}] seed0 {pt['r_step']['seed0']:.3f} | "
                  f"running {pt['running']['stock']:.2f} -> {pt['running']['compact']:.2f} | preempt "
                  f"{pt['preemptions_window']['stock']:.0f}/{pt['preemptions_window']['compact']:.0f} | "
                  f"prefill need {need:.0f} tok/s {'ok' if pt['feedable'] else 'TOO HIGH'} | "
                  f"cost ${pt['cost_two_arms_usd']:.2f} | max_len {lim} | {time.time() - t0:.0f} s", flush=True)
    ep.e64.gen_factory, ep.e67.ps.run_stream = orig_gen, orig_run


if __name__ == "__main__":
    main()
