"""E-71 A1 (after R-118): the frozen prediction intervals. For D3H seeds 0 to 4 and the backlog depths 2, 3 and 8 at the
usable D pool (6,143 blocks), window [2,000, 22,000) work steps, the E-67 model (all200, L20, tag `meas`) gives:
- r_step: compact over stock decode tokens per step (time-free; the primary statistic, README A1);
- Q_graph: the throughput ratio with the E-68 table (CUDA-graph kernels);
- Q_eager: the same with the eager costs of E-59 (both layouts +1.5 ms per step; compact a further +2.2 ms).
Requests with prompt + output above MAX_LEN (the engine's max_model_len) are skipped, in the model as in the engine.

Usage: e71_interval.py <out json>   (nice 10, one core)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e71_predict as ep  # noqa: E402

POOL, WARMUP, HORIZON, MAX_LEN = 6_143, 2_000, 22_000, 144_000         # R-119 F4: the engine limit
DEPTHS, SEEDS = (2, 3, 8), (0, 1, 2, 3, 4)
EAGER_COMMON, EAGER_COMPACT = 0.0015, 0.0022                 # seconds per work step (E-59 eager rows; R-118 B2)


def main() -> None:
    out = Path(sys.argv[1])
    ep.e64.draw = ep.draw
    orig_gen = ep.e64.gen_factory

    def gen_factory(name, seed):
        g = orig_gen(name, seed)
        return lambda: ((p, m) for p, m in g() if p + m <= MAX_LEN)
    ep.e64.gen_factory = gen_factory
    orig_run = ep.e67.ps.run_stream
    fits, pf = ep.e66.load()
    fits["meas"] = json.loads(ep.FIT.read_text())["fit"]
    models = {t: ep.an.Model(f) for t, f in fits.items()}
    res: dict = {"pool": POOL, "warmup": WARMUP, "horizon": HORIZON, "max_len": MAX_LEN, "cells": []}
    t0 = time.time()
    for k in DEPTHS:
        ep.e67.ps.run_stream = lambda *a, k=k, **kw: orig_run(*a, keep_waiting=k, **kw)
        for seed in SEEDS:
            row = ep.e67.run_cell("all200", "D3H", POOL, seed, models, fits, pf, horizon=HORIZON, warmup=WARMUP)
            n = HORIZON - WARMUP
            t = {lay: row[f"window_decoded_{lay}"] / row[f"dps_meas_M_{lay}"] for lay in ("stock", "compact")}
            dec = {lay: row[f"window_decoded_{lay}"] for lay in ("stock", "compact")}
            q_eager = (dec["compact"] / (t["compact"] + n * (EAGER_COMMON + EAGER_COMPACT))) / \
                      (dec["stock"] / (t["stock"] + n * EAGER_COMMON))
            res["cells"].append({"depth": k, "seed": seed, "r_step": row["r_step"], "Q_graph": row["r_meas_M_L20"],
                                 "Q_eager": q_eager, "preempt_stock": row["preemptions_stock"],
                                 "preempt_compact": row["preemptions_compact"],
                                 "step_ms_stock": 1e3 * t["stock"] / n, "step_ms_compact": 1e3 * t["compact"] / n})
        ep.e67.ps.run_stream = orig_run
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"depth {k}: done ({time.time() - t0:.0f} s)", flush=True)
    for key in ("r_step", "Q_graph", "Q_eager"):
        vals = [c[key] for c in res["cells"]]
        res[f"{key}_interval"] = [min(vals), max(vals)]
        res[f"{key}_seed0_depth3"] = next(c[key] for c in res["cells"] if c["seed"] == 0 and c["depth"] == 3)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print({k: v for k, v in res.items() if k.endswith(("interval", "depth3"))})


if __name__ == "__main__":
    main()
