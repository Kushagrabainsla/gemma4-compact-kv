"""E-73 (README_E73.md): the frozen prediction interval for the second operating point, by the method of
`e71_interval.py`: D3Q (prompt median 8,192, output median 2,048, sigma 0.6), seeds 0 to 4, backlog depths 2, 3 and 8,
the usable D pool (6,143 blocks), window [2,000, 22,000) work steps, the E-67 model (all200, L20, tag `meas`; the
64 GiB tier gives the same values, `e73_tier_check.py`), the engine length limit 144,000. It gives r_step (the primary
statistic), Q_graph and Q_eager (report only), the running requests and the preemptions per cell.

Usage: e73_interval.py <out json>   (nice 10, one core)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e71_predict as ep  # noqa: E402

WORKLOAD = "D3Q"
POOL, WARMUP, HORIZON, MAX_LEN = 6_143, 2_000, 22_000, 144_000
DEPTHS, SEEDS = (2, 3, 8), (0, 1, 2, 3, 4)
EAGER_COMMON, EAGER_COMPACT = 0.0015, 0.0022


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
    res: dict = {"workload": WORKLOAD, "pool": POOL, "warmup": WARMUP, "horizon": HORIZON, "max_len": MAX_LEN,
                 "cells": []}
    t0 = time.time()
    for k in DEPTHS:
        ep.e67.ps.run_stream = lambda *a, k=k, **kw: orig_run(*a, keep_waiting=k, **kw)
        for seed in SEEDS:
            row = ep.e67.run_cell("all200", WORKLOAD, POOL, seed, models, fits, pf, horizon=HORIZON, warmup=WARMUP)
            n = HORIZON - WARMUP
            t = {lay: row[f"window_decoded_{lay}"] / row[f"dps_meas_M_{lay}"] for lay in ("stock", "compact")}
            dec = {lay: row[f"window_decoded_{lay}"] for lay in ("stock", "compact")}
            q_eager = (dec["compact"] / (t["compact"] + n * (EAGER_COMMON + EAGER_COMPACT))) / \
                      (dec["stock"] / (t["stock"] + n * EAGER_COMMON))
            res["cells"].append({"depth": k, "seed": seed, "r_step": row["r_step"], "Q_graph": row["r_meas_M_L20"],
                                 "Q_eager": q_eager, "running_stock": dec["stock"] / n,
                                 "running_compact": dec["compact"] / n, "preempt_stock": row["preemptions_stock"],
                                 "preempt_compact": row["preemptions_compact"]})
        ep.e67.ps.run_stream = orig_run
        print(f"depth {k}: done ({time.time() - t0:.0f} s)", flush=True)
    for key in ("r_step", "Q_graph", "Q_eager"):
        vals = [c[key] for c in res["cells"]]
        res[f"{key}_interval"] = [min(vals), max(vals)]
        res[f"{key}_seed0_depth3"] = next(c[key] for c in res["cells"] if c["seed"] == 0 and c["depth"] == 3)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print({k: v for k, v in res.items() if k.endswith(("interval", "depth3"))})


if __name__ == "__main__":
    main()
