"""E-68 part 2, EXPLORATORY (after S6 attempt 3; the user: "Yes, run part 2").

The frozen word of the A100 SXM run is INVALID (V3 at 6 G points, a slow first round), so the frozen path
(run_e68.sh, e68.load, e65_model.main) refuses to run part 2. This file runs the same part 2 code on the same
`fit.json` and labels every output EXPLORATORY. It decides nothing; the frozen word stays INVALID.

The frozen code is not changed: this file calls e68.run_mode, e68.v8, e68.word, e65_model.run_cell and
e65_model.workload_label. Only the GPU-word gate is skipped, and the skip is written into each output.

Usage: explore_e68.py fit  <amended fit.json> <out fit_explore.json>
       explore_e68.py none <fit_explore.json> <prefill_fit.json> <out json>
       explore_e68.py mode <fit_explore.json> <prompt200 | all200> <out json>
       explore_e68.py word <e68_prompt200.json> <e68_all200.json> <none json> <fit_explore.json>
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
import e65_model  # noqa: E402
import e66  # noqa: E402
import e68  # noqa: E402

LABEL = "EXPLORATORY"
NOTE = "E-68 S6 attempt 3 (A100 SXM): frozen word INVALID (V3); part 2 run without the GPU-word gate; decides nothing"


def explore_fit(fit: dict) -> dict:
    out = json.loads(json.dumps(fit))
    out.update(frozen_word=fit["word"], frozen_failed=fit["failed"], word=LABEL, exploratory=NOTE)
    return out


def check(gpu: dict) -> None:
    if gpu.get("word") != LABEL or gpu.get("frozen_word") is None:
        raise SystemExit("explore_e68: give a fit_explore.json made by `explore_e68.py fit`")


def run_none(gpu: dict, pf: dict, out: Path, names=None, seeds=e64.SEEDS, pools=e64.POOLS, **kw) -> dict:
    """The `none` labels (no offload) under the measured table: the loop of e65_model.main without its word gate.
    names, seeds, pools and kw (horizon, warmup) are for the tests only; the run uses the E-64 grid."""
    model = an.Model(gpu["fit"])
    t_prof = gpu["fit"]["t_other_profile_s"]
    t0 = time.time()
    res: dict = {"rule": "README Section 7", "gpu_word": LABEL, "exploratory": NOTE, "cells": [], "labels": {}}
    for scenario, work in names or (("U", e64.U_WORK), ("D", e64.D_WORK)):
        for name in work:
            rows = [e65_model.run_cell(scenario, name, pool, seed, model, pf, t_prof, **kw) for seed in seeds
                    for pool in pools]
            res["cells"].extend(rows)
            res["labels"][f"{scenario}-{name}"] = e65_model.workload_label(rows, seeds=seeds)
            res["seconds"] = time.time() - t0
            out.write_text(json.dumps(res, indent=1))
            lab = res["labels"][f"{scenario}-{name}"]
            print(f"{LABEL} {scenario}-{name}: {lab['label']} {lab['summary']['Q_M']} ({res['seconds']:.0f} s)",
                  flush=True)
    res["word"] = f"{LABEL} | " + " | ".join(f"{k} {v['label']}" for k, v in res["labels"].items())
    out.write_text(json.dumps(res, indent=1))
    return res


def run_mode(gpu: dict, mode: str, out: Path, **kw) -> dict:
    fits, pf = e66.load()
    fits["meas"] = gpu["fit"]
    res = e68.run_mode(mode, fits, pf, out, **kw)
    res.update(exploratory=NOTE, gpu_word=LABEL)
    out.write_text(json.dumps(res, indent=1))
    return res


def word(p: dict, a: dict, n: dict, gpu: dict) -> dict:
    """The E-68 word rule (Section 7) on the exploratory data. e68.word refuses a non-VALID GPU word, so it gets a
    copy marked VALID; the result is then labelled EXPLORATORY and keeps the frozen word next to it."""
    e67p, e67a = (json.loads((HERE / "results/e67" / f"e67_{m}.json").read_text()) for m in ("prompt200", "all200"))
    res = e68.word(p, a, n, dict(gpu, word="VALID"), e68.v8(p, a, e67p, e67a))
    res.update(word=f"{LABEL} {res['word']}", frozen_word=gpu["frozen_word"], frozen_failed=gpu["frozen_failed"],
               exploratory=NOTE)
    return res


def main() -> None:
    cmd, args = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    if cmd == "fit":
        res = explore_fit(json.loads(args[0].read_text()))
        args[1].write_text(json.dumps(res, indent=1) + "\n")
        print(f"{LABEL} fit: frozen word {res['frozen_word']} {res['frozen_failed']}")
        return
    gpu = json.loads(args[0 if cmd in ("none", "mode") else 3].read_text())
    check(gpu)
    if cmd == "none":
        res = run_none(gpu, json.loads(args[1].read_text()), args[2])
        print(f"{res['word']} | {res['seconds']:.0f} s")
    elif cmd == "mode":
        mode = str(args[1])
        if mode not in ("prompt200", "all200"):
            raise SystemExit(f"explore_e68: mode {mode} is not in the plan")
        res = run_mode(gpu, mode, args[2])
        print(f"{LABEL} {mode}: GO at the measured table, L20: {res['go_meas_L20'] or 'none'} | {res['seconds']:.0f} s")
    elif cmd == "word":
        p, a, n = (json.loads(x.read_text()) for x in args[:3])
        res = word(p, a, n, gpu)
        seeds = {k: v["all200_meas"][2] for k, v in res.get("table", {}).items()}
        print(f"E-68: {res['word']}, frozen word {res['frozen_word']}, decides nothing | GO prompt200 {res.get('go_prompt200') or 'none'} | "
              f"GO all200 {res.get('go_all200') or 'none'} | seeds GO all200 {seeds} | "
              f"read ratio {res.get('read_ratio')} | Phi {res.get('Phi')}")
        print(json.dumps(res, indent=1))
    else:
        raise SystemExit(f"explore_e68: unknown command {cmd}")


if __name__ == "__main__":
    main()
