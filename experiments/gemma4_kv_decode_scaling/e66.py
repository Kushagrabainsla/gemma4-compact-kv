"""E-66 (README_E66.md): the E-64 grid with a prefix cache on the re-admission of preempted requests.

One run per mode (engine, global, perfect). Each simulation is timed with both E-65 bracket tables (read ratio 1.0 and
0.625), so one pass gives both columns. The labels use the E-65 A1 gate and the 5-seed rule.

Usage: e66.py <mode> <out json>   (run with `nice -n 10`; one core; about 40 minutes per mode)
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
import e65_model as em  # noqa: E402
import poolsim as ps  # noqa: E402

TAGS = {"r1": "fit_r1.0.json", "r0625": "fit_r0.625.json"}
STATS = ("decode_per_step", "pool_bound_share", "seq_bound_share", "preemptions", "recomputed_tokens", "readmits",
         "hits", "hit_tokens")


def load() -> tuple[dict, dict]:
    fits = {t: json.loads((HERE / "results/bracket" / f).read_text())["fit"] for t, f in TAGS.items()}
    return fits, json.loads((HERE / "prefill_fit.json").read_text())


def combined_fns(layout: str, models: dict, fits: dict, pf: dict) -> dict:
    out = {}
    for tag, model in models.items():
        for arm, fn in em.time_fns(layout, model, pf, fits[tag]["t_other_profile_s"]).items():
            out[f"{tag}:{arm}"] = fn
    return out


def run_cell(mode: str, scenario: str, name: str, pool: int, seed: int, models: dict, fits: dict, pf: dict,
             horizon: int = e64.HORIZON, warmup: int = e64.WARMUP) -> dict:
    row = {"mode": mode, "scenario": scenario, "workload": name, "pool": pool, "seed": seed}
    for lay in ("stock", "compact"):
        out = ps.run_stream(pool, lay, e64.gen_factory(name, seed), horizon, warmup,
                            time_fns=combined_fns(lay, models, fits, pf), decode_only=(scenario == "D"),
                            prefix_cache=mode)
        if out["decode_per_step"] <= 0:
            raise RuntimeError(f"V2: no decoding: {mode} {scenario} {name} {pool} {seed} {lay}")
        for k in STATS:
            row[f"{k}_{lay}"] = out[k]
        for tag in models:
            for a in em.ARMS:
                row[f"dps_{tag}_{a}_{lay}"] = out[f"decode_per_second_{tag}:{a}"]
    row["r_step"] = row["decode_per_step_compact"] / row["decode_per_step_stock"]
    for tag in models:
        for a in em.ARMS:
            row[f"r_{tag}_{a}"] = row[f"dps_{tag}_{a}_compact"] / row[f"dps_{tag}_{a}_stock"]
    row["included"] = (row["pool_bound_share_stock"] >= e64.MIN_BOUND
                       and row["seq_bound_share_stock"] <= e64.MAX_SEQ_BOUND)
    return row


def labels_for(rows: list[dict], tag: str) -> dict:
    view = [{**r, **{f"r_{a}": r[f"r_{tag}_{a}"] for a in em.ARMS}} for r in rows]
    return em.workload_label(view)


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    fits, pf = load()
    models = {t: an.Model(f) for t, f in fits.items()}
    t0 = time.time()
    res: dict = {"mode": mode, "rule": "README_E66.md Section 5", "cells": [], "labels": {t: {} for t in TAGS}}
    for scenario, names in (("U", e64.U_WORK), ("D", e64.D_WORK)):
        for name in names:
            rows = [run_cell(mode, scenario, name, pool, seed, models, fits, pf) for seed in e64.SEEDS
                    for pool in e64.POOLS]
            res["cells"].extend(rows)
            key = f"{scenario}-{name}"
            for tag in TAGS:
                res["labels"][tag][key] = labels_for(rows, tag)
            res["seconds"] = time.time() - t0
            out.write_text(json.dumps(res, indent=1))
            lab = res["labels"]["r0625"][key]
            print(f"{mode} {key}: {lab['label']} {lab['summary']['Q_M']} ({res['seconds']:.0f} s)", flush=True)
    res["go_r0625"] = [k for k, v in res["labels"]["r0625"].items() if v["label"] == "GO"]
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print(f"E-66 {mode}: GO at ratio 0.625: {res['go_r0625'] or 'none'} | {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
