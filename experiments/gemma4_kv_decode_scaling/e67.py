"""E-67 (README_E67.md): the E-64 decode-node grid with the vLLM CPU offload tier on preempted requests.

One run per mode. Each simulation is timed with both E-65 bracket tables; the load time is added after the simulation
for each link speed, so one pass gives every column. The labels use the E-65 A1 gate and the 5-seed rule.

Usage: e67.py <mode> <out json>   (run with `nice -n 10`; one core; about 35 minutes per mode)
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
import e66  # noqa: E402
import poolsim as ps  # noqa: E402

MODES = {"prompt200": (True, 200e9), "all200": (False, 200e9), "all64": (False, 64e9)}
LINKS = {"L20": 20e9, "L10": 10e9, "Linf": None}
STATS = ("decode_per_step", "pool_bound_share", "seq_bound_share", "preemptions", "recomputed_tokens", "readmits",
         "hits", "hit_tokens", "cpu_evictions", "load_bytes", "store_bytes", "window_decoded")


def charged(dec: float, dps: float, load: float, bw: float | None) -> tuple[float, float]:
    """Decode tokens per second with the serial load charge, and the seconds of the window."""
    t = dec / dps + (load / bw if bw else 0.0)
    return dec / t, t


def run_cell(mode: str, name: str, pool: int, seed: int, models: dict, fits: dict, pf: dict,
             horizon: int = e64.HORIZON, warmup: int = e64.WARMUP) -> dict:
    row = {"mode": mode, "scenario": "D", "workload": name, "pool": pool, "seed": seed}
    for lay in ("stock", "compact"):
        out = ps.run_stream(pool, lay, e64.gen_factory(name, seed), horizon, warmup,
                            time_fns=e66.combined_fns(lay, models, fits, pf), decode_only=True, offload=MODES[mode])
        if out["decode_per_step"] <= 0:
            raise RuntimeError(f"V2: no decoding: {mode} {name} {pool} {seed} {lay}")
        for k in STATS:
            row[f"{k}_{lay}"] = out[k]
        for tag in models:
            for a in em.ARMS:
                row[f"dps_{tag}_{a}_{lay}"] = out[f"decode_per_second_{tag}:{a}"]
    row["r_step"] = row["decode_per_step_compact"] / row["decode_per_step_stock"]
    for link, bw in LINKS.items():
        for tag in models:
            for a in em.ARMS:
                v = {lay: charged(row[f"window_decoded_{lay}"], row[f"dps_{tag}_{a}_{lay}"], row[f"load_bytes_{lay}"], bw)[0]
                     for lay in ("stock", "compact")}
                row[f"r_{tag}_{a}_{link}"] = v["compact"] / v["stock"]
        if bw:
            for lay in ("stock", "compact"):
                t = charged(row[f"window_decoded_{lay}"], row[f"dps_r0625_M_{lay}"], row[f"load_bytes_{lay}"], bw)[1]
                row[f"link_use_{link}_{lay}"] = (row[f"load_bytes_{lay}"] + row[f"store_bytes_{lay}"]) / (t * bw)
    row["included"] = (row["pool_bound_share_stock"] >= e64.MIN_BOUND
                       and row["seq_bound_share_stock"] <= e64.MAX_SEQ_BOUND)
    return row


def labels_for(rows: list[dict], tag: str, link: str) -> dict:
    view = [{**r, **{f"r_{a}": r[f"r_{tag}_{a}_{link}"] for a in em.ARMS}} for r in rows]
    res = em.workload_label(view)
    if link == "L20":                                # the link guard (Section 5)
        res["link_use_max"] = max((r[f"link_use_L20_{lay}"] for r in rows if r["included"]
                                   for lay in ("stock", "compact")), default=0.0)
        if res["label"] == "GO" and res["link_use_max"] > 1:
            res["label"] = "MID"
    return res


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    fits, pf = e66.load()
    models = {t: an.Model(f) for t, f in fits.items()}
    t0 = time.time()
    res: dict = {"mode": mode, "rule": "README_E67.md Section 5", "cells": [],
                 "labels": {t: {l: {} for l in LINKS} for t in e66.TAGS}}
    for name in e64.D_WORK:
        rows = [run_cell(mode, name, pool, seed, models, fits, pf) for seed in e64.SEEDS for pool in e64.POOLS]
        res["cells"].extend(rows)
        key = f"D-{name}"
        for tag in e66.TAGS:
            for link in LINKS:
                res["labels"][tag][link][key] = labels_for(rows, tag, link)
        res["seconds"] = time.time() - t0
        out.write_text(json.dumps(res, indent=1))
        lab = res["labels"]["r0625"]["L20"][key]
        print(f"{mode} {key}: {lab['label']} {lab['summary']['Q_M']} ({res['seconds']:.0f} s)", flush=True)
    res["go_r0625_L20"] = [k for k, v in res["labels"]["r0625"]["L20"].items() if v["label"] == "GO"]
    res["seconds"] = time.time() - t0
    out.write_text(json.dumps(res, indent=1))
    print(f"E-67 {mode}: GO at ratio 0.625, L20: {res['go_r0625_L20'] or 'none'} | {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
