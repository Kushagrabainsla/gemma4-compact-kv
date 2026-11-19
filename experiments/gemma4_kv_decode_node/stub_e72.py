"""E-72 stub (stage S3): a Mac stand-in for one role with the real file protocol (probe_go, q/x<k>.json, d_done) and the
real probe prompts. D reports a C2 window (tokens per step `STUB_C2`, default the E-71 C value 7.50915) and probe tokens
that depend on the prompt only (`STUB_DIFF=1` changes the EV tokens of probe 3); `STUB_LEASE=1` writes one lease line
into the P stdout file to test W3, `STUB_NOLEASE=1` drops the lease-setting line (the W3 positive control),
`STUB_EVFAIL=1` fails the EV probe after the build (C2 still runs)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import run_e72 as r72
from run_e71 import write_json


def run(args, plan: dict, rec: dict) -> None:
    rec.update(gpu="NVIDIA A100-SXM4-80GB", engine_seconds=1.0, mode="VIEWS", hooks={"hooked": [5], "problems": []},
               kv={"layout": "LBHNC", "pool_blocks": 6_144}, spec={"applied": r72.ARMS[args.arm] == "compact"})
    d = Path(args.dir)
    q = d / "q"
    q.mkdir(exist_ok=True)
    probes = r72.load_probes(args.data)
    if args.role == "p":
        if os.environ.get("STUB_NOLEASE") != "1":
            print("INFO [startup.py:96] non-default args: {'kv_connector_extra_config': {'ucx_backend_params': {}, "
                  "'kv_lease_duration': 600}}", flush=True)
        while not (d / "probe_go").exists():
            if (d / "d_done").exists():
                return
            time.sleep(0.01)
        for k in range(len(probes)):
            write_json(q / f"x{k}.json", {"k": k, "params": {"remote_block_ids": [[k]]}})
        rec["probe_p"] = [{"k": k, "first_token": [probes[k]["ids"][-1]]} for k in range(len(probes))]
        if os.environ.get("STUB_LEASE") == "1":
            print("WARNING [base_worker.py:3402] Releasing expired KV blocks for request p3 which were retrieved by 0 "
                  "remote worker(s) before lease expired.", flush=True)
        return
    if plan["window"]:
        tps = float(os.environ.get("STUB_C2", "7.50915"))
        rec["window"] = {"work_steps": plan["window"], "seconds": 0.074 * plan["window"], "decode_tokens": tps * 100,
                         "decode_tok_s": tps / 0.074, "tokens_per_step": tps, "running_per_step": tps,
                         "saturated_share": 0.999, "preemptions": 6, "added": 79, "early_finishes": 0,
                         "completed": 63, "q_depth_mean": 1.3}
    if args.arm == "EV" and os.environ.get("STUB_EVFAIL") == "1":   # the probe fails after the build
        raise RuntimeError("stub: EV probe failure")
    (d / "probe_go").write_text("go\n")
    rec["probe"] = []
    for k, it in enumerate(probes):
        while not (q / f"x{k}.json").exists():
            time.sleep(0.01)
        toks = [it["ids"][-1] % 1_000 + j for j in range(5)]
        if args.arm == "EV" and k == 3 and os.environ.get("STUB_DIFF") == "1":
            toks[2] += 1
        rec["probe"].append({"k": k, "id": it["id"], "tokens": toks, "logprobs": [-0.5] * len(toks),
                             "text": it["gold"], "finish_reason": "length", "gold_in_text": True, "seconds": 0.1})
