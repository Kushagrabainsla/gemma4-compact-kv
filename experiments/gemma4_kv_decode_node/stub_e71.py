"""E-71 stub (stage S3): a Mac stand-in for one engine role with the real file protocol (q/<i>.json, consumed.txt,
d_done) and the real request sequence; D reports a window with a fixed decode rate per layout (compact 1.32 x stock
in tokens per step and in tokens per second; `STUB_RATIO` overrides it, `STUB_AA` changes the S2 rate to test the A/A
rule, `STUB_EARLY` sets early finishes to test V7)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from run_e71 import ARMS, BUF, requests, write_json


def run(args, plan: dict, rec: dict) -> None:
    rec.update(gpu="NVIDIA A100-SXM4-80GB", engine_seconds=1.0,
               kv={"layout": "LBHNC", "pool_blocks": plan["pool_blocks"] or 2_000},
               spec={"applied": ARMS[args.arm] == "compact"}, patched=10 if ARMS[args.arm] == "compact" else 0)
    d = Path(args.dir)
    q = d / "q"
    q.mkdir(exist_ok=True)
    reqs = requests(40, plan["max_model_len"])
    if args.role == "p":
        for i in range(len(reqs)):
            if (d / "d_done").exists():
                break
            while i - (int((d / "consumed.txt").read_text() or 0) if (d / "consumed.txt").exists() else 0) >= BUF:
                if (d / "d_done").exists():
                    rec.update(produced=i)
                    return
                time.sleep(0.01)
            write_json(q / f"{i}.json", {"i": i, "params": {"remote_block_ids": [[1]]}})
        rec.update(produced=len(reqs))
        return
    nxt = 0
    for _ in range(200):
        while nxt < len(reqs) and (q / f"{nxt}.json").exists() and nxt < 12:
            nxt += 1
            (d / "consumed.txt").write_text(str(nxt))
        time.sleep(0.005)
    ratio = float(os.environ.get("STUB_RATIO", "1.32"))
    scale = (ratio if ARMS[args.arm] == "compact" else 1.0) * \
        (float(os.environ.get("STUB_AA", "1.0")) if args.arm == "S2" else 1.0)
    rate = 100.0 * scale
    rec["window"] = {"work_steps": plan["window"], "seconds": 0.05 * plan["window"], "decode_tokens": rate * 100.0,
                     "decode_tok_s": rate, "tokens_per_step": 5.0 * scale, "running_per_step": 5.0 * scale,
                     "saturated_share": 0.99, "preemptions": 3, "all_steps": plan["window"] + plan["warmup"],
                     "added": nxt, "early_finishes": int(os.environ.get("STUB_EARLY", "0")), "completed": 4,
                     "q_depth_mean": 2.0}
    rec["stats"] = {"load_gb_per_s": 24.0}
    rec["lookups"] = {"preempted": 3, "preempted_hits": 1, "preempted_tokens": 1}
    rec["counters"] = {"preemptions": 3, "fix_reads": 1 if ARMS[args.arm] == "compact" else 0}
