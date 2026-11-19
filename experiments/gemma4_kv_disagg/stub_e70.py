"""E-70 stage 1 stub (stage S3): a Mac stand-in for one engine role, with no vLLM and no GPU. It follows the real file
protocol (p_records.json, d_done) and the real comparison (`nixl_compact.compare`). COMPACT-NOPATCH raises the NIXL
partition assert. `STUB_CORRUPT=1` changes one D hash, so the stub run can show that the byte check finds it."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import nixl_compact as nx
from run_e70 import REF_TOKENS, wait_for, write_json


def _hashes(item_id: str) -> dict:
    return {0: {i: hash((item_id, 0, i)) for i in range(60, 64)}, 5: {i: hash((item_id, 5, i)) for i in range(0, 159)}}


def run(args, rec: dict, items: list[dict]) -> None:
    rec.update(gpu="NVIDIA L40S", engine_seconds=1.0, spec={"applied": args.pair != "STOCK"},
               kv={"layout": "LBHNC"}, patched=[["layer", [1], [1], "bf16", [1]]] if args.pair == "COMPACT" else [])
    if args.pair == "COMPACT-NOPATCH":
        raise AssertionError("KV cache view cannot be partitioned into NIXL regions: layer=stub")
    d = Path(args.dir)
    if args.role == "p":
        rec["ref"] = [{"id": it["id"], "tokens": [1] * REF_TOKENS, "text": it["gold"], "gold_in_text": True}
                      for it in items]
        records = [{"id": it["id"], "request_id": f"{k}", "params": {"remote_block_ids": [[1]]},
                    "hashes": {"groups": _hashes(it["id"]), "prompt": len(it["ids"]), "num_computed": len(it["ids"])}}
                   for k, it in enumerate(items)]
        write_json(d / "p_records.json", {"records": records, "sliding": [0]})
        rec["p_hashed"] = rec["p_params"] = len(records)
        rec["d_done_seen"] = wait_for(d / "d_done", args.wait)
        return
    if not wait_for(d / "p_records.json", args.wait):
        raise RuntimeError("no p_records.json from P")
    pr = json.loads((d / "p_records.json").read_text())
    checks = []
    for k, r in enumerate(pr["records"]):
        dg = {int(g): {int(i): h for i, h in b.items()} for g, b in r["hashes"]["groups"].items()}
        if os.environ.get("STUB_CORRUPT") == "1" and k == 0:
            dg[5][3] += 1
        checks.append({"id": r["id"], "hit": r["hashes"]["prompt"] - 1, "prompt": r["hashes"]["prompt"],
                       "failed_load": False, **nx.compare(r["hashes"]["groups"], dg, set(pr["sliding"]))})
    rec["checks"] = checks
    rec["answers"] = [{"id": it["id"], "tokens": [1] * REF_TOKENS, "text": it["gold"], "gold_in_text": True}
                      for it in items]
    rec["n_complete"] = len(items)
    if args.pair.endswith("2"):                    # stage 2: the E-69 summaries of D
        bad = 1 if os.environ.get("STUB_CORRUPT") == "1" else 0
        rec.update(counters={"fix_reads": 3, "pass_reads": 0, "preemptions": 40},
                   coverage={"median_c": 0.97, "min_c": 0.92, "zero_hit_events": 0, "decode_events": 24,
                             "preempted_events": 40},
                   reload={"checked_reloads": 24, "mismatched_blocks": bad, "a1_strict_mismatch": 5,
                           "recomputed_first_version": 5},
                   stats={"load_gb_per_s": 20.0})
    time.sleep(0.1)
