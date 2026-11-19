"""E-71 (README Sections 4 and 5, as amended by A1 after R-118): one engine role (P or D) of one arm.

The request sequence is the E-64 generator for D3H, seed 0, with the model's skip rule and the engine length limit (the
frozen prediction uses the same sequence). P prefills the requests in order (`max_tokens` 1, `do_remote_decode`) and
keeps at most BUF prefilled requests that D has not taken, writing each request's transfer parameters to
`q/<i>.json`. D keeps WAIT requests in the not-yet-allocated waiting states (WAITING, PREEMPTED; the model's
`keep_waiting`), adds the next request from the queue with its drawn output length (`ignore_eos`), and counts, per work
step (a step with a non-empty schedule), its decode tokens. The window is the work steps [WARMUP, WARMUP + WINDOW).
D writes `d_run.json` and `d_done`; P stops when it sees `d_done`.

Usage: run_e71.py --role p|d --arm C0|S1|C|S2 --model <dir> --dir <arm dir> [--rehearsal] [--stub]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_disagg"), str(EXP / "gemma4_kv_offload"), str(EXP / "gemma4_kv_engine"),
                str(EXP / "gemma4_kv_model"), str(EXP / "gemma4_kv_capacity_model")]

ARMS = {"C0": "compact", "S1": "stock", "C": "compact", "S2": "stock"}      # A1 (R-118 F1): C0 is a short smoke arm
GIB = 2**30
PORT = {"p": 5600, "d": 5601}
UCX = {"ucx_backend_params": {"ucx_error_handling_mode": "none"}}
# A1 (R-118 B1): the engines start only if one max_model_len request fits the pool (with 2 x the batch budget);
# 145,216 is the largest start for stock D, and 144,000 keeps a margin of 38 blocks (R-119 F4)
MAIN = {"warmup": 2_000, "window": 20_000, "kv_bytes": 15 * GIB, "cpu_bytes": 64 * GIB, "max_model_len": 144_000,
        "p_budget": 1_024, "d_budget": 2_048, "watchdog": 2_250, "pool_blocks": 6_144}   # A3: watchdog 2,250 s
SMOKE = dict(MAIN, warmup=50, window=300, watchdog=400)
REH = {"warmup": 300, "window": 1_500, "kv_bytes": 6 * GIB, "cpu_bytes": 16 * GIB, "max_model_len": 32_768,
       "p_budget": 1_024, "d_budget": 2_048, "watchdog": 300, "pool_blocks": None}
REH_SMOKE = dict(REH, warmup=50, window=300, watchdog=150)
# A1 (R-118 F2): each watchdog (counted from the end of the engine build) is shorter than the arm limit in e71_stage
BUF, WAIT = 2, 3                                  # A1 (R-118 B3): the model's backlog depth 3
STALL_S = 300                                     # A1 (R-118 F2): no new request from P for this long ends the arm
MODEL_POOL = 6_143                                # A1 (R-118 B1): the usable pool (6,144 blocks less the null block)
BASE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.rename(path)


def plan_for(arm: str, rehearsal: bool) -> dict:
    if rehearsal:
        return REH_SMOKE if arm == "C0" else REH
    return SMOKE if arm == "C0" else MAIN


def requests(n: int, max_len: int) -> list[tuple[int, int]]:
    """D3H, seed 0, as the model sees it: the E-64 draw (`random.Random("64-D3H-0")`, `sweep.lognormal`), the
    `poolsim.run_stream` skip rule on the usable pool, and the engine length limit (prompt + output at most max_len)."""
    import poolsim as ps
    import sweep
    rng = random.Random("64-D3H-0")
    probe = ps.PoolSim(MODEL_POOL, "stock", offload=(False, 200e9))
    out: list[tuple[int, int]] = []
    while len(out) < n:
        p = sweep.lognormal(rng, 16_384, 0.6, 256, 131_072)
        m = sweep.lognormal(rng, 2_048, 0.6, 16, 16_384)
        if p + m > max_len or probe.reservation(p + m) > MODEL_POOL - probe.wm:
            continue
        out.append((p, m))
    return out


def prompt_ids(i: int, length: int) -> list[int]:
    seed = int(hashlib.sha256(f"e71|{i}|{length}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).integers(1_000, 100_000, size=length).tolist()


def engine_kwargs(role: str, plan: dict) -> dict:
    kw = {**BASE, "kv_cache_memory_bytes": plan["kv_bytes"], "max_model_len": plan["max_model_len"]}
    if role == "p":
        kw.update(max_num_seqs=4, max_num_batched_tokens=plan["p_budget"],
                  kv_transfer_config={"kv_connector": "NixlConnector", "kv_role": "kv_both",
                                      "kv_connector_extra_config": UCX})
    else:
        kw.update(max_num_seqs=16, max_num_batched_tokens=plan["d_budget"], disable_log_stats=False,
                  prefix_cache_retention_interval=None,
                  kv_transfer_config={"kv_connector": "MultiConnector", "kv_role": "kv_both",
                                      "kv_connector_extra_config": {"connectors": [
                                          {"kv_connector": "NixlConnector", "kv_role": "kv_both",
                                           "kv_connector_extra_config": UCX},
                                          {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                                           "kv_connector_extra_config": {"cpu_bytes_to_use": plan["cpu_bytes"],
                                                                         "offload_prompt_only": False}}]}})
    return kw


def build(layout: str, role: str, model: str, plan: dict, rec: dict):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "LBHNC"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(PORT[role])
    os.environ.setdefault("UCX_NET_DEVICES", "all")
    import torch
    import compact_engine as ce
    import nixl_compact as nx
    ce.install(layout)
    nx.install_capture_and_patch(layout == "compact")     # the E-70 byte-view back-port (compact engines only)
    if role == "d":
        import run_e69
        from run_e70 import MultiStatsSum
        run_e69.HOOK["stats"] = MultiStatsSum(run_e69.HOOK["stats"])
        run_e69.HOOK["phase"] = "e71"                       # the lookup log and stats; no V-RELOAD snapshots
        run_e69.install_hooks(conn=True, corrupt=False)
    from vllm import LLM
    torch.backends.cuda.matmul.allow_tf32 = False
    rec["gpu"] = torch.cuda.get_device_name(0)
    rec["engine_kwargs"] = engine_kwargs(role, plan)
    t0 = time.time()
    try:
        llm = LLM(model=model, tokenizer=model, **rec["engine_kwargs"])
    finally:
        rec["spec"] = ce.STATE["spec"]
        rec["patched"] = len(nx.STATE["patched"])
    rec["engine_seconds"] = round(time.time() - t0, 1)
    rec["kv"] = ce.kv_record(llm)
    splice = ce.ViewsSplice()
    rec["hooks"] = llm.apply_model(lambda m: ce.register(m, splice))[0]
    splice.set_mode("VIEWS" if layout == "compact" else "OFF")
    return llm, ce


def run_p(args, plan: dict, rec: dict) -> None:
    llm, ce = build(ARMS[args.arm], "p", args.model, plan, rec)
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    d = Path(args.dir)
    q = d / "q"
    q.mkdir(exist_ok=True)
    reqs = requests(args.n, plan["max_model_len"])
    eng = llm.llm_engine
    sp = dict(max_tokens=1, temperature=0.0, seed=0, ignore_eos=True, detokenize=False,
              extra_args={"kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False,
                                                 "remote_engine_id": None, "remote_block_ids": None,
                                                 "remote_host": None, "remote_port": None}})
    nxt, produced, prefill_tokens, depth_log = 0, 0, 0, []
    t0 = time.time()
    while not (d / "d_done").exists() and nxt <= len(reqs):
        consumed = int((d / "consumed.txt").read_text() or 0) if (d / "consumed.txt").exists() else 0
        if nxt - consumed < BUF and nxt < len(reqs):
            plen, _ = reqs[nxt]
            eng.add_request(f"p{nxt}", TokensPrompt(prompt_token_ids=prompt_ids(nxt, plen)), SamplingParams(**sp))
            nxt += 1
        for o in eng.step() if eng.has_unfinished_requests() else []:
            if o.finished:
                i = int(o.request_id.split("-")[0][1:])
                write_json(q / f"{i}.json", {"i": i, "params": o.kv_transfer_params})
                produced += 1
                prefill_tokens += reqs[i][0]
        if not eng.has_unfinished_requests():
            eng.step()                                     # let the scheduler see D's finished reads
            time.sleep(0.01)
        if len(depth_log) < 5_000 and produced and produced % 10 == 0:
            depth_log.append([round(time.time() - t0, 1), produced, nxt - consumed])
    rec.update(produced=produced, prefill_tokens=prefill_tokens, prefill_seconds=round(time.time() - t0, 1),
               prefill_tok_s=prefill_tokens / max(1e-9, time.time() - t0), depth_log=depth_log[-200:])


def run_d(args, plan: dict, rec: dict) -> None:
    llm, ce = build(ARMS[args.arm], "d", args.model, plan, rec)
    import run_e69
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.request import RequestStatus
    d = Path(args.dir)
    q = d / "q"
    reqs = requests(args.n, plan["max_model_len"])

    def watchdog():
        rec.update(status="failed", error=f"watchdog: D did not finish within {plan['watchdog']} s", finished_at=now())
        write_json(d / "d_run.json", rec)
        (d / "d_done").write_text(now() + "\n")
        os._exit(1)

    timer = threading.Timer(plan["watchdog"], watchdog)
    timer.daemon = True
    timer.start()
    eng = llm.llm_engine
    sch = ce.engine_core(llm).scheduler
    ce.STATE["sched"] = []
    unallocated = (RequestStatus.WAITING, RequestStatus.PREEMPTED)
    nxt, work, steps, starved_since = 0, 0, 0, None
    lo, hi = plan["warmup"], plan["warmup"] + plan["window"]
    win = {"tokens": 0, "t0": None, "saturated": 0, "steps": 0, "running": 0, "preempt0": None, "completed": 0,
           "q_depth": 0, "q_samples": 0}
    early = 0

    def q_depth() -> int:                                                 # R-118 F5: prefilled requests not taken
        k = 0
        while (q / f"{nxt + k}.json").exists() and k < 50:
            k += 1
        return k
    while work < hi:
        while sum(r.status in unallocated for r in sch.waiting) < WAIT and (q / f"{nxt}.json").exists():
            r = json.loads((q / f"{nxt}.json").read_text())
            plen, olen = reqs[nxt]
            sp = SamplingParams(max_tokens=olen, temperature=0.0, seed=0, ignore_eos=True, detokenize=False,
                                output_kind=RequestOutputKind.DELTA, extra_args={"kv_transfer_params": r["params"]})
            eng.add_request(f"d{nxt}", TokensPrompt(prompt_token_ids=prompt_ids(nxt, plen)), sp)
            nxt += 1
            (d / "consumed.txt").write_text(str(nxt))
        # A1 (R-118 F2; R-119 F2): D is starved when nothing waits and no prefilled request is ready, after the first one
        if nxt == 0 or sch.waiting or (q / f"{nxt}.json").exists():
            starved_since = None
        elif starved_since is None:
            starved_since = time.time()
        elif time.time() - starved_since > STALL_S:
            raise RuntimeError(f"P starved: D had no request to admit for {STALL_S} s (P status in p_run.json)")
        if steps % 1_000 == 0 and (d / "p_run.json").exists() and \
                json.loads((d / "p_run.json").read_text()).get("status") == "failed":
            raise RuntimeError("P starved: P failed (see p_run.json)")
        if work == lo and win["t0"] is None:                              # A1 (R-118 F4): the clock starts first
            win["t0"], win["preempt0"] = time.time(), ce.STATE["preemptions"]
        n_sched = len(ce.STATE["sched"])
        outs = eng.step() if eng.has_unfinished_requests() else []
        steps += 1
        new_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        early += sum(1 for o in outs if o.finished and o.outputs[0].finish_reason != "length")
        if early:                                    # R-119 B1: a failed NIXL load ends with "error"; stop the arm
            raise RuntimeError(f"bad finish: {early} request(s) ended without reaching their length")
        if win["t0"] is not None:                                         # A1 (R-118 F4): tokens of every step
            win["tokens"] += new_tokens
        if len(ce.STATE["sched"]) == n_sched:                             # an idle step (a NIXL load wait)
            if not outs:
                time.sleep(0.001)
            continue
        if lo <= work < hi:
            win["steps"] += 1
            win["running"] += len(sch.running)
            win["saturated"] += int(any(r.status in unallocated for r in sch.waiting))
            win["completed"] += sum(1 for o in outs if o.finished)
            if work % 100 == 0:
                win["q_depth"] += q_depth()
                win["q_samples"] += 1
        work += 1
        if work % 500 == 0:
            write_json(d / "progress.json", {"work": work, "steps": steps, "added": nxt, "tokens": win["tokens"],
                                             "q_depth": q_depth(), "early_finishes": early, "t": now()})
        if len(ce.STATE["sched"]) > 50_000:
            ce.STATE["sched"] = ce.STATE["sched"][-10:]
    t1 = time.time()
    timer.cancel()
    secs = t1 - win["t0"]
    rec["window"] = {"work_steps": win["steps"], "seconds": round(secs, 2), "decode_tokens": win["tokens"],
                     "decode_tok_s": win["tokens"] / secs if secs > 0 else None,
                     "tokens_per_step": win["tokens"] / max(1, win["steps"]),
                     "running_per_step": win["running"] / max(1, win["steps"]),
                     "saturated_share": win["saturated"] / max(1, win["steps"]),
                     "preemptions": ce.STATE["preemptions"] - win["preempt0"], "all_steps": steps, "added": nxt,
                     "early_finishes": early, "completed": win["completed"],
                     "q_depth_mean": win["q_depth"] / max(1, win["q_samples"])}
    rec["stats"] = run_e69.HOOK["stats"].summary()
    log = run_e69.HOOK["events"].get("e71")
    rec["lookups"] = {"preempted": sum(e["preempted"] for e in log.events),
                      "preempted_hits": sum(e["hits"] for e in log.events if e["preempted"]),
                      "preempted_tokens": sum(e["tokens"] for e in log.events if e["preempted"])} if log else None
    rec["counters"] = {k: ce.STATE[k] for k in ("fix_reads", "pass_reads", "preemptions", "alloc_fail")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=("p", "d"))
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=2_000)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    plan = plan_for(args.arm, args.rehearsal)
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{args.role}_run.json"
    rec: dict = {"role": args.role, "arm": args.arm, "layout_kind": ARMS[args.arm], "status": "started",
                 "started_at": now(), "plan": plan, "manifest_ok": os.environ.get("E71_MANIFEST_OK") == "1"}
    write_json(out, rec)
    try:
        if args.stub:
            import stub_e71
            stub_e71.run(args, plan, rec)
        elif args.role == "p":
            run_p(args, plan, rec)
        else:
            run_d(args, plan, rec)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:3_000]
        rec["traceback"] = traceback.format_exc()[-8_000:]
    rec["finished_at"] = now()
    write_json(out, rec)
    if args.role == "d":
        (d / "d_done").write_text(now() + "\n")
    print(f"E-71 {args.arm} {args.role}: {rec['status']} | {json.dumps(rec.get('window'))} | "
          f"{rec.get('error', '')[:300]}", flush=True)
    sys.stdout.flush()
    os._exit(0 if rec["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
