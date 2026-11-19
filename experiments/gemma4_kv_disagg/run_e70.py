"""E-70 stage 1 (README Sections 4 to 6): one engine role (P or D) of one pair, in its own process.

P: build the engine, run the reference (local decode), run the prefills for D (`do_remote_decode`), hash each finished
request's blocks, write `p_records.json`, then wait for `d_done` while the blocks stay held for D.
D: build the engine, wait for `p_records.json`, add the requests with P's transfer parameters, hash each request's
blocks when its NIXL load ends, decode, compare the hashes with P's, write `d_result.json` and `d_done`.

Usage: run_e70.py --role p|d --pair STOCK|COMPACT-NOPATCH|COMPACT --model <dir> --data <data_e56.json> --dir <pair dir>
       [--n 8] [--wait 600] [--stub]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_offload"), str(EXP / "gemma4_kv_engine"), str(EXP / "gemma4_kv_model")]
import nixl_compact as nx  # noqa: E402

PAIRS = {"STOCK": ("stock", False), "COMPACT-NOPATCH": ("compact", False), "COMPACT": ("compact", True),
         "STOCK2": ("stock", False), "COMPACT2": ("compact", True)}            # stage 2 (README_stage2.md)
PORT = {"p": 5600, "d": 5601}
GIB = 2**30
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_num_batched_tokens": 2_048,
          "max_num_seqs": 8, "enable_prefix_caching": False, "kv_cache_memory_bytes": 8 * GIB, "max_model_len": 8_704,
          "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
          # A2 (try 1): the default UCX mode "peer" excludes the intra-node transports, and gVisor has no TCP
          "kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": "kv_both",
                                 "kv_connector_extra_config": {"ucx_backend_params":
                                                               {"ucx_error_handling_mode": "none"}}}}
REF_TOKENS = 32
D_WATCHDOG_S = 300                                 # A1 (R-114 N8): the transfer and decode of 8 requests
# Stage 2 (README_stage2.md Section 4): D with NIXL and the offload connector, long decodes, forced preemptions
S2_TOKENS, S2_CPU_BYTES, S2_WATCHDOG_S, S2_PROMPT = 1_536, 32 * GIB, 600, 7_666   # A1 (R-116 F3): 32 GiB tier
S2_MAX_MODEL_LEN = 9_216                           # A1 (R-116 B1): prompt 7,667 + 1,536 output tokens must fit
S2_EVENT_TOKENS = 256                              # A1 (R-116 B2): an event every 256 generated tokens
UCX = {"ucx_backend_params": {"ucx_error_handling_mode": "none"}}


def event_due(output_counts: list[int], nxt: int) -> bool:
    """A1 (R-116 B2): a forced event fires when every running request has at least `nxt` generated tokens (engine
    steps would count the idle steps of the NIXL loads)."""
    return bool(output_counts) and min(output_counts) >= nxt


class MultiStatsSum:
    """A1 (R-116 F2): the MultiConnector nests the stats of each child by connector name; give the E-69 `StatsSum`
    the offload child's payload."""

    def __init__(self, inner):
        self.inner = inner

    def add(self, stats) -> None:
        kv = getattr(stats, "kv_connector_stats", None) if stats is not None else None
        if isinstance(kv, dict) and "types" not in kv:
            sub = kv.get("OffloadingConnector")
            if sub is None:
                sub = next((v for v in kv.values() if isinstance(v, dict) and "types" in v
                            and any("kv_offload" in k for k in v.get("types", {}))), None)
            from types import SimpleNamespace
            stats = SimpleNamespace(connector_prefix_cache_stats=getattr(stats, "connector_prefix_cache_stats", None),
                                    kv_connector_stats=sub or {})
        self.inner.add(stats)

    def summary(self) -> dict:
        return self.inner.summary()


def d2_engine() -> dict:
    """The D engine of stage 2: a MultiConnector of NIXL and the offload connector (the CI child order), the dense store
    (`prefix_cache_retention_interval` None, E-69), and the stats on (the E-69 lookup log needs them)."""
    kw = dict(ENGINE)
    kw["kv_transfer_config"] = {"kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_connector_extra_config": {
        "connectors": [{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": UCX},
                       {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                        "kv_connector_extra_config": {"cpu_bytes_to_use": S2_CPU_BYTES,
                                                      "offload_prompt_only": False}}]}}
    kw["prefix_cache_retention_interval"] = None
    kw["disable_log_stats"] = False
    kw["max_model_len"] = S2_MAX_MODEL_LEN
    return kw
P_PARAMS = {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None, "remote_block_ids": None,
            "remote_host": None, "remote_port": None}                       # as the vLLM toy proxy sends them


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.rename(path)


def load_items(data: Path, n: int) -> list[dict]:
    d = json.loads(data.read_text())
    return [{"id": x["id"], "ids": x["ids"], "gold": str(x["gold"])} for x in d["needle"][:n]]


def sampling_kwargs(max_tokens: int, kv_params: dict | None) -> dict:
    """A1 (R-114 B2): `ignore_eos`, so every request decodes `max_tokens` (the chat answers end early otherwise)."""
    kw = {"max_tokens": max_tokens, "temperature": 0.0, "seed": 0, "ignore_eos": True}
    if kv_params is not None:
        kw["extra_args"] = {"kv_transfer_params": kv_params}
    return kw


def lookup(store: dict, rid: str):
    """A1 (R-114 B1): the hooks see the scheduler's request id, which can carry a random suffix ("<id>-<8 chars>")."""
    if rid in store:
        return store[rid]
    hits = [v for k, v in store.items() if k.startswith(rid + "-")]
    return hits[0] if len(hits) == 1 else None


def wait_for(path: Path, limit: float) -> bool:
    end = time.time() + limit
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(1.0)
    return False


def build(layout: str, patch: bool, role: str, model: str, rec: dict, engine: dict | None = None,
          offload_hooks: bool = False):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"          # A1 (R-114 B1)
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "LBHNC"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(PORT[role])
    os.environ.setdefault("UCX_NET_DEVICES", "all")
    import torch
    import compact_engine as ce
    ce.install(layout)
    nx.install_capture_and_patch(patch)
    if role == "p":
        nx.install_p_hook()
    else:
        nx.install_d_hook()
    if offload_hooks:                                   # stage 2: the E-69 hooks (lookup log, V-RELOAD, stats)
        import run_e69
        run_e69.HOOK["stats"] = MultiStatsSum(run_e69.HOOK["stats"])
        run_e69.install_hooks(conn=True, corrupt=False)
    from vllm import LLM
    torch.backends.cuda.matmul.allow_tf32 = False
    rec["gpu"] = torch.cuda.get_device_name(0)               # A1 (R-114 F1): recorded before a start failure
    rec["layout_env"] = os.environ["VLLM_KV_CACHE_LAYOUT"]
    t0 = time.time()
    try:
        llm = LLM(model=model, tokenizer=model, **(engine or ENGINE))
    finally:
        rec["spec"] = ce.STATE["spec"]                       # set by get_kv_cache_spec, before the NIXL registration
        rec["patched"], rec["skipped"] = nx.STATE["patched"], nx.STATE["skipped"]
    nx.STATE["sched"] = ce.engine_core(llm).scheduler
    rec.update({"engine_seconds": round(time.time() - t0, 1), "kv": ce.kv_record(llm)})
    splice = ce.ViewsSplice()
    rec["hooks"] = llm.apply_model(lambda m: ce.register(m, splice))[0]
    splice.set_mode("VIEWS" if layout == "compact" else "OFF")
    return llm, ce


def answers(outs, items) -> list[dict]:
    res = []
    for o, it in zip(outs, items):
        c = o.outputs[0]
        res.append({"id": it["id"], "tokens": list(c.token_ids), "text": c.text[:300],
                    "gold_in_text": it["gold"].lower() in (c.text or "").lower()})
    return res


def run_p(args, rec: dict, items: list[dict]) -> None:
    layout, patch = PAIRS[args.pair]
    llm, _ = build(layout, patch, "p", args.model, rec)
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=it["ids"]) for it in items]
    t0 = time.time()
    ref = llm.generate(prompts, SamplingParams(**sampling_kwargs(REF_TOKENS, None)), use_tqdm=False)
    rec["ref"], rec["ref_seconds"] = answers(ref, items), round(time.time() - t0, 1)
    t0 = time.time()
    sp = SamplingParams(**sampling_kwargs(1, dict(P_PARAMS)))
    outs = llm.generate(prompts, sp, use_tqdm=False)
    rec["prefill_seconds"] = round(time.time() - t0, 1)
    rec["p_hook_keys"] = sorted(nx.STATE["p"])[:16]
    records = []
    for o, it in zip(outs, items):
        records.append({"id": it["id"], "request_id": o.request_id, "params": o.kv_transfer_params,
                        "hashes": lookup(nx.STATE["p"], o.request_id)})
    rec["p_hashed"] = sum(r["hashes"] is not None for r in records)
    rec["p_params"] = sum(bool(r["params"]) for r in records)
    write_json(Path(args.dir) / "p_records.json", {"records": records, "sliding": sorted(nx.group_info(
        nx.STATE["sched"])[2])})
    rec["status"] = "waiting for D"
    write_json(Path(args.dir) / "p_run.json", rec)
    rec["d_done_seen"] = wait_for(Path(args.dir) / "d_done", args.wait)


def run_d(args, rec: dict, items: list[dict]) -> None:
    layout, patch = PAIRS[args.pair]
    llm, _ = build(layout, patch, "d", args.model, rec)
    if not wait_for(Path(args.dir) / "p_records.json", args.wait):
        raise RuntimeError("no p_records.json from P")
    pr = json.loads((Path(args.dir) / "p_records.json").read_text())
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=it["ids"]) for it in items]
    sps = [SamplingParams(**sampling_kwargs(REF_TOKENS, r["params"])) for r in pr["records"]]

    def watchdog():                                        # A1 (R-114 N8): a stuck transfer must not hold the GPUs
        rec.update(status="failed", error=f"watchdog: D did not finish within {D_WATCHDOG_S} s",
                   finished_at=now(), hook_errors=nx.STATE["errors"])
        write_json(Path(args.dir) / "d_run.json", rec)
        (Path(args.dir) / "d_done").write_text(now() + "\n")
        os._exit(1)

    timer = threading.Timer(D_WATCHDOG_S, watchdog)
    timer.daemon = True
    timer.start()
    t0 = time.time()
    rec["t_generate"] = t0
    outs = llm.generate(prompts, sps, use_tqdm=False)
    timer.cancel()
    rec["decode_seconds"] = round(time.time() - t0, 1)
    rec["answers"] = answers(outs, items)
    rec["d_hook_keys"] = sorted(nx.STATE["d"])[:16]
    sliding = set(pr["sliding"])
    checks = []
    for o, r in zip(outs, pr["records"]):
        d = lookup(nx.STATE["d"], o.request_id)
        if d is None or r["hashes"] is None:
            checks.append({"id": r["id"], "missing": "d" if d is None else "p"})
            continue
        c = nx.compare(r["hashes"]["groups"], d["groups"], sliding)
        checks.append({"id": r["id"], "hit": d["hit"], "prompt": r["hashes"]["prompt"], "failed_load": d["failed"],
                       "load_s_after_start": round(d["t_loaded"] - t0, 3), **c})
    rec["checks"] = checks
    rec["n_complete"] = sum(len(a["tokens"]) == REF_TOKENS for a in rec["answers"])


def run_d2(args, rec: dict, items: list[dict]) -> None:
    """Stage 2: D with NIXL plus the offload connector; the 8 NIXL-fed requests decode 1,536 tokens each, and every
    running request is preempted every 256 steps (the E-69 forced event); the E-69 hooks check the reloads."""
    layout, patch = PAIRS[args.pair]
    llm, ce = build(layout, patch, "d", args.model, rec, engine=d2_engine(), offload_hooks=True)
    import run_e69
    if not wait_for(Path(args.dir) / "p_records.json", args.wait):
        raise RuntimeError("no p_records.json from P")
    pr = json.loads((Path(args.dir) / "p_records.json").read_text())
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def watchdog():
        rec.update(status="failed", error=f"watchdog: D did not finish within {S2_WATCHDOG_S} s", finished_at=now(),
                   hook_errors=nx.STATE["errors"] + run_e69.HOOK["errors"])
        write_json(Path(args.dir) / "d_run.json", rec)
        (Path(args.dir) / "d_done").write_text(now() + "\n")
        os._exit(1)

    timer = threading.Timer(S2_WATCHDOG_S, watchdog)
    timer.daemon = True
    timer.start()
    eng = llm.llm_engine
    run_e69.HOOK["phase"] = "w"
    for k, (it, r) in enumerate(zip(items, pr["records"])):
        eng.add_request(f"d{k}", TokensPrompt(prompt_token_ids=it["ids"]),
                        SamplingParams(**sampling_kwargs(S2_TOKENS, r["params"])))
    toks: dict = {}
    steps, events, nxt = 0, [], S2_EVENT_TOKENS
    t0 = time.time()
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                toks[o.request_id] = (list(o.outputs[0].token_ids), o.outputs[0].text or "")
        steps += 1
        sch = ce.engine_core(llm).scheduler
        counts = [int(r.num_output_tokens) for r in sch.running]
        if eng.has_unfinished_requests() and event_due(counts, nxt):
            events.append([steps, min(counts), run_e69.force_preempt(sch), run_e69.loading_requests(sch)])
            nxt += S2_EVENT_TOKENS
    timer.cancel()
    rec["decode_seconds"] = round(time.time() - t0, 1)
    rec["w"] = {"steps": steps, "events": events}
    sliding = set(pr["sliding"])
    checks, answers_out = [], []
    for k, (it, r) in enumerate(zip(items, pr["records"])):
        tk = lookup({kk: v for kk, v in toks.items()}, f"d{k}") or ([], "")
        answers_out.append({"id": it["id"], "n_tokens": len(tk[0]), "text": tk[1][:200],
                            "gold_in_text": it["gold"].lower() in tk[1].lower()})
        d = lookup(nx.STATE["d"], f"d{k}")
        if d is None or r["hashes"] is None:
            checks.append({"id": r["id"], "missing": "d" if d is None else "p"})
            continue
        checks.append({"id": r["id"], "hit": d["hit"], "prompt": r["hashes"]["prompt"], "failed_load": d["failed"],
                       **nx.compare(r["hashes"]["groups"], d["groups"], sliding)})
    rec["answers"], rec["checks"] = answers_out, checks
    rec["n_complete"] = sum(a["n_tokens"] == S2_TOKENS for a in answers_out)
    rec["counters"] = {k: ce.STATE[k] for k in ("fix_reads", "pass_reads", "preemptions")}
    rec["stats"] = run_e69.HOOK["stats"].summary()
    rec["stats_preemptions"] = run_e69.HOOK["stats_preemptions"]
    rec["reload"] = run_e69.HOOK["reload"].summary()
    w_log = run_e69.HOOK["events"].get("w")
    rec["coverage"] = w_log.decode_coverage(S2_PROMPT) if w_log else None
    rec["offload_hook_errors"] = run_e69.HOOK["errors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=("p", "d"))
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--wait", type=float, default=600)
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.dir) / f"{args.role}_run.json"
    rec: dict = {"role": args.role, "pair": args.pair, "started_at": now(), "status": "started",
                 "manifest_ok": os.environ.get("E70_MANIFEST_OK") == "1"}
    write_json(out, rec)
    try:
        items = load_items(Path(args.data), args.n)
        if args.stub:
            import stub_e70
            stub_e70.run(args, rec, items)
        elif args.role == "p":
            run_p(args, rec, items)
        elif args.pair.endswith("2"):
            run_d2(args, rec, items)
        else:
            run_d(args, rec, items)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:3_000]
        rec["traceback"] = traceback.format_exc()[-8_000:]
    rec["hook_errors"] = nx.STATE["errors"]
    rec["finished_at"] = now()
    write_json(out, rec)
    if args.role == "d":
        (Path(args.dir) / "d_done").write_text(now() + "\n")
    print(f"E-70 {args.pair} {args.role}: {rec['status']} | {rec.get('error', '')[:400]}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if rec["status"] == "ok" else 1)      # A1 (R-114 B3): no NIXL or UCX teardown at interpreter exit


if __name__ == "__main__":
    main()
