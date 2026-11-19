"""E-69 (README Sections 4 to 6, as amended by A1): one arm of the engine test in one process.

The arm builds the E-57 engine (stock or compact, eager, in process) with or without the CPU offload connector, runs
the E2E row (report only), the EQ set and then the workload W, and writes `<out>/run_e69.json`. In W the driver forces
a preemption of every running request every EVENT_EVERY engine steps (A1 B1: `reset_prefix_cache(
reset_running_requests=True)`), so the reload path runs whatever the memory pressure. The hooks (`e69_hooks.py`) count
the stats, log every connector lookup, and check the bytes of the first reloads (V-RELOAD).

Usage: run_e69.py --arm C-C --model <id or snapshot dir> --out <dir> [--rehearsal] [--stub]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_engine"), str(EXP / "gemma4_kv_model")]
import e69_hooks as hk  # noqa: E402

# arm: layout, connector, retention None, planted corruption, start only (A1: C-CX and C-C64 are rehearsal arms)
ARMS = {"S-R": ("stock", False, None, False, False), "S-C": ("stock", True, True, False, False),
        "C-R": ("compact", False, None, False, False), "C-C": ("compact", True, True, False, False),
        "C-C0": ("compact", True, False, False, False), "C-CX": ("compact", True, True, True, False),
        "C-C64": ("compact", True, True, False, True)}
LAYOUT = "LBHNC"
GIB = 2**30
EVENT_EVERY = 256                  # A1 B1: a forced preemption every 256 engine steps of W
MAIN = {"prompts": 6, "prompt_len": 8_192, "max_tokens": 1_536, "kv_bytes": 15 * GIB, "cpu_bytes": 96 * GIB,  # A3
        "max_model_len": 16_896}
REH = {"prompts": 6, "prompt_len": 4_096, "max_tokens": 1_536, "kv_bytes": 4 * GIB, "cpu_bytes": 16 * GIB,
       "max_model_len": 8_192}
CPU64 = 64 * GIB                   # A1 F6: the C-C64 start with the main tier size
EQ = {"prompts": 8, "prompt_len": 2_048, "max_tokens": 128}
E2E = {"prompt_len": 2_048, "max_tokens": 512, "event_step": 384}     # A1 F2 (report only)
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_num_batched_tokens": 2_048,
          "max_num_seqs": 16, "enable_prefix_caching": False, "disable_log_stats": False,
          "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def prompt(kind: str, length: int, index: int) -> list[int]:
    """The E-57 `cap_prompt` generator: `length` token ids in [1,000, 100,000), distinct per request (seed "e69")."""
    rng = np.random.default_rng(seed_of("e69", kind, length, index))
    return rng.integers(1_000, 100_000, size=length).tolist()


def engine_kwargs(arm: str, plan: dict) -> dict:
    layout, conn, retention_none, _, start_only = ARMS[arm]
    kw = {**ENGINE, "kv_cache_memory_bytes": plan["kv_bytes"], "max_model_len": plan["max_model_len"]}
    if conn:
        kw["kv_transfer_config"] = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                                    "kv_connector_extra_config": {"cpu_bytes_to_use": CPU64 if start_only
                                                                  else plan["cpu_bytes"],
                                                                  "offload_prompt_only": False}}
        kw["prefix_cache_retention_interval"] = None if retention_none else 0
    return kw


class LayoutLog(logging.Handler):
    """V6 and the rehearsal VERIFY items: the vLLM log lines on the layout, the CPU tier and the host registration."""
    KEYS = ("KV cache layout", "mmap", "offload region", "cudaHostRegister", "pageable", "kernel block")

    def __init__(self):
        super().__init__()
        self.layout, self.lines = None, []

    def emit(self, record):
        msg = record.getMessage()
        if any(k.lower() in msg.lower() for k in self.KEYS) and len(self.lines) < 60:
            self.lines.append(msg[:300])
        if "KV cache layout" in msg:
            for name in ("LBHNC", "LBNHC", "NHD", "HND"):
                if name in msg:
                    self.layout = name
                    break


HOOK: dict = {"stats": hk.StatsSum(), "reload": hk.ReloadCheck(), "events": {}, "phase": "setup", "kv": {},
              "groups": None, "stats_preemptions": 0, "errors": [], "corrupt": False}


def group_info(sch) -> tuple[dict, dict, set]:
    """Layers, block size and the sliding set of each KV cache group, from the scheduler's kv_cache_config."""
    layers, bsz, sliding = {}, {}, set()
    for g, grp in enumerate(sch.kv_cache_config.kv_cache_groups):
        layers[g] = [HOOK["kv"][n] for n in grp.layer_names if n in HOOK["kv"]]
        bsz[g] = int(grp.kv_cache_spec.block_size)
        if getattr(grp.kv_cache_spec, "sliding_window", None):
            sliding.add(g)
    return layers, bsz, sliding


def snap(sch, req, limit: int | None) -> hk.Snapshot:
    import torch
    torch.cuda.synchronize()
    if HOOK["groups"] is None:
        HOOK["groups"] = group_info(sch)
    layers, bsz, _ = HOOK["groups"]
    ids = sch.kv_cache_manager.get_block_ids(req.request_id)
    return hk.snapshot(layers, {g: list(b) for g, b in enumerate(ids)}, int(req.num_computed_tokens), bsz, limit)


def corrupt_first_global(sch, req) -> None:
    """A1 B3, the planted control (arm C-CX): add 1 to one value of the first loaded global block."""
    layers, _, sliding = HOOK["groups"]
    ids = sch.kv_cache_manager.get_block_ids(req.request_id)
    for g, blocks in enumerate(ids):
        if g in sliding or not layers.get(g):
            continue
        live = [b for b in blocks if b != hk.NULL_BLOCK]
        if live:
            t = layers[g][0]
            t[(live[0],) + (0,) * (t.dim() - 1)] += 1      # direct index: the views can be non-contiguous
            return


def install_hooks(conn: bool, corrupt: bool) -> None:
    """The stats and lookup wrappers (every arm) and the V-RELOAD hooks (connector arms). After `ce.install`."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.metrics.stats import PrefixCacheStats
    orig_stats = Scheduler.make_stats

    def make_stats(self, *args, **kwargs):
        out = orig_stats(self, *args, **kwargs)
        HOOK["stats"].add(out)
        return out

    Scheduler.make_stats = make_stats
    orig_record = PrefixCacheStats.record

    def record(self, *args, **kwargs):
        try:
            names = ("num_tokens", "num_hits", "preempted")
            vals = dict(zip(names, args))
            vals.update({k: kwargs[k] for k in names if k in kwargs})
            if "preempted" in vals:                       # the connector call carries the preempted flag
                HOOK["events"].setdefault(HOOK["phase"], hk.EventLog()).add(vals["num_tokens"], vals["num_hits"],
                                                                           vals["preempted"])
        except Exception as e:  # noqa: BLE001 (a hook error is recorded, never raised into the engine)
            HOOK["errors"].append(f"record: {type(e).__name__}: {e}"[:300])
        return orig_record(self, *args, **kwargs)

    PrefixCacheStats.record = record
    orig_preempt = Scheduler._preempt_request

    def _preempt_request(self, request, *args, **kwargs):
        HOOK["stats_preemptions"] += 1
        if conn and HOOK["kv"] and HOOK["phase"] == "w":
            try:
                done = int(request.num_computed_tokens) - int(getattr(request, "num_in_flight_tokens", 0) or 0)
                HOOK["reload"].on_preempt(request.request_id, lambda: snap(self, request, done),
                                          int(request.num_computed_tokens))
            except Exception as e:  # noqa: BLE001
                HOOK["errors"].append(f"preempt: {type(e).__name__}: {e}"[:300])
        return orig_preempt(self, request, *args, **kwargs)

    Scheduler._preempt_request = _preempt_request
    if not conn:
        return
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
    orig_reg = OffloadingConnectorWorker.register_kv_caches

    def register_kv_caches(self, kv_caches, *args, **kwargs):
        HOOK["kv"] = dict(kv_caches)
        return orig_reg(self, kv_caches, *args, **kwargs)

    OffloadingConnectorWorker.register_kv_caches = register_kv_caches
    orig_update = Scheduler._update_waiting_for_remote_kv

    def _update_waiting_for_remote_kv(self, request, *args, **kwargs):
        # A1 B3: the hit is num_computed_tokens before the call (the call can lower it by the replay tokens).
        rid = request.request_id
        hit = int(request.num_computed_tokens)
        failed = rid in getattr(self, "failed_recving_kv_req_ids", set())
        out = orig_update(self, request, *args, **kwargs)
        if HOOK["phase"] == "w" and rid not in HOOK["reload"].pending:
            HOOK["reload"].note_load(rid, hit)                 # A3: keep the recomputed range of every request
        if rid in HOOK["reload"].pending:
            try:
                _, bsz, sliding = HOOK["groups"] or group_info(self)
                HOOK["reload"].on_loaded(rid, hit, lambda: snap(self, request, hit), bsz, sliding, failed,
                                         (lambda: corrupt_first_global(self, request)) if corrupt else None)
            except Exception as e:  # noqa: BLE001
                HOOK["errors"].append(f"loaded: {type(e).__name__}: {e}"[:300])
        return out

    Scheduler._update_waiting_for_remote_kv = _update_waiting_for_remote_kv


def loading_requests(sch) -> int:
    """Requests that wait for a KV load: a forced reset would raise while one exists (scheduler.py:3211)."""
    from vllm.v1.request import RequestStatus
    reqs = getattr(sch, "requests", {}) or {}
    return sum(1 for r in reqs.values() if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS)


def force_preempt(sch) -> int:
    """A2 (the rehearsal of 2026-11-13): the preemption part of `Scheduler.reset_prefix_cache(reset_running_requests=
    True)` (scheduler.py:3192-3206 at the pin), without its block-pool reset, which refuses while connector stores hold
    blocks (prefix caching is off, so that reset has no purpose here). Returns the number of preempted requests."""
    ts = time.monotonic()
    n = 0
    while sch.running:
        sch._preempt_request(sch.running.pop(), ts, drop_stale_output=True)
        n += 1
    ids = getattr(sch, "prev_step_scheduled_req_ids", None)
    if ids is not None:
        ids.clear()
    return n


def run_driven(llm, sp_cls, tp_cls, items: list[list[int]], max_tokens: int, tag: str, event_steps=None,
               every: int | None = None) -> dict:
    """Drive the engine step by step. Force a preemption of every running request at the given steps (or every
    `every` steps); an event waits for a step with a running request."""
    import compact_engine as ce
    eng = llm.llm_engine
    params = sp_cls(max_tokens=max_tokens, temperature=0.0, seed=0, ignore_eos=True, detokenize=False)
    for i, p in enumerate(items):
        eng.add_request(f"{tag}{i}", tp_cls(prompt_token_ids=p), params)
    toks: dict = {}
    steps, events, postponed = 0, [], 0
    nxt = (event_steps or [None])[0] if event_steps else every
    queue = list(event_steps or [])
    t0 = time.time()
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                toks[o.request_id] = list(o.outputs[0].token_ids)
        steps += 1
        if nxt is not None and steps >= nxt and eng.has_unfinished_requests():
            sch = ce.engine_core(llm).scheduler
            if not sch.running:
                postponed += 1
                continue
            events.append([steps, force_preempt(sch), loading_requests(sch)])
            if queue:
                queue.pop(0)
                nxt = queue[0] if queue else None
            else:
                nxt = steps + every
    sec = time.time() - t0
    order = [toks.get(f"{tag}{i}", []) for i in range(len(items))]
    return {"seconds": sec, "tokens": order, "out_tokens": sum(len(t) for t in order), "steps": steps,
            "events": events, "postponed": postponed,
            "tok_per_s": sum(len(t) for t in order) / sec if sec > 0 else None}


def runner_record(llm) -> dict:
    """A1 F4: the kernel block sizes of the attention groups, best effort (the attribute names can differ)."""
    import compact_engine as ce
    out: dict = {}
    try:
        core = ce.engine_core(llm)
        ex = core.model_executor
        w = getattr(ex, "driver_worker", None)
        w = getattr(w, "worker", w)
        mr = getattr(w, "model_runner", None)
        for name in dir(mr):
            if "kernel_block" in name or name in ("attn_groups",):
                try:
                    out[name] = repr(getattr(mr, name))[:1_500]
                except Exception as e:  # noqa: BLE001
                    out[name] = f"{type(e).__name__}"
        out["runner"] = type(mr).__name__
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--try", dest="tries", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = REH if args.rehearsal else MAIN
    layout, conn, _, corrupt, start_only = ARMS[args.arm]
    rec: dict = {"arm": args.arm, "layout_kind": layout, "connector": conn, "corrupt": corrupt,
                 "start_only": start_only, "status": "started", "started_at": now(), "tries": args.tries,
                 "plan": plan, "eq": EQ, "e2e": E2E, "event_every": EVENT_EVERY, "prompt_len": plan["prompt_len"],
                 "engine_kwargs": engine_kwargs(args.arm, plan),
                 "manifest_ok": os.environ.get("E69_MANIFEST_OK") == "1"}

    def save():
        (out / "run_e69.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")

    save()
    try:
        eq_items = [prompt("eq", EQ["prompt_len"], i) for i in range(EQ["prompts"])]
        w_items = [prompt("w", plan["prompt_len"], i) for i in range(plan["prompts"])]
        e2e_item = prompt("e2e", E2E["prompt_len"], 0)
        if args.stub:
            import stub_e69 as stub
            rec.update(stub.run_arm(args.arm, plan, eq_items, w_items, HOOK))
        else:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            os.environ["VLLM_KV_CACHE_LAYOUT"] = LAYOUT
            import torch
            import compact_engine as ce
            ce.install(layout)
            install_hooks(conn, corrupt)
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            logh = LayoutLog()                         # A2: after the vLLM import, which sets up its own logging
            logging.getLogger("vllm").addHandler(logh)
            torch.backends.cuda.matmul.allow_tf32 = False
            t0 = time.time()
            llm = LLM(model=args.model, tokenizer=args.model, **rec["engine_kwargs"])
            rec["engine_seconds"] = round(time.time() - t0, 1)
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["kv"] = ce.kv_record(llm)
            rec["spec"] = ce.STATE["spec"]
            rec["runner"] = runner_record(llm)
            rec["connector_layers"] = len(HOOK["kv"])
            rec["layout"], rec["layout_lines"] = logh.layout or rec["kv"].get("layout"), logh.lines
            save()
            if not start_only:
                splice = ce.ViewsSplice()
                rec["hooks"] = llm.apply_model(lambda model: ce.register(model, splice))[0]
                # EMULV is mode VIEWS on the stock engine (compact_engine.py, run_e57.py:831); A2 after try 2
                stock_eq, stock_w = ("VIEWS", "VIEWS") if layout == "compact" else ("VIEWS", "OFF")
                splice.set_mode(stock_w)
                HOOK["phase"] = "e2e"
                e2e = run_driven(llm, SamplingParams, TokensPrompt, [e2e_item], E2E["max_tokens"], "e",
                                 event_steps=[E2E["event_step"]] if conn else None)
                rec["e2e"] = {**E2E, "tokens": e2e["tokens"][0], "events": e2e["events"]}
                HOOK["phase"] = "eq"
                splice.set_mode(stock_eq)
                load0 = HOOK["stats"].kv.get("vllm:kv_offload_load_bytes", 0.0)
                eq = run_driven(llm, SamplingParams, TokensPrompt, eq_items, EQ["max_tokens"], "q")
                rec["eq_tokens"], rec["eq_seconds"] = eq["tokens"], eq["seconds"]
                rec["eq_load_bytes"] = HOOK["stats"].kv.get("vllm:kv_offload_load_bytes", 0.0) - load0  # A1 F5
                rec["counters_after_eq"] = {k: ce.STATE[k] for k in ("fix_reads", "pass_reads", "preemptions")}
                save()
                HOOK["phase"] = "w"
                splice.set_mode(stock_w)
                w = run_driven(llm, SamplingParams, TokensPrompt, w_items, plan["max_tokens"], "w", every=EVENT_EVERY)
                rec["w"] = {k: w[k] for k in ("seconds", "out_tokens", "tok_per_s", "steps", "events", "postponed")}
                rec["w_tokens_sha"] = [hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16] for t in w["tokens"]]
                (out / "w_tokens.json").write_text(json.dumps(w["tokens"]) + "\n")
            rec["counters"] = {k: ce.STATE[k] for k in ("fix_reads", "fix_writes", "pass_reads", "pass_writes",
                                                        "alloc_fail", "preemptions")}
            rec["layout"], rec["layout_lines"] = logh.layout or rec["kv"].get("layout"), logh.lines
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"[:2_000]
        rec["traceback"] = traceback.format_exc()[-8_000:]
    rec["stats"] = HOOK["stats"].summary()
    rec["stats_preemptions"] = HOOK["stats_preemptions"]
    rec["reload"] = HOOK["reload"].summary()
    rec["events"] = {ph: log.events for ph, log in HOOK["events"].items()}
    w_log = HOOK["events"].get("w")
    rec["coverage"] = w_log.decode_coverage(plan["prompt_len"]) if w_log else None
    rec["hook_errors"] = HOOK["errors"]
    rec["finished_at"] = now()
    save()
    print(f"E-69 {args.arm}: {rec['status']} | preemptions {rec.get('counters', {}).get('preemptions')} | "
          f"reload {rec['reload'].get('checked_reloads')} checked, {rec['reload'].get('mismatched_blocks')} bad | "
          f"coverage {rec['coverage']} | {rec.get('error', '')[:300]}", flush=True)
    sys.exit(0 if rec["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
