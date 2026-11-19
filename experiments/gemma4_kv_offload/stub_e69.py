"""E-69 stub (README Section 12, stage S3, as amended by A1): a Mac stand-in for one arm, with no vLLM and no GPU.

Real token counts (the plan), large stub blocks (512 tokens), two groups (one global, one sliding with a 2-block
window) as small CPU tensors. At most 3 requests run; every EVENT_EVERY steps all running requests are preempted (the
forced event of A1 B1), their completed blocks hashed (the real `ReloadCheck`), stored, freed, and later reloaded into
NEW block ids (the free list is FIFO). Retention 0 (arm C-C0) reloads only the prompt part; the arm C-CX plants one
changed value after the load; C-C64 only starts. The real `EventLog` gets one record per re-admission.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

import e69_hooks as hk
from run_e69 import ARMS, EVENT_EVERY

BS = {0: 512, 1: 512}
SLIDING = {1}
WINDOW = 2
POOL = 60
MAX_RUN = 3


def run_arm(arm: str, plan: dict, eq_items, w_items, hook: dict) -> dict:
    layout, conn, retention_none, corrupt, start_only = ARMS[arm]
    compact = layout == "compact"
    base = {"gpu": "STUB L40S A100 80GB", "layout": "LBHNC", "layout_lines": ["stub"], "engine_seconds": 1.0,
            "runner": {"runner": "stub"}, "kv": {"stub": True}}
    if start_only:
        return {**base, "counters": {"fix_reads": 0, "fix_writes": 0, "pass_reads": 0, "pass_writes": 0,
                                     "alloc_fail": 0, "preemptions": 0}}
    g = torch.Generator().manual_seed(7)
    kv = {grp: torch.zeros(POOL + 1, 2, 4, 8) for grp in (0, 1)}
    free = list(range(1, POOL + 1))
    P, T = plan["prompt_len"], plan["max_tokens"]
    reqs = {f"w{i}": {"tokens": P, "blocks": {0: [], 1: []}, "cpu": None} for i in range(len(w_items))}
    waiting, running, done = list(reqs), [], []
    preempt = 0
    hook["phase"] = "w"
    log = hook["events"].setdefault("w", hk.EventLog())

    def nblocks(tokens):
        return {0: -(-tokens // BS[0]), 1: -(-tokens // BS[1])}

    def alloc(rid, data=None, upto=0) -> bool:
        r = reqs[rid]
        need = nblocks(r["tokens"])
        new0 = need[0] - len(r["blocks"][0])
        new1 = sum(1 for i in range(len(r["blocks"][1]), need[1]) if i >= need[1] - WINDOW)
        if new0 + new1 > len(free):
            return False
        for grp in (0, 1):
            for i in range(len(r["blocks"][grp]), need[grp]):
                if grp in SLIDING and i < need[grp] - WINDOW:
                    r["blocks"][grp].append(hk.NULL_BLOCK)
                    continue
                b = free.pop(0)                                    # FIFO: a reload gets new block ids
                old = (data or {}).get(grp, {}).get(i) if (i + 1) * BS[grp] <= upto else None
                kv[grp][b] = old.clone() if old is not None else torch.randn(2, 4, 8, generator=g)
                r["blocks"][grp].append(b)
        for i in range(max(0, len(r["blocks"][1]) - WINDOW)):
            if r["blocks"][1][i] != hk.NULL_BLOCK:
                free.append(r["blocks"][1][i])
                r["blocks"][1][i] = hk.NULL_BLOCK
        return True

    def release(rid):
        for grp in (0, 1):
            free.extend(b for b in reqs[rid]["blocks"][grp] if b != hk.NULL_BLOCK)
            reqs[rid]["blocks"][grp] = []

    def snap(rid, limit):
        r = reqs[rid]
        return hk.snapshot({0: [kv[0]], 1: [kv[1]]}, {grp: list(r["blocks"][grp]) for grp in (0, 1)}, r["tokens"],
                           BS, limit)

    def corrupt_fn(rid):
        b = next(x for x in reqs[rid]["blocks"][0] if x != hk.NULL_BLOCK)
        kv[0][b].view(-1)[0] += 1

    steps = 0
    loads = 0.0
    while (running or waiting) and steps < 100_000:
        while waiting and len(running) < MAX_RUN:
            rid = waiting[0]
            r = reqs[rid]
            if r["cpu"] is None:
                if not alloc(rid):
                    break
            else:
                hit = (r["cpu"]["tokens"] if retention_none else P) // BS[0] * BS[0] if conn else 0
                if not alloc(rid, r["cpu"]["data"], hit):
                    break
                log.add(r["tokens"], hit, True)
                if conn and hit:
                    loads += hit * 1024
                    hook["reload"].on_loaded(rid, hit, lambda rid=rid, hit=hit: snap(rid, hit), BS, SLIDING, False,
                                             (lambda rid=rid: corrupt_fn(rid)) if corrupt else None)
                r["cpu"] = None
            waiting.pop(0)
            running.append(rid)
        for rid in list(running):
            reqs[rid]["tokens"] += 1
            if not alloc(rid):
                raise RuntimeError("stub pool too small")
            if reqs[rid]["tokens"] >= P + T:
                release(rid)
                running.remove(rid)
                done.append(rid)
        steps += 1
        if steps % EVENT_EVERY == 0 and running:            # the forced event: preempt every running request
            for rid in reversed(running):
                preempt += 1
                hook["stats_preemptions"] += 1
                r = reqs[rid]
                if conn:
                    hook["reload"].on_preempt(rid, lambda rid=rid: snap(rid, reqs[rid]["tokens"]))
                r["cpu"] = {"tokens": r["tokens"], "data": {grp: {i: kv[grp][b].clone() for i, b in
                                                                  enumerate(r["blocks"][grp]) if b != hk.NULL_BLOCK}
                                                            for grp in (0, 1)}}
                release(rid)
                waiting.insert(0, rid)
            running.clear()
        hook["stats"].add(SimpleNamespace(connector_prefix_cache_stats=None,
                                          kv_connector_stats={"vllm:kv_offload_load_bytes": loads,
                                                              "vllm:kv_offload_load_time": 1e-3} if loads else {}))
        loads = 0.0
    eq_tokens = [[(sum(x) + j) % 997 for j in range(4)] for x in eq_items]     # equal in the stock and compact arms
    return {**base, "eq_tokens": eq_tokens, "eq_load_bytes": 0.0,
            "w": {"seconds": 1.0, "out_tokens": T * len(done), "tok_per_s": None, "steps": steps},
            "e2e": {"tokens": [1, 2, 3]},
            "counters": {"fix_reads": 3 if compact else 0, "fix_writes": 0, "pass_reads": 0, "pass_writes": 0,
                         "alloc_fail": 0, "preemptions": preempt},
            "stub_unfinished": [k for k in reqs if k not in done]}
