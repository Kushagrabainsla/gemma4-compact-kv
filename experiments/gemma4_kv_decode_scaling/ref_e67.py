"""E-67 check V6: an independent chunk-level reference for the CPU offload model in `poolsim.py`.

It replays one `PoolSim(offload=...)` run with per-chunk keys, a copy of `_lookup_complete_chunks` (pinned commit
3966924d5, `offloading/scheduler.py`) and of the LRU rank rules (`kv_offload/cpu/policies/lru.py`: a request rank with
tails first, renewed per chunk at finish). The model decides the run. At each re-admission the reference compares its
hit with the model hit.

Usage: ref_e67.py <out json>
"""
from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import e64  # noqa: E402
import e67  # noqa: E402
import poolsim as ps  # noqa: E402

SB, GROUPS, WIN = 16, 5, 64


class Ref:
    def __init__(self, gb: int, prompt_only: bool, cap_chunks: int):
        self.gb, self.prompt_only, self.cap = gb, prompt_only, cap_chunks
        self.rank: dict = {}
        self.heap: list = []
        self.next_rank = 0
        self.req_rank: dict = {}
        self.next_g: dict = {}
        self.next_s: dict = {}
        self.evictions = 0

    def _insert(self, key, rid: int, position: int) -> None:
        if key in self.rank:
            return
        if rid not in self.req_rank:
            self.next_rank += 1
            self.req_rank[rid] = self.next_rank
        r = (self.req_rank[rid], -position)
        self.rank[key] = r
        heapq.heappush(self.heap, (r, key))
        while len(self.rank) > self.cap:
            r0, k0 = heapq.heappop(self.heap)
            if self.rank.get(k0) == r0:
                del self.rank[k0]
                self.evictions += 1

    def restart(self, rid: int, hit: int) -> None:
        """`get_num_new_matched_tokens` resets the store cursors; a hit moves them past the loaded chunks."""
        self.next_g[rid] = hit // self.gb
        self.next_s[rid] = hit // SB

    def store(self, rid: int, base: int, c0: int, c1: int, first_live: int | None = None) -> None:
        lim = min(c1, base) if self.prompt_only else c1
        g0, g1 = self.next_g.get(rid, 0), lim // self.gb
        for i in range(g0, g1):
            self._insert((0, rid, i), rid, (i + 1) * self.gb)
        self.next_g[rid] = max(g0, g1)
        s0, s1 = self.next_s.get(rid, 0), lim // SB
        live = c0 // SB if first_live is None else first_live
        for i in range(max(s0, live), s1):
            for grp in range(GROUPS):
                self._insert((1 + grp, rid, i), rid, (i + 1) * SB)
        self.next_s[rid] = max(s0, s1)

    def finish(self, rid: int) -> None:
        keys = [k for k in self.rank if k[1] == rid] if rid in self.req_rank else []
        keys.sort(key=lambda k: -(k[2] + 1) * (self.gb if k[0] == 0 else SB))    # tails get the oldest new ranks
        for k in keys:
            self.next_rank += 1
            self.rank[k] = (self.next_rank, 0)
            heapq.heappush(self.heap, (self.rank[k], k))

    def _prefix(self, rid: int, n: int) -> int:
        c = 0
        while c < n and (0, rid, c) in self.rank:
            c += 1
        return c

    def _window(self, rid: int, n: int, end_tokens: int) -> int:
        """`_sliding_window_lookup` on all groups: the end of the last run of stored chunks, scanning from the end. The
        first run may need a larger window for a partial rightmost chunk (`load_window_size_in_chunks`)."""
        run = 0
        need = max(WIN, -(-(1_023 + -end_tokens % SB) // SB))
        for i in range(n - 1, -1, -1):
            if all((1 + g, rid, i) in self.rank for g in range(GROUPS)):
                run += 1
                if run == need:
                    return i + need
            else:
                run, need = 0, WIN
        return run

    def lookup(self, rid: int, num_tokens: int) -> int:
        """`_lookup_complete_chunks` with no local hit: full-attention group first, then the sliding groups."""
        hit, mx = 0, num_tokens - 1
        again = True
        while again:
            again = False
            for tpc, sliding in ((self.gb, False), (SB, True)):
                mx = min(mx, num_tokens // tpc * tpc)
                if mx < tpc:
                    return 0
                n = -(-mx // tpc)
                got = self._window(rid, n, mx) if sliding else self._prefix(rid, n)
                if got == 0:
                    return 0
                mx = min(mx, tpc * got)
                if mx < tpc:
                    return 0
                if mx < hit:
                    again = True
                hit = mx
        return hit


def replay(pool: int, layout: str, mode: str, gen_fn, horizon: int) -> dict:
    off = e67.MODES[mode]
    sim = ps.PoolSim(pool, layout, decode_only=True, offload=off)
    probe = ps.PoolSim(pool, "stock")
    ref = Ref(sim.gb, off[0], sim.cpu_cap)
    it = gen_fn()
    out = {"readmits": 0, "both_zero": 0, "both_hit": 0, "same_hit": 0, "model_only": 0, "ref_only": 0, "model_tokens": 0,
           "ref_tokens": 0}
    for _ in range(horizon):
        while len(sim.waiting) < 8:
            p, m = next(it)
            if probe.reservation(p + m) <= pool:
                sim.add(p, m)
        before = list(sim.running)
        c_before = {r.rid: r.c for r in before}
        done_before = len(sim.done)
        waiting_before = {r.rid for r in sim.waiting}
        sim.step()
        finished = sim.done[done_before:]
        now = {r.rid for r in sim.running} | {r.rid for r in finished}
        for r in before:
            if r.rid in now and r.c != c_before[r.rid]:
                ref.store(r.rid, r.base, c_before[r.rid], r.c)
        for r in sim.running + finished:
            if r.rid not in waiting_before or r.seq < 0 or r.rid in c_before:
                continue
            if r.npre == 0:                                 # the prompt KV arrived from a prefill node
                ref.store(r.rid, r.base, r.p, r.p, first_live=max(0, r.p - 1_023) // SB)
                continue
            n = r.hist[-1] if r.hist else 0
            model_hit, ref_hit = r.c - n, ref.lookup(r.rid, r.p)
            out["readmits"] += 1
            out["model_tokens"] += model_hit
            out["ref_tokens"] += ref_hit
            key = ("both_hit" if ref_hit else "model_only") if model_hit else ("ref_only" if ref_hit else "both_zero")
            out[key] += 1
            out["same_hit"] += model_hit == ref_hit
            ref.restart(r.rid, model_hit)
            ref.store(r.rid, r.base, model_hit, r.c)
        for r in finished:
            ref.finish(r.rid)
    out["ref_evictions"], out["model_evictions"] = ref.evictions, sim.stats["cpu_evictions"]
    out["preemptions"] = sim.stats["preemptions"]
    return out


# A1 B2: cells with at least 20 re-admissions and with CPU evictions, chosen from a simulation-only scan.
CELLS = (("M2", 14_672, "compact", "prompt200", 60_000), ("M2", 19_027, "compact", "all200", 120_000),
         ("D3", 14_672, "compact", "all200", 60_000), ("D3", 14_672, "compact", "prompt200", 60_000),
         ("M1", 19_027, "compact", "all64", 60_000), ("M1", 19_027, "stock", "all200", 60_000),
         ("D2", 14_672, "stock", "prompt200", 60_000))


def main() -> None:
    rows = []
    for name, pool, layout, mode, horizon in CELLS:
        r = replay(pool, layout, mode, e64.gen_factory(name, 0), horizon)
        r.update(workload=name, pool=pool, layout=layout, mode=mode, horizon=horizon)
        rows.append(r)
        print(json.dumps(r), flush=True)
    keys = ("readmits", "both_zero", "both_hit", "same_hit", "model_only", "ref_only", "model_tokens", "ref_tokens")
    tot = {k: sum(r[k] for r in rows) for k in keys}
    Path(sys.argv[1]).write_text(json.dumps({"cells": rows, "total": tot}, indent=1) + "\n")
    print("TOTAL", json.dumps(tot))


if __name__ == "__main__":
    main()
