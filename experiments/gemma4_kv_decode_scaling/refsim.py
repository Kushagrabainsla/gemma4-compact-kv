"""E-66 A1 (R-103 B1): a block-level reference for the prefix-cache model in `poolsim.py`.

It replays one `PoolSim(prefix_cache="engine")` run with real block ids, one FIFO free queue, per-group block hashes
and a copy of the vLLM search for a hit (pinned commit 3966924d5: `FullAttentionManager` left to right,
`SlidingWindowManager` right to left with 64 contiguous blocks, alignment to lcm(16, global block), the hybrid fixed
point). At each re-admission it compares the reference hit with the model hit. The model decides the run; filler
blocks keep the free count of the reference equal to the model. Finished requests and victims free their blocks as
vLLM does: unhashed blocks to the front of the queue, hashed blocks to the back, newest first.

Usage: refsim.py <out json>
"""
from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import e64  # noqa: E402
import poolsim as ps  # noqa: E402

SB, GROUPS, WINDOW = ps.SBLOCK, ps.GROUPS, ps.WINDOW
NEED = math.ceil((WINDOW - 1) / SB)          # 64 contiguous sliding blocks for a hit


class Ref:
    def __init__(self, pool: int, gb: int, global_first: bool):
        self.gb, self.global_first = gb, global_first
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(pool))
        self.key: list = [None] * pool
        self.cached: dict = {}
        self.live: dict = {}
        self.short = 0

    def alloc(self) -> int:
        if not self.free:
            self.short += 1
            return -1
        b, _ = self.free.popitem(last=False)
        k = self.key[b]
        if k is not None:
            if self.cached.get(k) == b:
                del self.cached[k]
            self.key[b] = None
        return b

    def release(self, blocks: list[int]) -> None:
        """`free_blocks`: `blocks` in eviction order; unhashed to the front, hashed to the back."""
        front = []
        for b in blocks:
            if b < 0:
                continue
            if self.key[b] is None:
                front.append(b)
            else:
                self.free[b] = None
        for b in reversed(front):
            self.free[b] = None
            self.free.move_to_end(b, last=False)

    def state(self, rid: int) -> dict:
        return self.live.setdefault(rid, {"g": {}, "s": [{} for _ in range(GROUPS)], "fill": []})

    def _mark(self, group: int, rid: int, idx: int, b: int) -> None:
        if b >= 0 and self.key[b] is None:
            k = (group, rid, idx)
            self.key[b] = k
            self.cached[k] = b

    def grow(self, rid: int, c0: int, c1: int, inflight: int) -> None:
        st = self.state(rid)
        first = max(0, (c1 - inflight - WINDOW + 1) // SB)
        for grp in range(GROUPS):                      # `remove_skipped_blocks` runs before the allocation
            old = sorted((i for i in st["s"][grp] if i < first), reverse=True)
            self.release([st["s"][grp].pop(i) for i in old])
        def take() -> int:                             # a filler block stands for a block the model already counts
            return st["fill"].pop() if st["fill"] else self.alloc()
        for i in range(math.ceil(c1 / self.gb)):
            if i not in st["g"]:
                st["g"][i] = take()
            if (i + 1) * self.gb <= c1:
                self._mark(0, rid, i, st["g"][i])
        for grp in range(GROUPS):
            for i in range(first, math.ceil(c1 / SB)):
                if i not in st["s"][grp]:
                    st["s"][grp][i] = take()
                if (i + 1) * SB <= c1:
                    self._mark(1 + grp, rid, i, st["s"][grp][i])

    def free_request(self, rid: int) -> None:
        st = self.live.pop(rid, None)
        if st is None:
            return
        self.release(st["fill"])
        groups = [sorted(st["g"].items(), reverse=True)] + [sorted(g.items(), reverse=True) for g in st["s"]]
        if not self.global_first:
            groups = groups[1:] + groups[:1]
        for items in groups:
            self.release([b for _, b in items])

    def count(self, rid: int) -> int:
        st = self.live.get(rid)
        if st is None:
            return 0
        return len(st["g"]) + sum(len(g) for g in st["s"]) + len(st["fill"])

    def sync_fill(self, rid: int, target: int) -> None:
        st = self.state(rid)
        while self.count(rid) < target:
            st["fill"].append(self.alloc())
        while self.count(rid) > target and st["fill"]:
            self.release([st["fill"].pop()])

    def _sliding_cached(self, rid: int, i: int) -> bool:
        return all((1 + g, rid, i) in self.cached for g in range(GROUPS))

    def find_hit(self, rid: int, p: int) -> int:
        """The hybrid fixed point of `find_longest_cache_hit`, with max length p - 1."""
        align = math.lcm(SB, self.gb)
        length = (p - 1) // align * align
        while True:
            n = 0
            while n < length // self.gb and (0, rid, n) in self.cached:
                n += 1
            cur = min(length, n * self.gb) // align * align
            run, found = 0, None
            for i in range(cur // SB - 1, -1, -1):
                if self._sliding_cached(rid, i):
                    if run == 0 and ((i + 1) * SB) % align != 0:
                        continue
                    run += 1
                    if run >= NEED:
                        found = (i + run) * SB
                        break
                else:
                    run = 0
            if found is None:
                found = run * SB // align * align
            cur = min(cur, found)
            if cur == length:
                return cur
            length = cur
            if length == 0:
                return 0

    def adopt_hit(self, rid: int, hit: int) -> None:
        st = self.state(rid)
        for i in range(hit // self.gb):
            b = self.cached[(0, rid, i)]
            self.free.pop(b, None)
            st["g"][i] = b
        lo = max(0, (hit - WINDOW + 1) // SB)
        for grp in range(GROUPS):
            for i in range(lo, hit // SB):
                b = self.cached[(1 + grp, rid, i)]
                self.free.pop(b, None)
                st["s"][grp][i] = b


def inflight_of(sim: ps.PoolSim, r: ps.Req) -> int:
    h = r.hist[-sim.cap_batches:]
    return sum(h) if any(n > 1 for n in h) else 1


def replay(pool: int, layout: str, gen_fn, horizon: int, decode_only: bool, global_first: bool) -> dict:
    sim = ps.PoolSim(pool, layout, decode_only=decode_only, prefix_cache="engine")
    probe = ps.PoolSim(pool, "stock")
    ref = Ref(pool, sim.gb, global_first)
    it = gen_fn()
    out = {"readmits": 0, "both_zero": 0, "both_hit": 0, "model_only": 0, "ref_only": 0, "model_tokens": 0,
           "ref_tokens": 0, "prefill_victims": 0, "max_free_diff": 0}
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
        victims = [r for r in before if r.rid not in now]
        for v in victims:
            if v.cpre < v.p - 1:
                out["prefill_victims"] += 1
        vq = list(victims)
        vids = {v.rid for v in victims}
        for r in before:                               # growth in running order; a victim is freed when blocks run out
            if r.rid in vids:
                continue
            need = math.ceil(r.c / sim.gb) - math.ceil(c_before[r.rid] / sim.gb) \
                + GROUPS * (math.ceil(r.c / SB) - math.ceil(c_before[r.rid] / SB))
            while len(ref.free) < need and vq:
                ref.free_request(vq.pop().rid)
            if r.c != c_before[r.rid]:
                ref.grow(r.rid, c_before[r.rid], r.c, inflight_of(sim, r))
        while vq:
            ref.free_request(vq.pop().rid)
        admitted = [r for r in sim.running + finished if r.rid in waiting_before and r.seq >= 0
                    and r.rid not in c_before]
        for r in admitted:
            n = r.hist[-1] if r.hist else 0
            if r.npre > 0:
                model_hit = r.c - n
                ref_hit = ref.find_hit(r.rid, r.p)
                out["readmits"] += 1
                out["model_tokens"] += model_hit
                out["ref_tokens"] += ref_hit
                key = ("both_hit" if ref_hit else "model_only") if model_hit else ("ref_only" if ref_hit else "both_zero")
                out[key] += 1
                if ref_hit:
                    ref.adopt_hit(r.rid, ref_hit)
            ref.grow(r.rid, 0, r.c, inflight_of(sim, r))
        for r in finished:
            ref.free_request(r.rid)
        for r in sim.running:
            ref.sync_fill(r.rid, r.h)
        out["max_free_diff"] = max(out["max_free_diff"], abs(len(ref.free) - sim.free()))
    out["short"] = ref.short
    out["preemptions"] = sim.stats["preemptions"]
    return out


CELLS = (("D", "M2", 8_724, "compact"), ("D", "D3", 14_672, "compact"), ("D", "D1", 6_727, "stock"),
         ("U", "D4", 6_727, "compact"), ("D", "D2", 11_314, "stock"), ("D", "M1", 19_027, "compact"))


def main() -> None:
    rows = []
    for scenario, name, pool, layout in CELLS:
        for gf in (True, False):
            r = replay(pool, layout, e64.gen_factory(name, 0), 60_000, scenario == "D", gf)
            r.update(scenario=scenario, workload=name, pool=pool, layout=layout, global_first=gf)
            rows.append(r)
            print(json.dumps(r), flush=True)
    tot = {k: sum(r[k] for r in rows) for k in ("readmits", "both_zero", "both_hit", "model_only", "ref_only",
                                                 "model_tokens", "ref_tokens", "prefill_victims", "short")}
    tot["max_free_diff"] = max(r["max_free_diff"] for r in rows)
    Path(sys.argv[1]).write_text(json.dumps({"cells": rows, "total": tot}, indent=1) + "\n")
    print("TOTAL", json.dumps(tot))


if __name__ == "__main__":
    main()
