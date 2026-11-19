"""E-63 block-pool and scheduler model of vLLM 3966924d5 for Gemma 4 31B (BF16 KV, stock or compact K=V cache).

The rules copy `experiments/gemma4_kv_engine/run_e57.StubEngine` (the gate model that predicted the four E-57
batches) and add decode growth and preemption. README.md Section 4 (as amended by A1). Pure Python.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterator

SBLOCK = 16          # sliding block (tokens)
GROUPS = 5           # sliding groups of 10 layers each
WINDOW = 1_024       # sliding window
WINDOW_CHUNKS = 64   # E-67: cdiv(WINDOW, SBLOCK): the sliding chunks an offload hit needs
GBLOCK = {"stock": 32, "compact": 48}   # global block (tokens): stock unify ratio 2; compact B3 plan
BLOCK_BYTES = 2_621_440                  # one pool block: one page of each of the 10 layers of a group
T_DECODE, T_PREFILL_2048 = 0.02, 0.5     # A1: the time weights (seconds) of a step: fixed part + per 2,048 prefill tokens


@dataclass
class Req:
    rid: int
    base: int           # the original prompt
    m: int              # output tokens wanted
    p: int = 0          # tokens to prefill (base, or base + output so far after a preemption)
    c: int = 0          # computed tokens
    out: int = 0        # output tokens so far
    hist: list = field(default_factory=list)
    seq: int = -1       # admission order (for the preemption victim)
    npre: int = 0       # times preempted
    h: int = 0          # held blocks (cached each step)
    qstart: int = -1    # E-66: free-queue position of the first hashed block freed by the last preemption
    nhash: int = 0      # E-66: hashed blocks freed by the last preemption
    gl: int = 0         # E-66: global blocks held at the last preemption
    cpre: int = 0       # E-66: computed tokens at the last preemption
    ev: int = 0         # E-66: hashed blocks of the last preemption already reused
    og: int = 0         # E-67: global chunks in the CPU tier
    os0: int = 0        # E-67: first sliding block in the CPU tier
    os1: int = 0        # E-67: end (exclusive) of the sliding blocks in the CPU tier
    ob: int = 0         # E-67: CPU blocks held by this request
    orank: int = -1     # E-67: the LRU rank: set at the first store, kept after an eviction, renewed at finish
    ohole: bool = False  # E-67: evicted while it had computed tokens; nothing is stored until it restarts at token 0


class PoolSim:
    def __init__(self, blocks: int, layout: str, budget: int = 2_048, full_isl: bool = True,
                 max_num_seqs: int = 256, cap_batches: int = 2, watermark: float = 0.0, decode_only: bool = False,
                 prefix_cache: str = "none", offload: tuple[bool, float] | None = None):
        self.pool, self.gb, self.budget = blocks, GBLOCK[layout], budget
        self.offload = offload               # E-67: (offload_prompt_only, CPU tier bytes), or None
        self.cpu: dict[int, Req] = {}        # E-67: the requests in the CPU tier
        self.cpu_heap: list[tuple[int, int]] = []         # E-67: (LRU rank, rid), lazily invalidated
        self.cpu_next = 0
        self.cpu_blocks = 0
        self.cpu_cap = int(offload[1] // BLOCK_BYTES) if offload else 0
        self.step_load = self.step_store = 0
        self.prefix_cache = prefix_cache     # E-66: none | engine | global | perfect (re-admission of a victim only)
        self.popped = 0                      # E-66: the free-queue head position (blocks handed out so far)
        self.decode_only = decode_only       # E-64: a disaggregated decode node (the prompt KV arrives computed)
        self.full_isl, self.max_num_seqs = full_isl, max_num_seqs
        self.cap = math.ceil((WINDOW - 1 + cap_batches * budget) / SBLOCK) + 1
        self.cap_batches = cap_batches       # A2: the chunks in flight (2 with async scheduling, 1 with sync)
        self.wm = int(watermark * blocks)
        self.waiting: list[Req] = []
        self.running: list[Req] = []
        self.done: list[Req] = []
        self.total = 0                       # held blocks of the running requests
        self.next_seq = 0
        self.n_added = 0
        self.stats = {"steps": 0, "preemptions": 0, "recomputed_tokens": 0, "hol_stops": 0, "decoded": 0,
                      "max_held": 0, "readmits": 0, "hits": 0, "hit_tokens": 0, "cpu_evictions": 0}

    # --- block arithmetic -------------------------------------------------------------------------------------
    def reservation(self, p: int, remote: bool = True) -> int:
        if self.decode_only and remote:          # E-64: the first admission on a decode node (KV from outside)
            return self.steady(p)
        n = p if self.full_isl else min(p, self.budget)
        return math.ceil(n / self.gb) + GROUPS * min(math.ceil(n / SBLOCK), self.cap)

    def held(self, r: Req) -> int:
        if r.c == 0:
            return 0
        h = r.hist[-self.cap_batches:]
        inflight = sum(h) if any(n > 1 for n in h) else 1
        slide = min(math.ceil(r.c / SBLOCK), math.ceil((WINDOW - 1 + inflight) / SBLOCK) + 1)
        return math.ceil(r.c / self.gb) + GROUPS * slide

    def steady(self, tokens: int) -> int:
        """Blocks of one decoding request that holds `tokens` tokens."""
        return math.ceil(tokens / self.gb) + GROUPS * min(math.ceil(tokens / SBLOCK),
                                                          math.ceil(WINDOW / SBLOCK) + 1)

    def free(self) -> int:
        return self.pool - self.total

    # --- E-66: prefix-cache hit of a re-admitted victim ----------------------------------------------------------
    def gross_new(self, c0: int, n: int) -> int:
        """Blocks taken from the free queue when a request goes from c0 to c0 + n computed tokens."""
        c1 = c0 + n
        slide = min(math.ceil(c1 / SBLOCK) - math.ceil(c0 / SBLOCK), math.ceil((WINDOW - 1 + n) / SBLOCK) + 1)
        return math.ceil(c1 / self.gb) - math.ceil(c0 / self.gb) + GROUPS * slide

    def evicted(self, r: Req) -> int:
        return min(r.nhash, max(r.ev, self.popped - r.qstart, 0))

    def hit_tokens(self, r: Req) -> int:
        """Tokens a victim can reuse at re-admission (a multiple of the global block, at most p - 1).
        engine: vLLM needs the global prefix and a full window of sliding blocks at the hit point. Blocks are freed
          newest first and reused first, and older sliding blocks left the window long ago, so one reused hashed block
          removes the hit (an upper bound: the alignment of the window to the hit point is not checked).
        global: an optimistic bound: only the surviving global prefix counts; sliding blocks are reused first.
        perfect: nothing is ever reused."""
        if self.prefix_cache == "none" or r.qstart < 0:
            return 0
        full = min((r.p - 1) // self.gb, r.cpre // self.gb) * self.gb
        e = self.evicted(r)
        if self.prefix_cache == "perfect":
            return full
        if self.prefix_cache == "engine":
            return full if e == 0 else 0
        lost_global = max(0, e - (r.nhash - min(r.gl, r.nhash)))
        return max(0, min(full, (r.gl - lost_global) * self.gb))

    # --- E-67: CPU offload tier ----------------------------------------------------------------------------------
    def cpu_add(self, r: Req, blocks: int) -> None:
        """Store `blocks` more blocks of r. The rank is set at the first store and kept after an eviction (E-67 A1 B1:
        `lru.py` keeps the request rank in the request context); the lowest ranks are evicted."""
        if blocks <= 0:
            return
        if r.orank < 0:
            r.orank = self.cpu_next
            self.cpu_next += 1
        if r.rid not in self.cpu:
            self.cpu[r.rid] = r
            heapq.heappush(self.cpu_heap, (r.orank, r.rid))
        r.ob += blocks
        self.cpu_blocks += blocks
        self.step_store += blocks
        while self.cpu_blocks > self.cpu_cap:
            rank, rid = heapq.heappop(self.cpu_heap)
            q = self.cpu.get(rid)
            if q is None or q.orank != rank:
                continue
            del self.cpu[rid]
            self.cpu_blocks -= q.ob
            q.ob = q.og = q.os0 = q.os1 = 0
            q.ohole = True
            self.stats["cpu_evictions"] += 1

    def cpu_renew(self, r: Req) -> None:
        """A finished request gets a new LRU rank."""
        r.orank = self.cpu_next
        self.cpu_next += 1
        heapq.heappush(self.cpu_heap, (r.orank, r.rid))

    def offload_arrive(self, r: Req) -> None:
        """The prompt KV arrived from a prefill node: the global prefix and the last window are live, so they are stored."""
        r.og, r.os0, r.os1 = r.p // self.gb, max(0, r.p - WINDOW + 1) // SBLOCK, r.p // SBLOCK
        self.cpu_add(r, r.og + GROUPS * (r.os1 - r.os0))

    def offload_store(self, r: Req, c0: int, c1: int) -> None:
        """A local step computed the tokens [c0, c1) of r."""
        if c0 == 0:
            r.ohole = False
        if r.ohole:
            return
        lim = min(c1, r.base) if self.offload[0] else c1
        add = 0
        g = lim // self.gb
        if g > r.og:
            add, r.og = g - r.og, g
        lo, hi = c0 // SBLOCK, lim // SBLOCK
        if hi > lo:
            if r.os1 <= r.os0:
                add += GROUPS * (hi - lo)
                r.os0, r.os1 = lo, hi
            else:
                n0, n1 = min(r.os0, lo), max(r.os1, hi)
                add += GROUPS * ((n1 - n0) - (r.os1 - r.os0))
                r.os0, r.os1 = n0, n1
        self.cpu_add(r, add)

    def offload_hit(self, r: Req) -> int:
        """Tokens of a re-admitted request that load from the CPU tier (`_lookup_complete_chunks` at the pin)."""
        if r.npre == 0 or r.rid not in self.cpu:
            return 0
        h = min(r.p - 1, r.og * self.gb, r.os1 * SBLOCK)     # E-67 A1 F2: not rounded down to the global block
        if h < self.gb:
            return 0
        chunks = math.ceil(h / SBLOCK)
        window = max(WINDOW_CHUNKS, math.ceil((WINDOW - 1 + -h % SBLOCK) / SBLOCK))
        return h if max(0, chunks - window) >= r.os0 else 0

    # --- scheduler --------------------------------------------------------------------------------------------
    def add(self, p: int, m: int) -> None:
        self.waiting.append(Req(rid=self.n_added, base=p, m=m, p=p))
        self.n_added += 1

    def step(self) -> dict:
        """One scheduler step. Returns {decoding, prefill_tokens, evaluated, blocked}: the requests that decoded one
        token, the prefill tokens computed, whether the admission loop looked at a waiting request, and whether the
        loop stopped because the head did not fit the free blocks."""
        budget = self.budget
        self.step_load = self.step_store = 0
        sched: list[tuple[Req, int]] = []
        decoders: set[int] = set()
        ctx: dict[int, int] = {}
        start: dict[int, int] = {}                   # E-65: the context at the start of each prefill chunk
        for r in self.running:                       # decodes first
            if r.c >= r.p:
                sched.append((r, 1))
                budget -= 1
                decoders.add(r.rid)
                ctx[r.rid] = r.c                     # E-64: the context this decode step reads
        for r in self.running:                       # then running prefills
            if r.c < r.p and budget > 0:
                n = min(r.p - r.c, budget)
                sched.append((r, n))
                start[r.rid] = r.c
                budget -= n
        track = self.prefix_cache != "none"
        gross = 0
        for r, n in sched:
            if track:
                gross += self.gross_new(r.c, n)
            r.c += n
            r.hist = (r.hist + [n])[-2:]
        self.total = 0
        for r in self.running:
            r.h = self.held(r)
            self.total += r.h
        deficit = max(0, -self.free())
        if track:                                    # E-66: blocks taken before a victim's blocks become free
            self.popped += max(0, gross - deficit)
            tail = self.popped + max(self.free(), 0)
        # Growth beyond the pool: preempt the most recently admitted running request (recompute).
        preempted = False
        while self.running and self.free() < 0:
            victim = max(self.running, key=lambda q: q.seq)
            self.running.remove(victim)
            self.total -= victim.h
            preempted = True
            self.stats["preemptions"] += 1
            victim.npre += 1
            self.stats["recomputed_tokens"] += victim.c
            if track:                                # E-66: unhashed tail blocks go to the front, hashed ones behind
                unhashed = min(1 + GROUPS, victim.h)
                victim.qstart, victim.nhash, victim.ev = tail, victim.h - unhashed, 0
                victim.gl, victim.cpre = math.ceil(victim.c / self.gb), victim.c
                tail += victim.nhash
                self.popped -= unhashed
            victim.p, victim.c, victim.hist, victim.h = victim.base + victim.out, 0, [], 0   # A1 B2
            sched = [(q, n) for q, n in sched if q is not victim]
            self.waiting.insert(0, victim)
        if track:
            self.popped += deficit                   # E-66: the growth that forced the preemption now takes its blocks
            for q in self.waiting:
                if q.qstart >= 0:
                    q.ev = max(q.ev, self.popped - q.qstart)
        decoding = sum(1 for q, _ in sched if q.rid in decoders)
        prefill_tokens = sum(n for q, n in sched if q.rid not in decoders)
        # Admissions in FCFS order; none in a step that preempted (A1 F1, vLLM `if not preempted_reqs`).
        evaluated = blocked = False
        while (not preempted and self.waiting and (budget > 0 or self.decode_only)
               and len(self.running) < self.max_num_seqs):
            r = self.waiting[0]
            # E-64 A1 (R-99 B1): on a decode node a preempted request prefills again locally (vLLM NIXL clears
            # `do_remote_prefill` after the first allocation), through the token budget and the full-ISL reservation.
            remote = self.decode_only and r.npre == 0
            if not remote and budget <= 0:
                break
            evaluated = True
            free = self.free()
            hit = 0 if remote else (self.offload_hit(r) if self.offload else self.hit_tokens(r))
            need = self.reservation(r.p, remote)
            if hit > 0:                              # E-66: the sliding blocks before the hit point are not needed
                rest = min(r.p - hit, self.cap_batches * self.budget)
                need = math.ceil(r.p / self.gb) + GROUPS * min(math.ceil(r.p / SBLOCK),
                                                               math.ceil((WINDOW - 1 + rest) / SBLOCK) + 1)
            if free - self.wm < need:
                blocked = True
                if any(self.reservation(q.p, self.decode_only and q.npre == 0) <= free - self.wm
                       for q in self.waiting[1:]):
                    self.stats["hol_stops"] += 1
                break
            self.waiting.pop(0)
            if remote:                               # E-64: the prompt KV arrives computed; decode from next step
                r.c, r.hist, r.seq = r.p, [1], self.next_seq
                r.h = self.held(r)
                self.total += r.h
                self.next_seq += 1
                self.running.append(r)
                if track:
                    self.popped += r.h
                if self.offload:
                    self.offload_arrive(r)
                continue
            if self.offload and r.npre > 0:          # E-67: the hit blocks load from the CPU tier
                self.stats["readmits"] += 1
                if hit > 0:
                    self.stats["hits"] += 1
                    self.stats["hit_tokens"] += hit
                    self.step_load += hit // self.gb + GROUPS * min(hit // SBLOCK, WINDOW_CHUNKS)
            n = min(r.p - hit, budget)
            r.c, r.hist, r.seq = hit + n, [n], self.next_seq
            r.h = self.held(r)
            self.total += r.h
            self.next_seq += 1
            self.running.append(r)
            sched.append((r, n))
            start[r.rid] = hit
            prefill_tokens += n
            budget -= n
            if track:                                # E-66: hit blocks leave the queue; the rest are new blocks
                removed = 0
                if r.qstart >= 0:
                    self.stats["readmits"] += 1
                    if hit > 0:
                        removed = r.nhash - self.evicted(r)
                        self.stats["hits"] += 1
                        self.stats["hit_tokens"] += hit
                        for q in self.waiting:
                            if q.qstart > r.qstart:
                                q.qstart -= removed
                self.popped += max(0, r.h - removed)
                r.qstart = -1
        if self.offload:                             # E-67: the blocks computed in this step go to the CPU tier
            for r, n in sched:
                self.offload_store(r, r.c - n, r.c)
        # Outputs: a request that has all its tokens computed after this step samples one token.
        for r, _ in sched:
            if r.c >= r.p:
                r.out += 1
                self.stats["decoded"] += 1
                if r.out >= r.m:
                    self.running.remove(r)
                    self.total -= r.h
                    self.done.append(r)
                    if self.offload and r.rid in self.cpu:
                        self.cpu_renew(r)
        self.stats["max_held"] = max(self.stats["max_held"], self.pool - self.free())
        self.stats["steps"] += 1
        return {"decoding": decoding, "prefill_tokens": prefill_tokens, "evaluated": evaluated, "blocked": blocked,
                "decode_ctx": [ctx[q.rid] for q, _ in sched if q.rid in decoders],
                "prefill_chunks": [(n, start[q.rid]) for q, n in sched if q.rid not in decoders],
                "load_bytes": self.step_load * BLOCK_BYTES, "store_bytes": self.step_store * BLOCK_BYTES}


def closed_form_batch(blocks: int, layout: str, prompt: int, **kw) -> int:
    """b = 1 + floor((pool - watermark - reservation) / steady) for equal requests (0 if one does not fit)."""
    s = PoolSim(blocks, layout, **kw)
    res, st = s.reservation(prompt), s.steady(prompt)
    if res > blocks - s.wm:
        return 0
    return 1 + (blocks - s.wm - res) // st


def run_saturated(blocks: int, layout: str, reqs: list[tuple[int, int]], max_steps: int = 2_000_000,
                  **kw) -> dict:
    """All requests wait at time 0; runs to the end. Used for the FIX grid and the V1 tie."""
    sim = PoolSim(blocks, layout, **kw)
    for p, m in reqs:
        sim.add(p, m)
    dec_max, peak_single = 0, 0
    while sim.waiting or sim.running:
        if sim.stats["steps"] >= max_steps:
            raise RuntimeError(f"max_steps reached: pool {blocks} {layout}")
        if not sim.running and sim.waiting and sim.reservation(sim.waiting[0].p) > sim.pool:
            raise ValueError("a request does not fit the empty pool")
        d = sim.step()
        dec_max = max(dec_max, d["decoding"])
        if len(sim.running) == 1:
            peak_single = max(peak_single, sim.running[0].h)
    out = dict(sim.stats)
    out.update(max_decoding=dec_max, peak_single=peak_single, finished=len(sim.done),
               outputs_wanted=sum(m for _, m in reqs))
    return out


def run_stream(blocks: int, layout: str, gen: Callable[[], Iterator[tuple[int, int]]], horizon: int,
               warmup: int, keep_waiting: int = 8, time_fns: dict | None = None, **kw) -> dict:
    """A1 B1: the waiting queue is refilled from the seeded generator, so the backlog never drains. The window is the
    steps [warmup, horizon), the same for both layouts. Requests whose STOCK reservation of prompt + output does not
    fit the empty pool are skipped by the generator (the same set for both layouts)."""
    sim = PoolSim(blocks, layout, **kw)
    # A3: the skip rule uses the same budget, cap and watermark as the run (full-ISL, stock layout).
    # E-64 A1: on a decode node the probe also needs the local re-prefill to fit (a preempted request), so it drops
    # `decode_only` and uses the full-ISL reservation.
    probe = PoolSim(blocks, "stock", **{k: v for k, v in kw.items() if k not in ("full_isl", "decode_only")})
    it, skipped = gen(), 0

    def refill() -> None:
        nonlocal skipped
        while len(sim.waiting) < keep_waiting:
            p, m = next(it)
            if probe.reservation(p + m) > blocks - probe.wm:
                skipped += 1
                continue
            sim.add(p, m)

    dec = t = 0.0
    load_b = store_b = 0
    evaluated = blocked = seq_bound = 0
    extra = {k: 0.0 for k in (time_fns or {})}
    for s in range(horizon):
        refill()
        d = sim.step()
        assert sim.free() >= 0, "V2: held blocks exceed the pool"
        if s >= warmup:
            dec += d["decoding"]
            load_b += d["load_bytes"]
            store_b += d["store_bytes"]
            t += T_DECODE + T_PREFILL_2048 * d["prefill_tokens"] / 2_048
            for k, fn in (time_fns or {}).items():
                extra[k] += fn(d)
            evaluated += d["evaluated"]
            blocked += d["blocked"]
            seq_bound += len(sim.running) >= sim.max_num_seqs
    n = horizon - warmup
    out = dict(sim.stats)
    out.update(decode_per_step=dec / n, decode_per_second=dec / t, pool_bound_share=blocked / max(evaluated, 1),
               evaluated_steps=evaluated, skipped=skipped, finished=len(sim.done), seq_bound_share=seq_bound / n,
               load_bytes=load_b, store_bytes=store_b, window_decoded=dec)
    for k, tk in extra.items():                      # E-64: decode tokens per second under each time model
        out[f"decode_per_second_{k}"] = dec / tk if tk > 0 else 0.0
    return out
