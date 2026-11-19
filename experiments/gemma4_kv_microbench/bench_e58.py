"""E-58: where the step cost of the E-57 compact K=V cache comes from (README Sections 4 and 5).

A weight-free microbenchmark on one GPU at the E-57 decode shapes of the Gemma 4 31B global layers (32 query heads, 4
KV heads, head 512, 10 layers): D32 (3 requests of 32,768 tokens) and D64 (2 requests of 65,536 tokens). The vLLM build
and the call arguments are those of E-57 (`TritonAttentionImpl.forward` and `do_kv_cache_update` at 3966924d5).

Arms (README Section 4.4):
- R: the attention read (`unified_attention`, the 3D decode path) for 4 cache layouts that hold the same logical K and
  V: S32 (stock row 1,024, block 32, page 262,144 B: the E-57 stock cache), S48 (stock row, block 48), C32 (compact row
  640 read through the two views, block 32, page padded to 262,144 B) and C48 (compact row, block 48, padded page: the
  E-57 compact cache). The GPU time of 10 layer calls (CUDA events).
- W: the cache write (`triton_reshape_and_cache_flash`) of the decode tokens, S32 and C48 (GPU time).
- P: the read and the write of S32 and C48 through the E-57 wrappers (`compact_engine.install`) and as raw calls (CPU
  issue time, with the GPU kept busy so that the issue loop never waits).
- H: the hook transforms (`compact_engine.ViewsSplice`) on modules that do no attention: no hooks, mode OFF and mode
  VIEWS; and the VIEWS transform without the module hooks, eager and with torch.compile. The CPU issue time, the GPU
  time (eager) and the GPU time of a CUDA graph replay.
- X: one global-layer step (10 modules: the hooks, the write and the read) as STOCK (S32, no hooks), EMULV (S32, hooks
  VIEWS) and COMPACT (C48, hooks VIEWS, the wrappers): the CPU issue time, the GPU time (eager), the wall time with a
  sync after each step and the GPU time of a CUDA graph replay.

The timed arms run in interleaved rounds. --stub runs the same flow on the Mac with a reference attention and small
shapes (no CUDA and no vLLM; only perf_counter times, no graph or compile arms).

Usage: bench_e58.py --out <dir> [--shapes D32,D64] [--layers 10] [--rounds 3] [--rehearsal] [--stub]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "gemma4_kv_engine"), str(HERE.parent / "gemma4_kv_model")]
import compact_engine as ce  # noqa: E402
import compact_layout as cl  # noqa: E402

HQ, HK, D = 32, 4, 512
ROW_S, ROW_C = 2 * D, cl.ROW           # 1,024 and 640 values per KV head and token
PAGE_ELEMS = 262_144 // 2              # the E-57 pool page (the sliding page) in BF16 elements
W_VALUE = 0.0615234375                 # a k_norm value of the 31B (timing only)
SEQ_THRESHOLD_3D = 128 // 16           # triton_attn.py: MIN_LAUNCH_GRID_SIZE_2D // the model's KV heads (16)
SEGMENTS, SCRATCH_TOKENS = 16, 8       # NUM_PAR_SOFTMAX_SEGMENTS; min(seq_threshold_3D, max_num_seqs)
SHAPES = {"D32": (3, 32_768), "D64": (2, 65_536)}
STUB_SHAPES = {"D32": (2, 192), "D64": (2, 320)}
LAYOUTS = {"S32": ("stock", 32), "S48": ("stock", 48), "C32": ("compact", 32), "C48": ("compact", 48)}
ITERS, CPU_STEPS, CPU_REPEATS, WARM = 50, 3, 9, 3      # A1 (R-84 F3): 3 steps per CPU loop (was 5)
SLEEP_CYCLES = 100_000_000             # about 70 ms of GPU time at 1.4 GHz: longer than any CPU issue loop here
GUARD_BATCH, GUARD_LEN = 8, 2_048      # A1 (R-84 F5): the scratch check (8 decode requests, the 3D path)
ENGINE_HEADDIM = 256                   # vLLM 3966924d5 sizes the 3D scratch with model_config.get_head_size() = 256


def page_elems(kind: str, block: int) -> int:
    """Elements per block page: the pool page (262,144 B), or more if the content needs more (S48: 393,216 B)."""
    return max(PAGE_ELEMS, block * HK * (ROW_S if kind == "stock" else ROW_C))


class Layout:
    """One layer's paged KV cache of one layout: [blocks, HK, block, row] with the LBNHC strides of the engine (the
    token index outside the head index), and the block table (each request in a contiguous range of blocks, with one
    more slot for the decode write)."""

    def __init__(self, name: str, batch: int, length: int, device):
        self.name = name
        self.kind, self.block = LAYOUTS[name]
        self.row = ROW_S if self.kind == "stock" else ROW_C
        self.per_req = math.ceil((length + 1) / self.block)
        self.num_blocks = batch * self.per_req
        self.page = page_elems(self.kind, self.block)
        raw = torch.zeros(self.num_blocks * self.page, dtype=torch.bfloat16, device=device)
        self.cache = torch.as_strided(raw, (self.num_blocks, HK, self.block, self.row),
                                      (self.page, self.row, HK * self.row, 1))
        self.block_table = (torch.arange(self.num_blocks, dtype=torch.int32, device=device)
                            .view(batch, self.per_req).contiguous())
        self.slots = torch.tensor([(r * self.per_req + length // self.block) * self.block + length % self.block
                                   for r in range(batch)], dtype=torch.int64, device=device)

    def split(self):
        """What the backend passes to the kernels: kv_cache.transpose(1, 2).split(head_size)."""
        return self.cache.transpose(1, 2).split(D, dim=-1)

    def views(self):
        """The raw read and write operands: the K view and the V view (for a compact row, the fixed V view)."""
        kc, vc = self.split()
        if self.kind == "compact":
            vc, fixed = ce.fix_value_cache(kc, vc)
            assert fixed, "the compact split did not give the 128-value tail"
        return kc, vc

    def fill(self, x: torch.Tensor) -> None:
        """x: [batch, length, HK, 640], the logical compact rows. A stock row holds K = x[..., :512] and V = x[...,
        128:]: the two views of the compact row, so every layout holds the same logical K and V."""
        rows = x if self.kind == "compact" else torch.cat([x[..., :D], x[..., cl.NR:]], dim=-1)
        batch, length = x.shape[0], x.shape[1]
        for r in range(batch):
            pad = torch.zeros(self.per_req * self.block, HK, self.row, dtype=rows.dtype, device=rows.device)
            pad[:length] = rows[r]
            lo = r * self.per_req
            self.cache[lo:lo + self.per_req] = pad.view(self.per_req, self.block, HK, self.row).permute(0, 2, 1, 3)


def read_kwargs(q, kc, vc, out, lay: Layout, length: int, cu, seqused, scratch, ones) -> dict:
    """The arguments of `unified_attention` as `TritonAttentionImpl.forward` passes them for a global layer at decode
    (BF16 cache, no sinks, no softcap, no window, the 3D scratch of the metadata builder)."""
    return dict(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seqused, max_seqlen_k=length,
                softmax_scale=1.0, causal=True, window_size=(-1, -1), block_table=lay.block_table, softcap=0.0,
                q_descale=None, k_descale=ones, v_descale=ones, seq_threshold_3D=SEQ_THRESHOLD_3D,
                num_par_softmax_segments=SEGMENTS, softmax_segm_output=scratch[0], softmax_segm_max=scratch[1],
                softmax_segm_expsum=scratch[2])


def ref_read(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size,
             block_table, softcap, q_descale, k_descale, v_descale, **unused):
    """The semantics of `unified_attention` for one decode token per sequence (the stub; float32 math)."""
    block = v.shape[1]
    g = q.shape[1] // k.shape[2]
    for r in range(len(seqused_k)):
        idx = torch.arange(int(seqused_k[r]))
        blocks, offs = block_table[r, idx // block].long(), idx % block
        kk = k[blocks, offs].float().repeat_interleave(g, dim=1)
        vv = v[blocks, offs].float().repeat_interleave(g, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[r].float(), kk) * softmax_scale, dim=-1)
        out[r] = torch.einsum("hl,lhd->hd", p, vv).to(out.dtype)


def ref_write(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale, **unused):
    """The semantics of `triton_reshape_and_cache_flash` (the stub): K first, then V."""
    block = key_cache.shape[1]
    for i, s in enumerate(slot_mapping.tolist()):
        key_cache[s // block, s % block] = key[i]
        value_cache[s // block, s % block] = value[i]


def stub_install(ns) -> None:
    """The stub form of the E-57 wrappers: the same `fix_value_cache` on the read and the write."""
    orig_ua, orig_w = ns.unified_attention, ns.triton_reshape_and_cache_flash

    def unified_attention(**kw):
        v, fixed = ce.fix_value_cache(kw["k"], kw["v"])
        if fixed:
            kw["v"] = v
        return orig_ua(**kw)

    def write(key, value, key_cache, value_cache, *rest, **kw):
        value_cache, _ = ce.fix_value_cache(key_cache, value_cache)
        return orig_w(key, value, key_cache, value_cache, *rest, **kw)

    ns.unified_attention, ns.triton_reshape_and_cache_flash = unified_attention, write


class GlobalLayer(torch.nn.Module):
    """One global Attention layer at decode: forward(q, k, v) with [T, H x 512] inputs. With a layout it writes k and
    v and reads the cache through `ns` (the triton_attn module attributes, so the installed wrappers apply); without a
    layout it returns a fixed output (the hook-only arms)."""

    def __init__(self, ns, layout: Layout | None = None, length: int = 0, aux: dict | None = None,
                 out_buf: torch.Tensor | None = None):
        super().__init__()
        self.ns, self.layout, self.length, self.aux, self.out_buf = ns, layout, length, aux, out_buf
        self.num_heads, self.num_kv_heads, self.head_size = HQ, HK, D

    def forward(self, q, k, v):
        if self.layout is None:
            return self.out_buf
        t, a = q.shape[0], self.aux
        kc, vc = self.layout.split()
        self.ns.triton_reshape_and_cache_flash(k.view(t, HK, D), v.view(t, HK, D), kc, vc, self.layout.slots, "auto",
                                               a["ks"], a["vs"])
        out = torch.empty(t, HQ, D, dtype=q.dtype, device=q.device)
        self.ns.unified_attention(**read_kwargs(q.view(t, HQ, D), kc, vc, out, self.layout, self.length, a["cu"],
                                                a["seqused"], a["scratch"], a["ones"]))
        return out.view(t, HQ * D)


class Timer:
    """The timing methods (README Section 5.1). On the GPU: CUDA events, a CPU issue loop behind a GPU sleep, a wall
    loop with a sync per step and CUDA graph replays. In the stub: perf_counter, fewer steps, and no graph."""

    def __init__(self, stub: bool):
        self.stub = stub
        self.iters, self.repeats = (4, 2) if stub else (ITERS, CPU_REPEATS)
        self.graph_errors: list[str] = []

    def sync(self) -> None:
        if not self.stub:
            torch.cuda.synchronize()

    def gpu_ms(self, step, iters: int | None = None) -> float:
        iters = iters or self.iters
        for _ in range(WARM):
            step()
        self.sync()
        if self.stub:
            t0 = time.perf_counter()
            for _ in range(iters):
                step()
            return 1_000 * (time.perf_counter() - t0) / iters
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            step()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / iters

    def sleep_ms(self) -> float | None:
        """The GPU time of one `torch.cuda._sleep(SLEEP_CYCLES)` (A1, R-84 F3: the V3 loop-time bound)."""
        if self.stub:
            return None
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        torch.cuda._sleep(SLEEP_CYCLES)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b)

    def cpu_ms(self, step) -> tuple[float, bool, float]:
        """The median CPU issue time of one step over the repeats of CPU_STEPS steps, each loop behind a GPU sleep.
        Returns the time, a flag that the GPU was still busy when every loop ended, and the longest loop in ms (V3:
        a loop must stay well inside the sleep)."""
        for _ in range(WARM):
            step()
        self.sync()
        times, busy, longest = [], True, 0.0
        for _ in range(self.repeats):
            if not self.stub:
                torch.cuda._sleep(SLEEP_CYCLES)
            t0 = time.perf_counter()
            for _ in range(CPU_STEPS):
                step()
            t1 = time.perf_counter()
            if not self.stub:
                busy &= not torch.cuda.current_stream().query()
            self.sync()
            times.append(1_000 * (t1 - t0) / CPU_STEPS)
            longest = max(longest, 1_000 * (t1 - t0))
        return statistics.median(times), busy, longest

    def wall_ms(self, step, iters: int | None = None) -> float:
        iters = iters or self.iters
        for _ in range(WARM):
            step()
        self.sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
            self.sync()
        return 1_000 * (time.perf_counter() - t0) / iters

    def graph_ms(self, step, iters: int | None = None) -> float | None:
        """The GPU time of a CUDA graph replay of the step; None in the stub or if the capture fails (recorded)."""
        if self.stub:
            return None
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(WARM):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            return self.gpu_ms(g.replay, iters)
        except Exception as exc:  # recorded; the graph number of that arm is NA
            self.graph_errors.append(f"{type(exc).__name__}: {exc}"[:300])
            torch.cuda.synchronize()
            return None


def kernel_count(step, stub: bool) -> int | None:
    """The CUDA kernels of one step (the torch profiler); None if the profiler fails (a report-only number, A1)."""
    if stub:
        return None
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            step()
            torch.cuda.synchronize()
        return sum(1 for e in p.events() if str(getattr(e, "device_type", "")).endswith("CUDA"))
    except Exception:  # report only
        torch.cuda.synchronize()
        return None


def scratch_guard(raw, dev, stub: bool) -> dict:
    """A1 (R-84 F5; report only): vLLM 3966924d5 sizes the 3D decode scratch with the model head size (256 for Gemma 4),
    while the kernel indexes it with the layer head size (512). This runs one global-layer decode call with 8 requests
    (the 3D path) on a scratch of the engine size, placed at the start of a larger guard buffer, and counts the guard
    values that the kernel changed. A raw call with a scratch of the right size gives the reference output."""
    lay = Layout("S32", GUARD_BATCH, GUARD_LEN, dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    lay.fill((torch.randn(GUARD_BATCH, GUARD_LEN, HK, ROW_C, generator=gen, device=dev) * 0.5).to(torch.bfloat16))
    q = (torch.randn(GUARD_BATCH, HQ, D, generator=gen, device=dev) * 0.1).to(torch.bfloat16)
    cu = torch.arange(GUARD_BATCH + 1, dtype=torch.int32, device=dev)
    seqused = torch.full((GUARD_BATCH,), GUARD_LEN, dtype=torch.int32, device=dev)
    ones = torch.ones((), dtype=torch.float32, device=dev).expand(GUARD_BATCH, HK)
    kc, vc = lay.views()
    small = SCRATCH_TOKENS * HQ * SEGMENTS * ENGINE_HEADDIM
    guard = torch.full((SCRATCH_TOKENS * HQ * SEGMENTS * D + 1_024,), 12_345.0, dtype=torch.float32, device=dev)
    smax = torch.empty(SCRATCH_TOKENS, HQ, SEGMENTS, dtype=torch.float32, device=dev)
    sexp = torch.empty(SCRATCH_TOKENS, HQ, SEGMENTS, dtype=torch.float32, device=dev)
    engine = (guard[:small].view(SCRATCH_TOKENS, HQ, SEGMENTS, ENGINE_HEADDIM), smax, sexp)
    right = (torch.empty(SCRATCH_TOKENS, HQ, SEGMENTS, D, dtype=torch.float32, device=dev), smax.clone(),
             sexp.clone())
    outs = {}
    for name, scratch in (("engine_size", engine), ("right_size", right)):
        o = torch.empty(GUARD_BATCH, HQ, D, dtype=torch.bfloat16, device=dev)
        raw.unified_attention(**read_kwargs(q, kc, vc, o, lay, GUARD_LEN, cu, seqused, scratch, ones))
        outs[name] = o
    if not stub:
        torch.cuda.synchronize()
    tail = guard[small:]
    return {"batch": GUARD_BATCH, "length": GUARD_LEN, "engine_scratch_floats": small,
            "kernel_needs_floats": GUARD_BATCH * HQ * SEGMENTS * D,
            "guard_values_changed": int((tail != 12_345.0).sum()),
            "outputs_identical": bool(torch.equal(outs["engine_size"], outs["right_size"]))}


def run_shape(name: str, batch: int, length: int, layers: int, rounds: int, ns, raw, timer: Timer, dev,
              stub: bool, rec: dict, save) -> None:
    """All arms at one decode shape (README Section 4.4); each arm writes its record, so a failed arm keeps the rest."""
    torch.manual_seed(58)
    lays = {lo: [Layout(lo, batch, length, dev) for _ in range(layers)] for lo in LAYOUTS}
    for i in range(layers):
        gen = torch.Generator(device=dev).manual_seed(1_000 * (length // 1_024) + i)
        x = (torch.randn(batch, length, HK, ROW_C, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        for lo in LAYOUTS:
            lays[lo][i].fill(x)
        del x
    cu = torch.arange(batch + 1, dtype=torch.int32, device=dev)
    seqused = torch.full((batch,), length, dtype=torch.int32, device=dev)
    scratch = (torch.empty(SCRATCH_TOKENS, HQ, SEGMENTS, D, dtype=torch.float32, device=dev),
               torch.empty(SCRATCH_TOKENS, HQ, SEGMENTS, dtype=torch.float32, device=dev),
               torch.empty(SCRATCH_TOKENS, HQ, SEGMENTS, dtype=torch.float32, device=dev))
    ones = torch.ones((), dtype=torch.float32, device=dev).expand(batch, HK)
    scale = torch.ones((), dtype=torch.float32, device=dev)
    aux = {"cu": cu, "seqused": seqused, "scratch": scratch, "ones": ones, "ks": scale, "vs": scale}
    q = (torch.randn(batch, HQ, D, device=dev) * 0.1).to(torch.bfloat16)
    k_new = (torch.randn(batch, HK, D, device=dev) * 0.5).to(torch.bfloat16)
    v_new = (torch.randn(batch, HK, D, device=dev) * 0.5).to(torch.bfloat16)
    sh = rec.setdefault(name, {"batch": batch, "length": length, "layers": layers, "errors": [],
                               "pages": {lo: lays[lo][0].page * 2 for lo in LAYOUTS},
                               "blocks": {lo: lays[lo][0].num_blocks for lo in LAYOUTS}})

    def arm(label, fn):
        try:
            fn()
        except Exception as exc:  # recorded; the analysis marks the words that need it NA
            sh["errors"].append({"arm": label, "error": f"{type(exc).__name__}: {exc}"[:400],
                                 "trace": traceback.format_exc()[-1_500:]})
        save()

    # V1: every layout gives the same output for the same logical K and V (layer 0), raw and through the wrappers.
    outs = {}

    def v1():
        for lo in LAYOUTS:
            kc, vc = lays[lo][0].views()
            o = torch.empty(batch, HQ, D, dtype=torch.bfloat16, device=dev)
            raw.unified_attention(**read_kwargs(q, kc, vc, o, lays[lo][0], length, cu, seqused, scratch, ones))
            outs[lo] = o
        kc, vc = lays["C48"][0].split()
        o = torch.empty(batch, HQ, D, dtype=torch.bfloat16, device=dev)
        ns.unified_attention(**read_kwargs(q, kc, vc, o, lays["C48"][0], length, cu, seqused, scratch, ones))
        outs["C48_wrapped"] = o
        timer.sync()
        ref = outs["S32"].float()
        sh.setdefault("v1", {}).update({lo: {"identical": bool(torch.equal(o, outs["S32"])),
                                             "max_abs": float((o.float() - ref).abs().max())}
                                        for lo, o in outs.items()})
        sh["out_scale"] = float(ref.abs().max())
    arm("v1", v1)

    # R: the attention read, raw, 4 layouts, interleaved rounds.
    kws = {lo: [] for lo in LAYOUTS}
    for lo in LAYOUTS:
        for i in range(layers):
            kc, vc = lays[lo][i].views()
            kws[lo].append(read_kwargs(q, kc, vc, torch.empty(batch, HQ, D, dtype=torch.bfloat16, device=dev),
                                       lays[lo][i], length, cu, seqused, scratch, ones))

    def read_step(lo):
        def step():
            for kw in kws[lo]:
                raw.unified_attention(**kw)
        return step
    sh["R"] = {lo: [] for lo in LAYOUTS}
    order = list(LAYOUTS)
    sh["R_warm"] = {}                                  # A1 (R-84 B3): one untimed-for-decision warm round
    for lo in order:
        arm(f"R warm {lo}", lambda lo=lo: sh["R_warm"].__setitem__(lo, timer.gpu_ms(read_step(lo), 10)))
    for r in range(rounds):
        for lo in order[r % len(order):] + order[:r % len(order)]:
            arm(f"R {lo} round {r}", lambda lo=lo: sh["R"][lo].append(timer.gpu_ms(read_step(lo))))

    # W: the cache write of the decode tokens, raw (S32, C48).
    def write_step(lo):
        ops = [lays[lo][i].views() for i in range(layers)]

        def step():
            for i, (kc, vc) in enumerate(ops):
                raw.triton_reshape_and_cache_flash(k_new, v_new, kc, vc, lays[lo][i].slots, "auto", scale, scale)
        return step
    sh["W"] = {}
    for lo in ("S32", "C48"):
        arm(f"W {lo}", lambda lo=lo: sh["W"].__setitem__(lo, timer.gpu_ms(write_step(lo))))

    # P: the write and the read of each layer as the backend makes them (the split, then the wrapped calls: E-57) and
    # as a backend that slices the two views itself and calls the kernels directly (raw): the CPU issue time.
    def rw_step(lo, wrapped):
        def step():
            for i in range(layers):
                lay = lays[lo][i]
                if wrapped:
                    fns = ns
                    kc, vc = lay.split()
                else:
                    fns = raw
                    t = lay.cache.transpose(1, 2)
                    kc, vc = t[..., :D], t[..., lay.row - D:]
                fns.triton_reshape_and_cache_flash(k_new, v_new, kc, vc, lay.slots, "auto", scale, scale)
                fns.unified_attention(**read_kwargs(q, kc, vc, kws[lo][i]["out"], lay, length, cu, seqused, scratch,
                                                    ones))
        return step
    sh["P"] = {}
    sh["sleep_ms"] = timer.sleep_ms()
    for lo in ("S32", "C48"):
        for label in ("raw", "wrapped"):
            def p_arm(lo=lo, label=label):
                ms, busy, longest = timer.cpu_ms(rw_step(lo, label == "wrapped"))
                sh["P"][f"{lo} {label}"] = {"cpu": ms, "busy": busy, "loop_max_ms": longest}
            arm(f"P {lo} {label}", p_arm)

    # H: the hook transforms on modules that do no attention.
    xq = q.reshape(batch, HQ * D)
    xk, xv = k_new.reshape(batch, HK * D), v_new.reshape(batch, HK * D)
    out_buf = torch.zeros(batch, HQ * D, dtype=torch.bfloat16, device=dev)
    w = torch.full((D,), W_VALUE, device=dev)
    sh["H"] = {}

    def module_step(mods):
        def step():
            for m in mods:
                m(xq, xk, xv)
        return step

    def h_arm(label, mode):
        mods = [GlobalLayer(ns, out_buf=out_buf) for _ in range(layers)]
        splice = None
        if mode is not None:
            splice = ce.ViewsSplice()
            for i, m in enumerate(mods):
                splice.attach(i, m, w, (HQ, HK, D))
            splice.set_mode(mode)
        step = module_step(mods)
        ms, busy, longest = timer.cpu_ms(step)
        # A1 (R-84 B2): the NONE arm has no GPU work by design, so its graph time is 0 and no empty graph is captured.
        graph = (0.0 if not stub else None) if mode is None else timer.graph_ms(step)
        sh["H"][label] = {"cpu": ms, "busy": busy, "loop_max_ms": longest, "gpu": timer.gpu_ms(step),
                          "graph": graph, "kernels": kernel_count(step, stub)}
        if mode is None:
            sh["H"][label]["graph_note"] = "no GPU work by design; no capture"
        if splice is not None:
            splice.detach()
    for label, mode in (("NONE", None), ("OFF", "OFF"), ("VIEWS", "VIEWS")):
        arm(f"H {label}", lambda label=label, mode=mode: h_arm(label, mode))

    ix = ce.Indexer()
    wu = w.index_select(0, ix.get(dev)["unrot"])
    q3, k3, v3, o3 = q, k_new, v_new, out_buf.view(batch, HQ, D)

    def transform(q3, k3, v3, o3):
        k2, v2 = ix.write_operands(k3, v3)
        return ix.prep_query(q3, wu), k2, v2, ix.unpermute(o3)

    def fn_arm(label, fn):
        def step():
            for _ in range(layers):
                fn(q3, k3, v3, o3)
        ms, busy, longest = timer.cpu_ms(step)
        sh["H"][label] = {"cpu": ms, "busy": busy, "loop_max_ms": longest, "gpu": timer.gpu_ms(step),
                          "graph": timer.graph_ms(step), "kernels": kernel_count(step, stub)}
    arm("H EAGER_FN", lambda: fn_arm("EAGER_FN", transform))
    if not stub:
        arm("H COMPILED_FN", lambda: fn_arm("COMPILED_FN", torch.compile(transform, dynamic=False)))

    # X: one global-layer step as STOCK, EMULV and COMPACT; V1b: COMPACT equals EMULV on layer 0.
    sh["X"] = {}
    x_out = {}

    def x_arm(label, lo, mode):
        mods = [GlobalLayer(ns, lays[lo][i], length, aux) for i in range(layers)]
        splice = None
        if mode is not None:
            splice = ce.ViewsSplice()
            for i, m in enumerate(mods):
                splice.attach(i, m, w, (HQ, HK, D))
            splice.set_mode(mode)
        x_out[label] = mods[0](xq, xk, xv).clone()
        step = module_step(mods)
        ms, busy, longest = timer.cpu_ms(step)
        sh["X"][label] = {"cpu": ms, "busy": busy, "loop_max_ms": longest, "gpu": timer.gpu_ms(step),
                          "wall": timer.wall_ms(step), "graph": timer.graph_ms(step),
                          "kernels": kernel_count(step, stub)}
        if splice is not None:
            splice.detach()
    for label, lo, mode in (("STOCK", "S32", None), ("EMULV", "S32", "VIEWS"), ("COMPACT", "C48", "VIEWS")):
        arm(f"X {label}", lambda label=label, lo=lo, mode=mode: x_arm(label, lo, mode))
    if "EMULV" in x_out and "COMPACT" in x_out:
        sh.setdefault("v1", {})["X_COMPACT_vs_EMULV"] = {
            "identical": bool(torch.equal(x_out["COMPACT"], x_out["EMULV"])),
            "max_abs": float((x_out["COMPACT"].float() - x_out["EMULV"].float()).abs().max())}
    sh["bytes_read"] = {lo: batch * length * HK * 2 * D * 2 * layers for lo in LAYOUTS}       # K and V, logical
    sh["bytes_distinct"] = {lo: batch * length * HK * lays[lo][0].row * 2 * layers for lo in LAYOUTS}
    sh["graph_errors"] = list(timer.graph_errors)
    save()
    del lays, kws
    if not stub:
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shapes", default="D32,D64")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rec = {"meta": {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stub": args.stub,
                    "rehearsal": args.rehearsal, "layers": args.layers, "rounds": args.rounds,
                    "iters": ITERS, "cpu_steps": CPU_STEPS, "cpu_repeats": CPU_REPEATS,
                    "seq_threshold_3D": SEQ_THRESHOLD_3D, "segments": SEGMENTS}, "shapes": {}, "status": "running"}
    path = args.out / "bench.json"

    def save():
        path.write_text(json.dumps(rec, indent=1) + "\n")
    code = 1
    try:
        if args.stub:
            torch.set_num_threads(1)
            dev = torch.device("cpu")
            raw = SimpleNamespace(unified_attention=ref_read, triton_reshape_and_cache_flash=ref_write)
            ns = SimpleNamespace(unified_attention=ref_read, triton_reshape_and_cache_flash=ref_write)
            stub_install(ns)
            sizes = STUB_SHAPES
            rec["meta"].update(gpu="STUB", vllm="STUB")
        else:
            import vllm
            import vllm.v1.attention.backends.triton_attn as ta
            dev = torch.device("cuda")
            raw = SimpleNamespace(unified_attention=ta.unified_attention,
                                  triton_reshape_and_cache_flash=ta.triton_reshape_and_cache_flash)
            ce.install("compact")                       # wraps ta.unified_attention and the write, as in E-57
            ns = ta
            sizes = SHAPES
            rec["meta"].update(gpu=torch.cuda.get_device_name(0), vllm=vllm.__version__, torch=torch.__version__,
                               installed=ce.STATE["installed"])
        save()
        for name in [s for s in args.shapes.split(",") if s]:
            batch, length = sizes[name]
            run_shape(name, batch, length, args.layers, args.rounds, ns, raw, Timer(args.stub), dev, args.stub,
                      rec["shapes"], save)
        try:                                            # A1 (R-84 F5): the scratch check, report only, after the shapes
            rec["scratch_guard"] = scratch_guard(raw, dev, args.stub)
        except Exception as exc:  # report only
            rec["scratch_guard"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        save()
        rec["status"] = "ok"
        code = 0
    except BaseException as exc:  # recorded
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
        rec["traceback"] = traceback.format_exc()[-3_000:]
    rec["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["wrapper_counts"] = {k: ce.STATE[k] for k in ("fix_reads", "fix_writes", "pass_reads", "pass_writes")}
    save()
    print(json.dumps({"status": rec["status"], "shapes": list(rec["shapes"]),
                      "errors": {s: [e["arm"] for e in x.get("errors", [])] for s, x in rec["shapes"].items()}}))
    sys.exit(code)


if __name__ == "__main__":
    main()
