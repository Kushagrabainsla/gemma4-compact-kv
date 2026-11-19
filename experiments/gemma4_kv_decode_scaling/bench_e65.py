"""E-65: the decode-step attention time of the Gemma 4 31B stock and compact K=V caches over a grid of batch sizes and
contexts (README Sections 4 and 5). A weight-free microbenchmark on one GPU, built on the E-58 code (`bench_e58.py`:
the cache layouts, the kernel arguments, the timer).

Per grid point (B requests of L tokens, one decode token each):
- G: the read of 5 global layers (32 query heads, 4 KV heads, head 512), each with its own cache, in the two engine
  layouts S32 (stock row 1,024, block 32) and C48 (compact row 640 read through the two views, block 48, page padded
  to 262,144 B). One CUDA graph per layout; 3 interleaved rounds of 50 replays (GPU time, CUDA events).
- SL: the read of 5 sliding layers (32 query heads, 16 KV heads, head 256, block 16, the stock pool page), each with
  its own cache of min(L, 1,024) tokens. The same in both layouts. One graph; one timing of 50 replays.
- H: the hook transform of the compact build (`compact_engine.ViewsSplice`, mode VIEWS) on 5 modules that do no
  attention, at batch B. One graph; one timing of 50 replays.
- V1: the S32 and C48 outputs of global layer 0 (raw calls) are equal.

The 3D decode path runs for B <= 8 (seq_threshold_3D = 8, as in the engine with CUDA graphs: the capture size nearest
to 8); the kernel takes the 2D path for larger B. --stub runs the flow on a Mac with the E-58 reference attention and
small shapes (no CUDA, no vLLM, perf_counter times).

Usage: bench_e65.py --out <dir> [--rehearsal] [--stub] [--points B:L,B:L,...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "gemma4_kv_microbench"), str(HERE.parent / "gemma4_kv_engine"),
                str(HERE.parent / "gemma4_kv_model")]
import bench_e58 as b58  # noqa: E402
import compact_engine as ce  # noqa: E402

LAYERS = 5                                   # per arm; the analysis scales G by 2 (10 global) and SL by 10 (50 sliding)
ROUNDS, ITERS, WARM = 3, 50, 3
B_GRID = (1, 2, 4, 8, 16, 32, 64)
L_GRID = (1_024, 4_096, 16_384, 32_768, 65_536, 131_072)     # A1 (R-101 B3): the 32K and 128K rows
MAX_TOKENS = 2 ** 19                         # B x L cap of the core grid
EXTRA = ((16, 65_536), (8, 131_072))         # A1 (R-101 B3): the 2D path at 64K; 128K at B = 8 (not core: V5)
SEQ_TOKENS = 2 ** 19                         # A1: above this, the two layouts are built and timed one after the other
TIE = (3, 32_768)                            # the E-58 D32 shape and the E-59 decode point; runs first (A1, B4)
REH_POINTS = ((3, 32_768), (1, 1_024), (64, 1_024), (16, 4_096), (32, 4_096), (8, 16_384), (4, 32_768),
              (2, 65_536))                   # A1 (R-101 F9): 8 rehearsal points; 3D and 2D paths
REH_SEQ_TOKENS = 2 ** 16                     # the rehearsal tests the one-after-the-other path at (2, 65,536)
STUB_B, STUB_L, STUB_MAX, STUB_SEQ = (1, 2, 4), (48, 96, 192), 384, 300
SL_HK, SL_D, SL_BLOCK, SL_WINDOW = 16, 256, 16, 1_024
SL_PAGE = 262_144 // 2                       # BF16 elements of the pool page: 16 tokens x 16 heads x 512 values


def grid(max_tokens: int = MAX_TOKENS, bs=B_GRID, ls=L_GRID, tie=TIE, extra=EXTRA) -> tuple[list, list]:
    """(plan, core): the tie point first, then the core points (B x L <= max_tokens) by tokens, then the extra
    points. A1 (R-101 B4): the cheap tie check runs before the money goes to the large points."""
    core = [(b, ln) for ln in ls for b in bs if b * ln <= max_tokens]
    if tie is not None and tie not in core:
        core.append(tie)
    rest = sorted((p for p in core if p != tie), key=lambda p: (p[0] * p[1], p[1]))
    plan = ([tie] if tie is not None else []) + rest + [p for p in extra if p not in core]
    return plan, core


class SlidingCache:
    """One sliding layer's paged cache: [blocks, 16, 16, 512] with the engine strides; K = [..., :256], V = [..., 256:].
    A1 (R-101 B2): each request has the full context L in its block table, as in the engine; the entries that lie
    before the window point at one shared block (the kernel does not read them), and the window tail has its own
    blocks."""

    def __init__(self, batch: int, tokens: int, device):
        self.per_req = math.ceil((tokens + 1) / SL_BLOCK)
        self.first_real = max(0, (tokens - SL_WINDOW) // SL_BLOCK - 1)
        self.real = self.per_req - self.first_real
        self.num_blocks = 1 + batch * self.real
        raw = torch.zeros(self.num_blocks * SL_PAGE, dtype=torch.bfloat16, device=device)
        self.cache = torch.as_strided(raw, (self.num_blocks, SL_HK, SL_BLOCK, 2 * SL_D),
                                      (SL_PAGE, 2 * SL_D, SL_HK * 2 * SL_D, 1))
        bt = torch.zeros(batch, self.per_req, dtype=torch.int32)
        for r in range(batch):
            bt[r, self.first_real:] = torch.arange(1 + r * self.real, 1 + (r + 1) * self.real, dtype=torch.int32)
        self.block_table = bt.to(device)

    def views(self):
        return self.cache.transpose(1, 2).split(SL_D, dim=-1)


def sliding_kwargs(q, kc, vc, out, sc: SlidingCache, tokens: int, cu, seqused, scratch, ones) -> dict:
    """As `TritonAttentionImpl.forward` for a sliding layer: window (sliding_window - 1, 0) over the full context."""
    return dict(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seqused, max_seqlen_k=tokens,
                softmax_scale=1.0, causal=True, window_size=(SL_WINDOW - 1, 0), block_table=sc.block_table,
                softcap=0.0, q_descale=None, k_descale=ones, v_descale=ones, seq_threshold_3D=b58.SEQ_THRESHOLD_3D,
                num_par_softmax_segments=b58.SEGMENTS, softmax_segm_output=scratch[0], softmax_segm_max=scratch[1],
                softmax_segm_expsum=scratch[2])


class Clock:
    """GPU time of a step: a CUDA graph captured once, then replays between two CUDA events. The stub uses
    perf_counter on the eager step."""

    def __init__(self, stub: bool):
        self.stub = stub
        self.iters = 3 if stub else ITERS

    def capture(self, step):
        if self.stub:
            return step
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARM):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        torch.cuda.synchronize()
        return g.replay

    def ms(self, replay) -> float:
        for _ in range(WARM):
            replay()
        if self.stub:
            t0 = time.perf_counter()
            for _ in range(self.iters):
                replay()
            return 1_000 * (time.perf_counter() - t0) / self.iters
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(self.iters):
            replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / self.iters


def run_point(batch: int, length: int, ns, raw, clock: Clock, dev, stub: bool, layers: int = LAYERS,
              seq_tokens: int = SEQ_TOKENS) -> dict:
    """All arms at one point. Raises on an error; the caller records it."""
    t0 = time.time()
    torch.manual_seed(65)
    pt: dict = {"batch": batch, "length": length, "layers": layers, "tokens": batch * length}
    cu = torch.arange(batch + 1, dtype=torch.int32, device=dev)
    ones_g = torch.ones((), dtype=torch.float32, device=dev).expand(batch, b58.HK)
    scratch = (torch.empty(b58.SCRATCH_TOKENS, b58.HQ, b58.SEGMENTS, b58.D, dtype=torch.float32, device=dev),
               torch.empty(b58.SCRATCH_TOKENS, b58.HQ, b58.SEGMENTS, dtype=torch.float32, device=dev),
               torch.empty(b58.SCRATCH_TOKENS, b58.HQ, b58.SEGMENTS, dtype=torch.float32, device=dev))
    q = (torch.randn(batch, b58.HQ, b58.D, device=dev) * 0.1).to(torch.bfloat16)

    # G: the global reads, S32 and C48, from the same logical rows (bench_e58.Layout.fill). A1: above seq_tokens the
    # two layouts are built and timed one after the other (one resident at a time); else interleaved rounds.
    seqused = torch.full((batch,), length, dtype=torch.int32, device=dev)
    sequential = batch * length > seq_tokens
    pt["sequential"] = sequential

    def build(lo):
        lays = [b58.Layout(lo, batch, length, dev) for _ in range(layers)]
        for i in range(layers):
            gen = torch.Generator(device=dev).manual_seed(65_000 + 100 * i + int(math.log2(length)))
            x = (torch.randn(batch, length, b58.HK, b58.ROW_C, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            lays[i].fill(x)
            del x
        kws = []
        for i in range(layers):
            kc, vc = lays[i].views()
            kws.append(b58.read_kwargs(q, kc, vc, torch.empty(batch, b58.HQ, b58.D, dtype=torch.bfloat16, device=dev),
                                       lays[i], length, cu, seqused, scratch, ones_g))
        raw.unified_attention(**kws[0])              # V1: the layer-0 output (the timed steps write the same buffer)
        if not stub:
            torch.cuda.synchronize()
        out0 = kws[0]["out"].clone()

        def step():
            for kw in kws:
                raw.unified_attention(**kw)
        return lays, kws, out0, clock.capture(step)

    pt["G"] = {"S32": [], "C48": []}
    outs, info = {}, {}
    if sequential:
        for lo in ("S32", "C48"):
            lays, kws, outs[lo], rep = build(lo)
            info[lo] = (lays[0].page * 2, lays[0].num_blocks, lays[0].row)
            for _ in range(ROUNDS):
                pt["G"][lo].append(clock.ms(rep))
            del lays, kws, rep
            if not stub:
                torch.cuda.empty_cache()
    else:
        built = {lo: build(lo) for lo in ("S32", "C48")}
        for lo, (lays, _, out0, _) in built.items():
            outs[lo], info[lo] = out0, (lays[0].page * 2, lays[0].num_blocks, lays[0].row)
        order = ["S32", "C48"]
        for r in range(ROUNDS):
            for lo in (order if r % 2 == 0 else order[::-1]):
                pt["G"][lo].append(clock.ms(built[lo][3]))
        del built
    o_s, o_c = outs["S32"], outs["C48"]
    pt["v1"] = {"identical": bool(torch.equal(o_s, o_c)), "max_abs": float((o_s.float() - o_c.float()).abs().max()),
                "out_scale": float(o_s.float().abs().max())}
    pt["pages"] = {lo: v[0] for lo, v in info.items()}
    pt["blocks_per_layer"] = {lo: v[1] for lo, v in info.items()}
    pt["bytes_distinct"] = {lo: batch * length * b58.HK * v[2] * 2 * layers for lo, v in info.items()}
    del outs
    if not stub:
        torch.cuda.empty_cache()

    # SL: the sliding reads with the engine window over the full context (A1, R-101 B2); the same in both layouts.
    seq_sl = torch.full((batch,), length, dtype=torch.int32, device=dev)
    ones_s = torch.ones((), dtype=torch.float32, device=dev).expand(batch, SL_HK)
    scr_s = (torch.empty(b58.SCRATCH_TOKENS, b58.HQ, b58.SEGMENTS, SL_D, dtype=torch.float32, device=dev),
             torch.empty(b58.SCRATCH_TOKENS, b58.HQ, b58.SEGMENTS, dtype=torch.float32, device=dev),
             torch.empty(b58.SCRATCH_TOKENS, b58.HQ, b58.SEGMENTS, dtype=torch.float32, device=dev))
    qs = (torch.randn(batch, b58.HQ, SL_D, device=dev) * 0.1).to(torch.bfloat16)
    sls = []
    for i in range(layers):
        sc = SlidingCache(batch, length, dev)
        gen = torch.Generator(device=dev).manual_seed(66_000 + i)
        sc.cache.copy_((torch.randn(sc.cache.shape, generator=gen, device=dev) * 0.5).to(torch.bfloat16))
        kc, vc = sc.views()
        sls.append(sliding_kwargs(qs, kc, vc, torch.empty(batch, b58.HQ, SL_D, dtype=torch.bfloat16, device=dev), sc,
                                  length, cu, seq_sl, scr_s, ones_s))

    def sl_step():
        for kw in sls:
            raw.unified_attention(**kw)
    rep = clock.capture(sl_step)
    pt["SL"] = [clock.ms(rep) for _ in range(ROUNDS)]          # A1 (R-101 F3): 3 rounds
    pt["SL_blocks_per_layer"] = sls[0]["block_table"].shape[1]
    del sls, rep

    # H: the hook transform of the compact build at batch B (no attention).
    xq = q.reshape(batch, b58.HQ * b58.D)
    xk = (torch.randn(batch, b58.HK * b58.D, device=dev) * 0.5).to(torch.bfloat16)
    xv = (torch.randn(batch, b58.HK * b58.D, device=dev) * 0.5).to(torch.bfloat16)
    out_buf = torch.zeros(batch, b58.HQ * b58.D, dtype=torch.bfloat16, device=dev)
    w = torch.full((b58.D,), b58.W_VALUE, device=dev)
    mods = [b58.GlobalLayer(ns, out_buf=out_buf) for _ in range(layers)]
    splice = ce.ViewsSplice()
    for i, m in enumerate(mods):
        splice.attach(i, m, w, (b58.HQ, b58.HK, b58.D))
    splice.set_mode("VIEWS")

    def h_step():
        for m in mods:
            m(xq, xk, xv)
    rep = clock.capture(h_step)
    pt["H"] = [clock.ms(rep) for _ in range(ROUNDS)]           # A1 (R-101 F3): 3 rounds
    splice.detach()
    if not stub:
        torch.cuda.empty_cache()
    pt["seconds"] = round(time.time() - t0, 2)
    return pt


def tie_ok(pt: dict) -> bool:
    """V4 (README Section 6): at (3, 32K), 2 G_S and 2 G_C within 5% of the E-58 reads (5.076 and 4.582 ms).
    E-68 A3 as changed by R-109 (only if the environment sets E68_TIE_RULE=ratio): the read ratio C48 / S32 within 3%
    of the E-58 ratio, and each read within 10% of its E-58 value; the default (E-65) is not changed."""
    import os
    import statistics
    import analyze_e65 as an
    gs, gc = 2 * statistics.median(pt["G"]["S32"]), 2 * statistics.median(pt["G"]["C48"])
    if os.environ.get("E68_TIE_RULE") == "ratio":
        ref = an.E58_READ_MS["C48"] / an.E58_READ_MS["S32"]
        return (abs((gc / gs) / ref - 1) <= 0.03 and abs(gs / an.E58_READ_MS["S32"] - 1) <= 0.10
                and abs(gc / an.E58_READ_MS["C48"] - 1) <= 0.10)
    return (abs(gs / an.E58_READ_MS["S32"] - 1) <= an.TIE_TOL and abs(gc / an.E58_READ_MS["C48"] - 1) <= an.TIE_TOL)


def parse_points(text: str) -> list[tuple[int, int]]:
    return [(int(b), int(ln)) for b, ln in (p.split(":") for p in text.split(",") if p.strip())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", default="")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-plant", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"meta": {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stub": args.stub,
                          "rehearsal": args.rehearsal, "layers": LAYERS, "rounds": ROUNDS, "iters": ITERS,
                          "seq_threshold_3D": b58.SEQ_THRESHOLD_3D, "segments": b58.SEGMENTS,
                          "tie_rule": os.environ.get("E68_TIE_RULE", "absolute")},   # E-68 A3 (R-109 item 4)
                 "points": [], "errors": [], "status": "running"}
    path = args.out / "bench.json"

    def save():
        path.write_text(json.dumps(rec, indent=1) + "\n")
    code = 1
    try:
        if args.stub:
            torch.set_num_threads(1)
            dev = torch.device("cpu")
            raw = SimpleNamespace(unified_attention=b58.ref_read, triton_reshape_and_cache_flash=b58.ref_write)
            ns = SimpleNamespace(unified_attention=b58.ref_read, triton_reshape_and_cache_flash=b58.ref_write)
            b58.stub_install(ns)
            pts, core = grid(STUB_MAX, STUB_B, STUB_L, None, ((4, 192),))
            seq_tokens = STUB_SEQ
            rec["meta"].update(gpu="STUB", vllm="STUB")
        else:
            import vllm
            import vllm.v1.attention.backends.triton_attn as ta
            dev = torch.device("cuda")
            raw = SimpleNamespace(unified_attention=ta.unified_attention,
                                  triton_reshape_and_cache_flash=ta.triton_reshape_and_cache_flash)
            ce.install("compact")                     # as E-58: the wrappers on the backend's two kernel entry points
            ns = ta
            if args.rehearsal:
                pts, core, seq_tokens = list(REH_POINTS), list(REH_POINTS), REH_SEQ_TOKENS
            else:
                (pts, core), seq_tokens = grid(), SEQ_TOKENS
            rec["meta"].update(gpu=torch.cuda.get_device_name(0), vllm=vllm.__version__, torch=torch.__version__,
                               installed=ce.STATE["installed"])
        if args.points:
            pts = core = parse_points(args.points)
        rec["meta"]["plan"] = [list(p) for p in pts]
        rec["meta"]["core"] = [list(p) for p in core]
        rec["meta"]["seq_tokens"] = seq_tokens
        save()
        clock = Clock(args.stub)
        main_run = not (args.stub or args.rehearsal)
        tries = 0
        i = 0
        while i < len(pts):
            b, ln = pts[i]
            try:
                pt = run_point(b, ln, ns, raw, clock, dev, args.stub, seq_tokens=seq_tokens)
                if args.stub and args.stub_plant == "compact_fast":    # a test plant: C48 reads at the byte ratio
                    pt["G"]["C48"] = [0.625 * x for x in pt["G"]["S32"]]
                if (b, ln) == TIE and main_run:       # A1 (R-101 B4, F5): V4 at once; one rerun; else stop here
                    ok = tie_ok(pt)
                    rec.setdefault("tie_tries", []).append({"G": pt["G"], "ok": ok})
                    if not ok:
                        tries += 1
                        if tries < 2:
                            save()
                            continue
                        rec["points"].append(pt)
                        rec["tie_status"] = "failed twice: the bench stops (V4)"
                        save()
                        break
                    rec["tie_status"] = f"ok after {tries + 1} tries"
                rec["points"].append(pt)
            except Exception as exc:  # recorded; the analysis marks the point missing
                rec["errors"].append({"point": [b, ln], "error": f"{type(exc).__name__}: {exc}"[:400],
                                      "trace": traceback.format_exc()[-1_500:]})
                if not args.stub:
                    torch.cuda.empty_cache()
            save()
            i += 1
        rec["status"] = "ok"
        code = 0
    except BaseException as exc:  # recorded
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
        rec["traceback"] = traceback.format_exc()[-3_000:]
    rec["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["wrapper_counts"] = {k: ce.STATE[k] for k in ("fix_reads", "fix_writes", "pass_reads", "pass_writes")}
    save()
    print(json.dumps({"status": rec["status"], "points": len(rec["points"]),
                      "errors": [e["point"] for e in rec["errors"]]}))
    sys.exit(code)


if __name__ == "__main__":
    main()
