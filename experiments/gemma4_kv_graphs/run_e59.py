"""E-59: one engine process (README Sections 4 and 5).

The E-57 build (`compact_engine.install`: the B3 spec in a compact engine, the V-view wrappers in every engine) with
the VIEWS hooks attached right after the model loads (a wrapper of `GPUModelRunner.load_model`), so that a CUDA graph
capture records them. A graph engine uses full decode graphs with no torch.compile (FULL_DECODE_ONLY, capture sizes
1, 2, 3, 4, 8); an eager engine uses `enforce_eager` (E-57). The process records the engine, the KV plan, the hooks,
the compilation settings and the Triton 3D scratch (V3), runs a warm-up, then the rows: `batch` requests of 32K
tokens (the E-57 C2 prompts), a window of 128 full decode steps after 3 skipped steps, and in profiled rows 6 more
steps under the torch profiler. It keeps the first output tokens of each request (V1) and counts CUDA graph replays
(V2).

--stub runs the same flow on a Mac: the E-57 stub scheduler (`run_e57.StubEngine`), a step-time model with planted
effects, the real hooks on stub modules (L-26), and synthetic tokens (no vLLM, no GPU).

Usage: run_e59.py --engine G1 --layout stock --graphs 1 --capture OFF --rows STOCK:3,STOCK:3 --profile-rows 1
                  --out <res>/G1 --snapshot <dir> --model-id <id> --deadline <epoch> [--rehearsal]
                  [--stub [--stub-plant <plant>]]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E57, E55 = HERE.parent / "gemma4_kv_engine", HERE.parent / "gemma4_kv_model"
sys.path[:0] = [str(HERE), str(E57)]
sys.path += [str(E55)]
import compact_engine as ce  # noqa: E402
import compact_layout as cl  # noqa: E402
import run_e57 as r57  # noqa: E402

LENGTH, GAP, MAX_TOKENS = 32_768, r57.CAP_GAP, r57.CAP_MAX_TOKENS      # 32K rows of L - 256 prompt tokens, 512 out
SKIP, WINDOW, MIN_WINDOW, PROFILE_STEPS = 3, 128, 100, 6
TOKENS_KEEP, MIN_COMMON, SLOW = 192, 96, 1.05      # A1 (R-86 B2): 96 common tokens (the rows hold about 134)
CAPTURE_SIZES, MAX_BATCH, GLOBAL_HEAD = (1, 2, 3, 4, 8), 4, 512
WARM_PROMPT, WARM_TOKENS, EST_ROW = 1_024, 32, 120.0
ENGINE = dict(r57.ENGINE)                # E-57: eager, 15 GiB pool, 66,048 tokens, chunks of 2,048, 16 sequences
REH = {"length": 16_384, "kv_bytes": 12 * 2**30, "max_model_len": 16_896, "window": 32, "min_window": 24,
       "min_common": 24, "est_row": 60.0}
ROW_MODE = {"STOCK": "OFF", "EMULV": "VIEWS", "COMPACT": "VIEWS"}
ROW_LAYOUT = {"STOCK": "stock", "EMULV": "stock", "COMPACT": "compact"}
STUB_HEADS, STUB_LAYERS, STUB_W = r57.STUB_HEADS, r57.STUB_LAYERS, r57.STUB_W
CATEGORIES = (("attention", r"unified_attention|reduce_segments"), ("kv_write", r"reshape_and_cache"),
              ("gemm", r"gemm|Gemm|cutlass|xmma|sm80_|ampere_|cublas|matmul"), ("norm", r"norm|rms"),
              ("index_and_elementwise", r"index|gather|scatter|elementwise|vectorized|unrolled|copy|cat|mul|add"))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_rows(text: str) -> list[tuple[str, int]]:
    """'STOCK:3,EMULV:3' -> [('STOCK', 3), ('EMULV', 3)]."""
    out = []
    for part in text.split(","):
        arm, _, b = part.strip().partition(":")
        out.append((arm.strip().upper(), int(b or 3)))
    return out


def plan_problems(layout: str, graphs: bool, capture: str, rows: list[tuple[str, int]]) -> list[str]:
    """The engine plan (README Section 4.5): the arms fit the layout, a graph engine's rows use its capture mode, the
    compact layout needs VIEWS (its cache is wrong without the hooks), and no batch exceeds MAX_BATCH (V3)."""
    p = []
    if capture not in ("OFF", "VIEWS"):
        p.append(f"capture mode {capture}")
    if layout == "compact" and capture != "VIEWS":
        p.append("the compact layout needs the capture mode VIEWS")
    for arm, b in rows:
        if arm not in ROW_MODE:
            p.append(f"unknown arm {arm}")
            continue
        if ROW_LAYOUT[arm] != layout:
            p.append(f"arm {arm} needs the {ROW_LAYOUT[arm]} layout")
        if graphs and ROW_MODE[arm] != capture:
            p.append(f"graph row {arm} needs mode {ROW_MODE[arm]}, the capture mode is {capture}")
        if not 1 <= b <= MAX_BATCH:
            p.append(f"batch {b} outside 1 to {MAX_BATCH}")
    return p


def scratch_problems(builders: list[dict], max_batch: int = MAX_BATCH) -> list[str]:
    """V3 (A1, R-86 B1): every Triton builder's 3D scratch is GLOBAL_HEAD wide in its last dimension (so each of its
    rows holds one decode token of a global layer, also in the capture warm-up of the largest capture size) and has
    `max_batch` rows or more. The pinned build sizes the scratch with the model head size (256) and the kernel indexes
    it with the layer head size (512 at the global layers); `widen_scratch` makes it 512 wide."""
    if not builders:
        return ["no Triton metadata builder found"]
    p = []
    for b in builders:
        rows, shape = b.get("rows"), b.get("shape")
        if rows is None or not shape:
            p.append(f"builder without a scratch record: {b}")
            continue
        if shape[-1] < GLOBAL_HEAD:
            p.append(f"scratch width {shape[-1]} < {GLOBAL_HEAD} (rows {rows}, threshold {b.get('seq_threshold_3D')})")
        if rows < max_batch:
            p.append(f"scratch rows {rows} < batch {max_batch}")
    return p


WIDENED = {"count": 0, "init_calls": 0, "after_kv_init": 0}
RUNNER: dict = {"obj": None}


def widen_scratch() -> None:
    """A1 (R-86 B1): after the stock `TritonAttentionMetadataBuilder.__init__`, replace a 3D scratch narrower than
    GLOBAL_HEAD with one of the same rows, heads and segments that is GLOBAL_HEAD wide. The kernels compute the scratch
    offsets from their own HEAD_SIZE_PADDED (256 at the sliding layers, 512 at the global layers), in the store and in
    the reduction alike, so the arithmetic does not change; only the writes past the old buffer stop."""
    import vllm.v1.attention.backends.triton_attn as ta
    cls = ta.TritonAttentionMetadataBuilder
    if getattr(cls, "_e59_widened", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        WIDENED["init_calls"] += 1
        s = getattr(self, "softmax_segm_output", None)
        print(f"E59 builder init {type(self).__module__}.{type(self).__qualname__} scratch "
              f"{None if s is None else list(s.shape)}", flush=True)        # A2: the diagnosis of rehearsal 1
        if s is not None and s.shape[-1] < GLOBAL_HEAD:
            self.softmax_segm_output = torch.empty((*s.shape[:-1], GLOBAL_HEAD), dtype=s.dtype, device=s.device)
            WIDENED["count"] += 1

    cls.__init__ = __init__
    cls._e59_widened = True


def builders_of(runner) -> list:
    """A2 (rehearsal 1): the attention metadata builders that the model runner holds (`attn_groups`, a list of lists of
    AttentionGroup, each with `metadata_builders`). A gc scan does not find them: vLLM freezes the gc heap after the
    start (`freeze_gc_heap` in v1/engine/core.py and gpu_worker.py), and gc.get_objects() skips frozen objects."""
    out = []
    for groups in getattr(runner, "attn_groups", None) or []:
        for g in groups if isinstance(groups, (list, tuple)) else [groups]:
            out.extend(getattr(g, "metadata_builders", None) or [])
    return out


def widen_builders(runner, where: str) -> int:
    """A2: widen the 3D scratch of every builder of the runner to GLOBAL_HEAD (the same rows, heads and segments), as
    `widen_scratch` does at the builder's __init__. Called right after `initialize_kv_cache`, before the warm-up runs
    and the graph capture, so the captured graphs use the wide buffer whatever the builder class is."""
    n = 0
    for b in builders_of(runner):
        s = getattr(b, "softmax_segm_output", None)
        if s is not None and s.shape[-1] < GLOBAL_HEAD:
            b.softmax_segm_output = torch.empty((*s.shape[:-1], GLOBAL_HEAD), dtype=s.dtype, device=s.device)
            n += 1
    WIDENED[where] = WIDENED.get(where, 0) + n
    return n


def scratch_record(runner=None) -> list[dict]:
    """The record of V3: each builder of the runner with its class, its 3D threshold and its scratch shape."""
    out = []
    for o in builders_of(runner if runner is not None else RUNNER["obj"]):
        s = getattr(o, "softmax_segm_output", None)
        out.append({"class": f"{type(o).__module__}.{type(o).__qualname__}",
                    "seq_threshold_3D": int(getattr(o, "seq_threshold_3D", -1)),
                    "headdim": int(getattr(o, "headdim", -1)), "block_size": int(getattr(o, "block_size", -1)),
                    "rows": int(s.shape[0]) if s is not None else None,
                    "shape": list(s.shape) if s is not None else None,
                    "decode_cudagraph_enabled": bool(getattr(o, "decode_cudagraph_enabled", False))})
    return out


def window_stats(steps: list[dict], batch: int, skip: int = SKIP, window: int = WINDOW) -> dict:
    """The decode window (as `run_e57.window_from_steps`): the first run of steps in which all `batch` requests run
    and decode, before the profiled steps, without its first `skip` steps, at most `window` steps. Steps above
    CAP_TRIM times the median are dropped for the mean (and counted); the median uses every window step."""
    run: list[int] = []
    for i, s in enumerate(steps):
        if s.get("probe"):
            break
        if s["running"] == batch and s["decoding"] == batch:
            run.append(i)
        elif run:
            break
    win = run[skip:skip + window]
    out = {"full_steps": len(run), "window_steps": len(win), "dropped": 0}
    if not win:
        return out
    secs = np.array([steps[i]["seconds"] for i in win], dtype=np.float64)
    med = float(np.median(secs))
    keep = secs[secs <= r57.CAP_TRIM * med]
    cpu = [steps[i].get("cpu") for i in win if steps[i].get("cpu") is not None]
    out.update(median_ms=round(1_000 * med, 4), mean_ms=round(1_000 * float(keep.mean()), 4),
               dropped=int(secs.size - keep.size), slow_share=round(float((secs > SLOW * med).mean()), 4),
               tokens_per_s=round(batch / med, 4), cpu_ms=round(1_000 * float(np.median(cpu)), 4) if cpu else None,
               replays=int(sum(steps[i].get("replays", 0) for i in win)),
               held=float(np.median([steps[i]["held"] for i in win])))
    return out


def merged_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    start = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        total += end - start
    return total


def summarize_trace(trace: dict, steps: int) -> dict:
    """Per-step numbers of a profiled stretch (report only): the GPU kernel time, the busy time (the union of kernel
    and copy intervals), the gaps, the launch calls and graph launches, the kernel time by category, the top kernels."""
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "ts" in e]
    kern = [e for e in gpu if e.get("cat") == "kernel"]
    if not gpu or steps <= 0:
        return {"steps": steps, "kernels": 0}
    iv = [(float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in gpu]
    busy, span = merged_length(iv), max(b for _, b in iv) - min(a for a, _ in iv)
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    by_name: dict[str, list[float]] = {}
    for e in kern:
        by_name.setdefault(e.get("name", "?"), []).append(float(e.get("dur", 0)))
    cats = {c: 0.0 for c, _ in CATEGORIES}
    cats["other"] = 0.0
    for name, durs in by_name.items():
        c = next((c for c, rx in CATEGORIES if re.search(rx, name)), "other")
        cats[c] += sum(durs)
    top = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[:25]
    per = lambda us: round(us / steps / 1_000, 4)  # noqa: E731  (microseconds over the stretch -> ms per step)
    return {"steps": steps, "kernels": len(kern), "kernels_per_step": round(len(kern) / steps, 1),
            "kernel_ms": per(sum(sum(d) for d in by_name.values())), "busy_ms": per(busy), "span_ms": per(span),
            "gap_ms": per(span - busy),
            "launches_per_step": round(sum(1 for e in rt if "LaunchKernel" in e.get("name", "")) / steps, 1),
            "graph_launches_per_step": round(sum(1 for e in rt if "GraphLaunch" in e.get("name", "")) / steps, 2),
            "category_ms": {c: per(v) for c, v in cats.items()},
            "top": [{"name": n[:160], "calls_per_step": round(len(d) / steps, 2), "ms": per(sum(d))} for n, d in top]}


class Replays:
    """Counts `torch.cuda.CUDAGraph.replay` calls (V2: graph rows replay graphs, eager rows do not)."""

    def __init__(self):
        self.count = 0
        self.installed = False

    def install(self) -> None:
        if self.installed:
            return
        orig = torch.cuda.CUDAGraph.replay
        counter = self

        def replay(graph, *a, **k):
            counter.count += 1
            return orig(graph, *a, **k)

        torch.cuda.CUDAGraph.replay = replay
        self.installed = True


REPLAYS = Replays()


class Profiler:
    """The torch profiler over the steps after the window (report only). Saves a summary and the trace (gzip) when the
    trace is small."""

    def __init__(self, path: Path):
        self.path, self.p = path, None

    def start(self) -> None:
        from torch.profiler import ProfilerActivity, profile
        self.p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        self.p.__enter__()

    def stop(self, steps: int) -> dict:
        try:
            self.p.__exit__(None, None, None)
            raw = self.path.with_suffix(".json")
            self.p.export_chrome_trace(str(raw))
            trace = json.loads(raw.read_text())
            out = summarize_trace(trace, steps)
            if raw.stat().st_size <= 40 * 2**20:
                with gzip.open(self.path.with_suffix(".json.gz"), "wt") as f:
                    json.dump(trace, f)
            raw.unlink()
            return out
        except Exception as exc:  # noqa: BLE001  (report only)
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}


class StubProfiler:
    def __init__(self, path: Path):
        self.path = path

    def start(self) -> None:
        pass

    def stop(self, steps: int) -> dict:
        fake = {"traceEvents": [{"cat": "kernel", "name": n, "ts": 10.0 * i, "dur": 5.0}
                                for i, n in enumerate(["kernel_unified_attention_3d", "ampere_bf16_gemm",
                                                       "index_select_kernel"] * steps)]}
        return summarize_trace(fake, steps)


class E59Engine(r57.VllmEngine):
    """The in-process engine (E-57 `VllmEngine`) with the step CPU time, the first output tokens and the replays."""

    def __init__(self, llm):
        super().__init__(llm)
        self.tokens: dict[str, list[int]] = {}

    def step(self) -> tuple[float, list[str], float]:
        c0, t1 = time.process_time(), time.perf_counter()   # A1 (R-86 F2): the CPU time of every thread
        outs = self.eng.step()
        dt, cpu = time.perf_counter() - t1, time.process_time() - c0
        done = []
        for o in outs:
            rid = o.request_id
            if len(self.tokens.get(rid, ())) < TOKENS_KEEP and o.outputs:
                self.tokens[rid] = list(o.outputs[0].token_ids)[:TOKENS_KEEP]
            if getattr(o, "finished", False):
                done.append(rid)
        return dt, done, cpu

    def replays(self) -> int:
        return REPLAYS.count


class StubWorld59(r57.StubWorld):
    """The step-time model of the stub (seconds): a base that graphs make smaller, the KV read of the requests (the
    compact row reads 640 of 1,024 values), the hook cost (eager 2.55 ms, graph 0.40 ms: E-57, E-58), the wrapper cost
    of the compact engine in eager mode, a process offset, small noise and 20% slow steps (R-83). Plants: kill, pivot
    (a graph cost of the compact engine), noisy (a large process offset), graph_drops_hooks (wrong compact tokens with
    graphs), scratch_small (V3), no_replay (V2), and the E-57 plants."""

    def __init__(self, layout: str, plant: str, rehearsal: bool, graphs: bool, process: str):
        super().__init__(layout, plant, rehearsal)
        self.graphs, self.mode = graphs, "OFF"
        rng = np.random.default_rng(r57.seed_of("process", process, plant))
        self.offset = float(rng.uniform(-1, 1)) * (0.04 if plant == "noisy" else 0.003)
        self.rng = np.random.default_rng(r57.seed_of("steps", process, plant))

    def step_seconds(self, batch: int, length: int) -> float:
        kv = (cl.ROW if self.compact else 1_024) / 1_024
        t = (0.043 if self.graphs else 0.048) + 0.0006 * batch * (length / 32_768) * kv
        if self.mode == "VIEWS":
            t += 0.0004 if self.graphs else 0.00255
        if self.compact and not self.graphs:
            t += 0.0005
        if self.compact and self.graphs:
            t += {"kill": 0.004, "pivot": 0.0015}.get(self.plant, 0.0)
        t = t * (1 + self.offset) + abs(float(self.rng.normal(0, 0.0002)))
        return t + (0.005 if self.rng.random() < 0.2 else 0.0)


class StubEngine59(r57.StubEngine):
    """The E-57 stub scheduler with the E-59 interface: the CPU time, synthetic tokens and graph replays."""

    def __init__(self, world: StubWorld59, num_blocks: int, length: int):
        super().__init__(world, num_blocks)
        self.length, self.tokens, self.count = length, {}, 0

    def _token(self, rid: str, pos: int) -> int:
        idx = rid.rsplit("-", 1)[1]
        w = self.world
        arith = "stock" if w.mode == "OFF" else "views"
        if w.compact and (w.mode != "VIEWS" or (w.graphs and w.plant == "graph_drops_hooks")):
            arith = "broken"
        return r57.seed_of("tok", self.length, idx, pos, arith if pos >= 40 else "same") % 50_000

    def step(self) -> tuple[float, list[str], float]:
        before = {k: r["out"] for k, r in self.reqs.items()}
        dt, done = super().step()
        run = [r for r in self.reqs.values() if r["state"] == "running"]
        for k, r in self.reqs.items():
            if r["out"] > before.get(k, 0) and len(self.tokens.setdefault(k, [])) < TOKENS_KEEP:
                self.tokens[k].append(self._token(k, r["out"] - 1))
        decode_only = bool(run) and all(r["hist"] and r["hist"][-1] == 1 for r in run)
        w = self.world
        if w.graphs and decode_only and len(run) in CAPTURE_SIZES and w.plant != "no_replay":
            self.count += 1
        return dt, done, dt * (0.25 if w.graphs else 0.92)

    def replays(self) -> int:
        return self.count


def run_row(eng, tag: str, batch: int, length: int, window: int, check, profiler=None, max_tokens: int = MAX_TOKENS,
            profile_steps: int = PROFILE_STEPS) -> dict:
    """One row (README Section 4.4): `batch` requests of L - 256 prompt tokens; the steps until the window of
    `window` full steps after SKIP; with a profiler, `profile_steps` more steps under it. Then abort and drain."""
    ids = [f"{tag}-{i}" for i in range(batch)]
    st0 = eng.state()
    free0 = st0["free"]
    for i, rid in enumerate(ids):
        eng.add(rid, r57.cap_prompt(length, i + 1, GAP), max_tokens)
    steps: list[dict] = []
    full, end, prof_n, profiling = 0, "step cap", 0, False
    limit = batch * (math.ceil(length / 1_024) + 8) + SKIP + window + profile_steps + 64
    while eng.busy() and len(steps) < limit:
        r0 = eng.replays()
        dt, done, cpu = eng.step()
        st = eng.state()
        st.update(seconds=dt, cpu=cpu, finished=len(done), held=free0 - st["free"], probe=profiling,
                  replays=eng.replays() - r0)
        steps.append(st)
        check()
        if st["preemptions"] > st0["preemptions"]:
            end = "preempted"
            break
        if done:
            end = "finished before the window"
            break
        if profiling:
            prof_n += 1
            if prof_n >= profile_steps:
                end = "window"
                break
            continue
        if st["running"] == batch and st["decoding"] == batch:
            full += 1
        if full >= SKIP + window:
            if profiler is None:
                end = "window"
                break
            profiler.start()
            profiling = True
    profile = profiler.stop(prof_n) if profiling else None
    eng.abort(ids)
    drain = 0
    while eng.busy() and drain < 64:
        eng.step()
        drain += 1
    pre = steps[-1] if steps else st0
    row = {"tag": tag, "batch": batch, "end": end, "steps": len(steps),
           "preemptions": pre["preemptions"] - st0["preemptions"], "alloc_fail": pre["alloc_fail"] - st0["alloc_fail"],
           "free_before": int(free0), "free_after": int(eng.state()["free"]), "drain_steps": drain,
           "max_decoding": max((s["decoding"] for s in steps), default=0), "profiled_steps": prof_n,
           "profile": profile, "tokens": {str(i): list(eng.tokens.get(rid, [])) for i, rid in enumerate(ids)}}
    row.update(window_stats(steps, batch, SKIP, window))
    row["trace"] = {k: [s.get(k) for s in steps] for k in ("seconds", "cpu", "running", "decoding", "replays")}
    return row


def warm(eng, tag: str, check) -> dict:
    """One request of WARM_PROMPT tokens and WARM_TOKENS output tokens, stepped to the end."""
    t0 = time.time()
    eng.add(tag, r57.cap_prompt(WARM_PROMPT + 16, 99, 16), WARM_TOKENS)
    n = 0
    while eng.busy() and n < 4 * WARM_TOKENS + 64:
        eng.step()
        n += 1
        check()
    if eng.busy():
        eng.abort([tag])
    return {"tag": tag, "steps": n, "seconds": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--layout", choices=("stock", "compact"), required=True)
    ap.add_argument("--graphs", type=int, choices=(0, 1), required=True)
    ap.add_argument("--capture", choices=("OFF", "VIEWS"), required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--profile-rows", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--model-id", default="google/gemma-4-31B-it")
    ap.add_argument("--revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    ap.add_argument("--deadline", type=float, default=time.time() + 3_600)
    ap.add_argument("--reserve", type=float, default=30.0)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-plant", default="")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = parse_rows(args.rows)
    prof_rows = {int(x) for x in args.profile_rows.split(",") if x.strip()}
    graphs = bool(args.graphs)
    length, window, min_window, min_common, est_row = LENGTH, WINDOW, MIN_WINDOW, MIN_COMMON, EST_ROW
    kw = dict(ENGINE)
    if args.rehearsal:
        length, window, min_window = REH["length"], REH["window"], REH["min_window"]
        min_common, est_row = REH["min_common"], REH["est_row"]
        kw.update(kv_cache_memory_bytes=REH["kv_bytes"], max_model_len=REH["max_model_len"])
    rec = {"engine_name": args.engine, "layout": args.layout, "graphs": graphs, "capture": args.capture,
           "rows_plan": [list(r) for r in rows], "profile_rows": sorted(prof_rows), "length": length,
           "window": window, "skip": SKIP, "min_window": min_window, "min_common": min_common,
           "rehearsal": args.rehearsal, "stub": args.stub, "model_id": args.model_id, "snapshot": args.snapshot,
           "revision": args.revision, "started_at": now(),
           "status": "running", "warm": [], "rows": [], "cut": [], "errors": []}
    path = args.out / "run_e59.json"

    def save() -> None:
        path.write_text(json.dumps(rec, indent=1, default=str) + "\n")

    def on_term(signum, frame):
        raise r57.Terminated("SIGTERM (the session timeout)")
    signal.signal(signal.SIGTERM, on_term)

    def check() -> None:
        if time.time() > args.deadline - args.reserve:
            raise r57.Terminated("the deadline")

    code = 1
    splice = ce.ViewsSplice()
    try:
        problems = plan_problems(args.layout, graphs, args.capture, rows)
        if problems:
            rec["status"], rec["preflight"] = "preflight", problems
            save()
            sys.exit(3)
        if args.stub:
            torch.set_num_threads(1)
            world = StubWorld59(args.layout, args.stub_plant, args.rehearsal, graphs, args.engine)
            mods = {layer: ce.StubAttention(*STUB_HEADS) for layer in STUB_LAYERS}
            for layer, m in mods.items():               # "at load", before the warm-up and the rows (L-26)
                splice.attach(layer, m, torch.full((512,), STUB_W), STUB_HEADS)
            splice.set_mode(args.capture)
            world.mode = args.capture
            rec["hooks"] = {"hooked": [{"layer": layer, "head_size": 512} for layer in STUB_LAYERS], "problems": [],
                            "attached_at": "load_model"}
            rec["engine"] = {"vllm": "STUB"}
            rec["compilation"] = ({"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                                   "capture_sizes": list(CAPTURE_SIZES)} if graphs else
                                  {"mode": "NONE", "cudagraph_mode": "NONE", "capture_sizes": []})
            num_blocks = kw["kv_cache_memory_bytes"] // 2_621_440
            rec["kv"] = {"num_blocks": num_blocks}
            rec["spec"] = {"applied": True, "block": 48} if args.layout == "compact" else None
            small = args.stub_plant == "scratch_small"            # the stock (narrow) scratch: V3 stops the engine
            rows_scr, width = (4, 256) if small else (8, GLOBAL_HEAD)
            rec["scratch"] = [{"seq_threshold_3D": rows_scr, "headdim": 256, "block_size": b, "rows": rows_scr,
                               "shape": [rows_scr, 32, 16, width], "decode_cudagraph_enabled": graphs}
                              for b in (16, 48 if args.layout == "compact" else 32)]
            rec["gpu"] = "STUB"
            eng = StubEngine59(world, num_blocks, length)
            make_prof = StubProfiler

            def set_mode(mode: str) -> None:
                splice.set_mode(mode)
                world.mode = mode
                for layer, m in mods.items():           # one call of the real hooks per mode change
                    m(*ce.stub_qkv(4, STUB_HEADS, STUB_W, layer))
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            ce.install(args.layout)
            from vllm import LLM
            from vllm.v1.worker.gpu import model_runner as mr
            import run_e55 as r55
            torch.backends.cuda.matmul.allow_tf32 = False
            orig_load = mr.GPUModelRunner.load_model

            def load_model(self, *a, **k):              # the hooks before the profile run and the capture
                out = orig_load(self, *a, **k)
                RUNNER["obj"] = self
                rec["hooks"] = ce.register(self.model, splice)
                rec["hooks"]["attached_at"] = "load_model"
                splice.set_mode(args.capture)
                return out

            mr.GPUModelRunner.load_model = load_model
            orig_kv = mr.GPUModelRunner.initialize_kv_cache

            def initialize_kv_cache(self, *a, **k):     # A2: the builders exist here; no warm-up or capture yet
                out = orig_kv(self, *a, **k)
                RUNNER["obj"] = self
                widen_builders(self, "after_kv_init")
                return out

            mr.GPUModelRunner.initialize_kv_cache = initialize_kv_cache
            widen_scratch()
            REPLAYS.install()
            if graphs:
                from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
                kw["enforce_eager"] = False
                kw["compilation_config"] = CompilationConfig(mode=CompilationMode.NONE,
                                                             cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                                                             cudagraph_capture_sizes=list(CAPTURE_SIZES))
            t0 = time.time()
            llm = LLM(model=args.snapshot or args.model_id, revision=None if args.snapshot else args.revision,
                      tokenizer=args.snapshot or args.model_id, **kw)
            rec["engine_seconds"] = round(time.time() - t0, 1)
            signal.signal(signal.SIGTERM, on_term)
            rec["engine_kwargs"] = {k: (str(v) if k == "compilation_config" else v) for k, v in kw.items()}
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["engine"] = r55.engine_record(llm)
            cc = llm.llm_engine.vllm_config.compilation_config
            rec["compilation"] = {"mode": getattr(cc.mode, "name", str(cc.mode)),
                                  "cudagraph_mode": getattr(cc.cudagraph_mode, "name", str(cc.cudagraph_mode)),
                                  "capture_sizes": [int(x) for x in (cc.cudagraph_capture_sizes or [])]}
            rec["spec"] = ce.STATE["spec"]
            rec["kv"] = ce.kv_record(llm)
            rec["tensors"] = llm.apply_model(ce.tensor_record)[0]
            rec["scratch"] = scratch_record()
            rec["scratch_widened"] = dict(WIDENED)
            free, total = torch.cuda.mem_get_info()                # A1 (R-86 F1): the memory after the start
            rec["memory"] = {"after_start_free_gib": round(free / 2**30, 3), "total_gib": round(total / 2**30, 3),
                             "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
            eng = E59Engine(llm)
            make_prof = Profiler

            def set_mode(mode: str) -> None:
                splice.set_mode(mode)

        hooks = rec.get("hooks") or {}
        pre = list(hooks.get("problems", []))
        if not hooks.get("hooked"):
            pre.append("no hooked global layer (the hooks did not attach at load)")
        if args.layout == "compact" and not (rec.get("spec") or {}).get("applied"):
            pre.append("the compact spec did not apply")
        comp = rec.get("compilation") or {}
        if graphs and (comp.get("cudagraph_mode") != "FULL_DECODE_ONLY" or comp.get("mode") != "NONE"
                       or not {3, 4, 8} <= set(comp.get("capture_sizes", []))):
            pre.append(f"graph settings {comp}")
        if not graphs and comp.get("cudagraph_mode") != "NONE":
            pre.append(f"eager settings {comp}")
        pre += scratch_problems(rec.get("scratch") or [], max(b for _, b in rows))
        rec["preflight"] = pre
        save()
        if pre:
            rec["status"] = "preflight"
            save()
            sys.exit(3)

        modes = [args.capture] if graphs else sorted({ROW_MODE[a] for a, _ in rows}, key=lambda m: m != args.capture)
        for mode in modes:
            set_mode(mode)
            r0 = eng.replays()
            rec["warm"].append({"mode": mode, **warm(eng, f"warm-{mode}", check), "replays": eng.replays() - r0})
            save()
        if graphs and eng.replays() == 0:                   # A1 (R-86 B3): the config can fall back to no graphs
            rec["status"], rec["preflight"] = "preflight", ["no CUDA graph replay in the warm-up"]
            save()
            sys.exit(3)
        row_seconds: list[float] = []
        for i, (arm, batch) in enumerate(rows, start=1):
            est = sum(row_seconds) / len(row_seconds) if row_seconds else est_row
            if time.time() + 1.3 * est > args.deadline - args.reserve:
                rec["cut"].append({"row": i, "arm": arm, "estimate_s": round(est, 1), "at": now()})
                save()
                break
            set_mode(ROW_MODE[arm])
            t0 = time.time()
            prof = make_prof(args.out / f"prof_row{i}") if i in prof_rows else None
            row = run_row(eng, f"{args.engine}-r{i}", batch, length, window, check, prof)
            row.update(arm=arm, mode=ROW_MODE[arm], index=i, seconds=round(time.time() - t0, 1))
            row_seconds.append(time.time() - t0)
            rec["rows"].append(row)
            save()
        rec["status"] = "ok"
        code = 0
    except SystemExit:
        rec["finished_at"] = now()
        rec["hook_counts"], rec["hook_errors"] = dict(splice.counts), list(splice.errors)
        save()
        raise
    except r57.Terminated as exc:
        rec["status"] = "terminated"
        rec["errors"].append(str(exc))
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
        rec["traceback"] = traceback.format_exc()[-3_000:]
    rec["finished_at"] = now()
    if not args.stub and torch.cuda.is_available():
        rec.setdefault("memory", {})["max_reserved_end_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    rec["wrapper_counts"] = {k: ce.STATE[k] for k in ("fix_reads", "fix_writes", "pass_reads", "pass_writes")}
    rec["hook_counts"] = dict(splice.counts)
    rec["hook_errors"] = list(splice.errors)
    save()
    print(json.dumps({"engine": args.engine, "status": rec["status"], "rows": len(rec["rows"]),
                      "medians_ms": [r.get("median_ms") for r in rec["rows"]], "cut": len(rec["cut"])}))
    sys.exit(code)


if __name__ == "__main__":
    main()
