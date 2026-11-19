"""E-57 main phases in one engine process (README Sections 4 and 5).

--layout stock (engine A): the stock KV cache with the kernel wrappers (pass-through) and the hooks. The warm-up, the
pre-flight (OFF and VIEWS on the probe), STOCK (mode OFF), the health gate (V5; exit 3), EMULV (mode VIEWS: the
compact arithmetic on the stock cache), the determinism rerun (OFF on the first 32 GSM8K items), then the capacity
phase.
--layout compact (engine B): the compact KV cache (the B3 spec, the V-view fix, the hooks). The warm-up, the
pre-flight (VIEWS, GROSS, NAIVE), COMPACT (mode VIEWS), the capacity phase (with the equal-batch rows at the batch of
engine A), then GROSS (a subset) and NAIVE while the time allows. Mode OFF is not allowed here: the compact cache
needs the VIEWS operands.

The output phase of an arm: the GSM8K scoring sequences (the reference NLL), the GSM8K answers (greedy, at most 640
tokens) and the needle answers (greedy, at most 128 tokens, with the prompt log-probabilities), in the E-56 format.
The engine runs at most 16 sequences, and the needles go in groups of 6, so the KV pool never limits a step and both
engines schedule the same batches (V8 checks the counters).

The capacity phase at each context L (README Section 4.7 as amended by A1): C1, the pool blocks that one request
holds at steady decode (the median over its last 32 decode steps) and at its prefill peak; C2, the largest batch of
L-token requests that all decode at once with no preemption (from the gate-model start, down to a clean row, up by
one while a probe request is admitted), with the decode rate over a window of full steps; the A/A repeat and the
EMULV row at the accepted batch (engine A); C3 (engine B), the same at the batch of engine A. A row ends early (the
first preemption, a finish before the window, a refused admission, the window and the probe) and aborts its
requests.

--stub runs the same flow on the Mac with synthetic outputs and a simulated scheduler and block pool (`StubEngine`,
the gate model; no vLLM; README Section 12).

Usage: run_e57.py --layout stock|compact --data data_e56.json --out <res>/<layout> --snapshot <dir> --model-id <id>
                  --revision <sha> --start <epoch> --deadline <epoch> [--stock-capacity <res>/stock/capacity.json]
                  [--rehearsal] [--stub ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E56, E55 = HERE.parent / "gemma4_kv_tasks", HERE.parent / "gemma4_kv_model"
sys.path[:0] = [str(HERE)]
sys.path += [str(E56), str(E55)]
import compact_engine as ce  # noqa: E402
import compact_layout as cl  # noqa: E402

KV_BYTES = 15 * 2**30                    # the fixed KV pool of both engines (V3)
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_model_len": 66_048,
          "max_num_batched_tokens": 2_048, "max_num_seqs": 16, "kv_cache_memory_bytes": KV_BYTES,
          "enable_prefix_caching": False, "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}
GSM_MAX_TOKENS, NEEDLE_MAX_TOKENS, NEEDLE_GROUP = 640, 128, 6
PROBE_TOKENS, WARM_TOKENS, DET_ITEMS = 1_536, 512, 32
CAP_LENGTHS = (32_768, 65_536)
C1_DECODE, C1_LAST = 64, 32               # C1: 64 decode tokens; the steady footprint is the median of the last 32 steps
CAP_GAP, CAP_MAX_TOKENS = 256, 512        # C2 and C3 rows: prompts of L - 256 tokens; a row ends by abort (A1)
CAP_SKIP, CAP_WINDOW, CAP_MIN_WINDOW = 3, 128, 32
CAP_TRIM = 5.0                            # window steps above 5 times the window median are dropped and counted
PROBE_STEPS, REFUSE_STEPS = 4, 16
GROSS_GSM, GROSS_NEEDLE = 64, 12
STUB_HEADS, STUB_LAYERS, STUB_W = (4, 1, 512), (5, 11, 17, 23, 29, 35, 41, 47, 53, 59), 0.0615234375
REH = {"gsm": 48, "needle": 8, "lengths": (8_192, 16_384), "kv_bytes": 12 * 2**30}


class HealthError(RuntimeError):
    """STOCK failed the health gate (V5): the run stops before the other arms (exit 3)."""


class PreflightError(RuntimeError):
    """A pre-flight check failed (V1, V2 or V4): exit 3."""


class Terminated(BaseException):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def cap_prompt(length: int, index: int, decode: int) -> list[int]:
    """A synthetic prompt of `length - decode` token ids in [1,000, 100,000): no special tokens, distinct per request."""
    rng = np.random.default_rng(seed_of("cap", length, index))
    return rng.integers(1_000, 100_000, size=length - decode).tolist()


def footprint_from_steps(steps: list[dict], free0: int, last: int = C1_LAST) -> dict:
    """C1 (A1, R-82 B1). The steady decode footprint is the median of the held blocks over the last `last` steps in
    which the request decodes: the first decode steps still hold the sliding blocks of the last prefill chunk (with
    async scheduling the engine frees them one or two steps later). The peak over all steps is the prefill peak,
    which is about the admission reservation of the request."""
    held = [free0 - s["free"] for s in steps]
    dec = [free0 - s["free"] for s in steps if s["running"] == 1 and s["decoding"] == 1]
    return {"blocks": float(np.median(dec[-last:])) if len(dec) >= last + CAP_SKIP else None,
            "blocks_peak": int(max(held)) if held else None, "blocks_first_decode": int(dec[0]) if dec else None,
            "decode_steps": len(dec), "steps": len(steps), "free_idle": int(free0)}


def window_from_steps(steps: list[dict], batch: int) -> dict:
    """The decode window of a row (A1, R-82 B3): the first run of steps in which all `batch` requests run and decode
    (before the probe), without its first CAP_SKIP steps (the async lag: those calls still wait on prefill chunks),
    at most CAP_WINDOW steps. Steps above CAP_TRIM times the window median are dropped and counted."""
    run: list[int] = []
    for i, s in enumerate(steps):
        if s.get("probe"):
            break
        if s["running"] == batch and s["decoding"] == batch:
            run.append(i)
        elif run:
            break
    win = run[CAP_SKIP:CAP_SKIP + CAP_WINDOW]
    out = {"full_steps": len(run), "window_steps": 0, "dropped": 0}
    if win:
        secs = np.array([steps[i]["seconds"] for i in win], dtype=np.float64)
        med = float(np.median(secs))
        keep = secs[secs <= CAP_TRIM * med]
        out.update(window_steps=int(keep.size), dropped=int(secs.size - keep.size), median_ms=round(1_000 * med, 3),
                   step_ms=round(1_000 * float(keep.mean()), 3), tokens_per_s=round(batch / float(keep.mean()), 3),
                   window_seconds=round(float(keep.sum()), 4),
                   held=float(np.median([steps[i]["held"] for i in win])))
    return out


def clean_row(r: dict | None) -> bool:
    """A clean row (A1, R-82 B2): no preemption and a decode window of CAP_MIN_WINDOW steps or more. A deferred
    admission (a failed allocation while the previous request frees its prefill blocks) is allowed and reported."""
    return bool(r) and r.get("preemptions") == 0 and r.get("window_steps", 0) >= CAP_MIN_WINDOW


def start_batch(free: int, steady: float | None, peak: int | None, cap: int) -> int:
    """The first batch of the search (the gate model, R-82 B2): the engine admits a request only if its full prefill
    reservation (about the C1 peak) fits the free blocks, so b - 1 requests at the steady footprint and one peak must
    fit the pool."""
    if not steady or not peak:
        return 1
    return int(max(1, min(cap, 1 + (free - peak) // steady)))


def search_batch(run_row, b0: int, cap: int, fits) -> tuple[list[dict], int | None, str]:
    """C2 (A1): from b0 down to the first clean row; then up by one while the probe of the largest clean row was
    admitted. Each batch runs once. Returns the rows, the index of the accepted row (the largest clean batch) and how
    the limit shows: "next batch failed", "probe refused", "max_num_seqs", "time" or "none"."""
    rows: list[dict] = []
    tried: set[int] = set()
    b, best = max(1, min(b0, cap)), None
    while 1 <= b <= cap and b not in tried:
        if rows and not fits(b):
            break
        tried.add(b)
        r = run_row(b)
        rows.append(r)
        if clean_row(r):
            best = len(rows) - 1
            if r.get("probe_admitted") and b + 1 not in tried:
                b += 1
                continue
            break
        if best is not None:
            break
        b -= 1
    if best is None:
        return rows, None, "none"
    nb = rows[best]["batch"]
    if any(x["batch"] == nb + 1 and not clean_row(x) for x in rows):
        limit = "next batch failed"
    elif nb >= cap:                       # A2: at the cap the engine refuses the probe for max_num_seqs, not the pool
        limit = "max_num_seqs"
    elif rows[best].get("probe_admitted") is False:
        limit = "probe refused"
    else:
        limit = "time"
    return rows, best, limit


def row_limit(length: int, batch: int) -> int:
    """A step cap for one row: the prefill chunks of every request, the window, the probe and a margin."""
    return batch * (math.ceil(length / 1_024) + 8) + CAP_SKIP + CAP_WINDOW + PROBE_STEPS + REFUSE_STEPS + 64


def footprint_run(eng, length: int, check) -> dict:
    """C1: one request of L - 64 prompt tokens and 64 decode tokens, stepped to the end."""
    free0 = eng.state()["free"]
    rid = f"fp-{length}-{time.time_ns()}"
    eng.add(rid, cap_prompt(length, 0, C1_DECODE), C1_DECODE)
    steps: list[dict] = []
    while eng.busy() and len(steps) < row_limit(length, 1) + C1_DECODE:
        dt, _ = eng.step()
        st = eng.state()
        st["seconds"] = dt
        steps.append(st)
        check()
    if eng.busy():
        eng.abort([rid])
    out = footprint_from_steps(steps, free0)
    out["trace"] = {"held": [free0 - s["free"] for s in steps], "decoding": [s["decoding"] for s in steps]}
    return out


def capacity_run(eng, length: int, batch: int, probe: bool, check) -> dict:
    """One C2, C3, repeat or EMULV row (A1): `batch` requests of L - CAP_GAP prompt tokens. The row ends at the first
    preemption, at a finish before the window, when a waiting request is refused for REFUSE_STEPS steps while every
    running request decodes, at the step cap, or after the window (CAP_SKIP + CAP_WINDOW full steps) and, with
    `probe`, one more request: admitted within PROBE_STEPS steps or not. Then it aborts its requests."""
    base = time.time_ns()
    ids = [f"cap-{base}-{i}" for i in range(batch)]
    st0 = eng.state()
    free0 = st0["free"]
    for i, rid in enumerate(ids):
        eng.add(rid, cap_prompt(length, i + 1, CAP_GAP), CAP_MAX_TOKENS)
    steps: list[dict] = []
    end, full, refused, probe_at = "step cap", 0, 0, None
    while eng.busy() and len(steps) < row_limit(length, batch):
        dt, done = eng.step()
        st = eng.state()
        st.update(seconds=dt, finished=len(done), held=free0 - st["free"], probe=probe_at is not None)
        steps.append(st)
        check()
        if probe_at is not None:
            if st["waiting"] == 0 or len(steps) - probe_at >= PROBE_STEPS:
                end = "window"
                break
            continue
        if st["preemptions"] > st0["preemptions"]:
            end = "preempted"
            break
        if done:
            end = "finished before the window"
            break
        if st["running"] == batch and st["decoding"] == batch:
            full += 1
        refused = refused + 1 if st["waiting"] and st["running"] and st["running"] == st["decoding"] else 0
        if refused >= REFUSE_STEPS:
            end = "admission refused"
            break
        if full >= CAP_SKIP + CAP_WINDOW:
            if not probe:
                end = "window"
                break
            pid = f"cap-{base}-probe"
            ids.append(pid)
            eng.add(pid, cap_prompt(length, batch + 1, CAP_GAP), CAP_MAX_TOKENS)
            probe_at = len(steps)
    eng.abort(ids)
    drain = 0
    while eng.busy() and drain < 64:
        eng.step()
        drain += 1
    pre = steps[probe_at - 1] if probe_at else (steps[-1] if steps else st0)
    row = {"batch": batch, "end": end, "steps": len(steps), "preemptions": pre["preemptions"] - st0["preemptions"],
           "alloc_fail": pre["alloc_fail"] - st0["alloc_fail"], "free_before": int(free0),
           "free_after": int(eng.state()["free"]), "drain_steps": drain,
           "max_decoding": max((s["decoding"] for s in steps), default=0)}
    row.update(window_from_steps(steps, batch))
    if probe_at is not None:
        last = steps[-1]
        row.update(probe_admitted=bool(last["waiting"] == 0), probe_steps=len(steps) - probe_at,
                   probe_alloc_fail=last["alloc_fail"] - pre["alloc_fail"])
    row["trace"] = {k: [s[k] for s in steps] for k in ("seconds", "held", "running", "decoding", "waiting")}
    return row


class VllmEngine:
    """The capacity phase's view of the in-process vLLM engine: add, step, state, abort."""

    def __init__(self, llm):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        self.llm, self.eng, self.sp, self.tp = llm, llm.llm_engine, SamplingParams, TokensPrompt

    def add(self, rid: str, prompt: list[int], max_tokens: int) -> None:
        params = self.sp(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False)
        self.eng.add_request(rid, self.tp(prompt_token_ids=prompt), params)

    def step(self) -> tuple[float, list[str]]:
        t1 = time.perf_counter()
        outs = self.eng.step()
        return time.perf_counter() - t1, [o.request_id for o in outs if getattr(o, "finished", False)]

    def state(self) -> dict:
        st = ce.running_state(self.llm)
        st["free"] = ce.free_blocks(self.llm)
        return st

    def busy(self) -> bool:
        return bool(self.eng.has_unfinished_requests())

    def abort(self, ids: list[str]) -> None:
        self.eng.abort_request(list(ids))


class StubEngine:
    """The --stub stand-in for the vLLM scheduler and block pool in the capacity phase (the gate model of R-82, A1):
    chunked prefill with 2,048 tokens per step (decodes first), the full-sequence admission check
    (`scheduler_reserve_full_isl`: the global blocks of the prompt plus at most 321 blocks per sliding group must be
    free), sliding blocks kept for the window plus the tokens in flight (two steps in prefill, one in decode), one
    decode token per step, and the step time of StubWorld. The plant "preempt" counts a preemption when two requests
    run."""

    BUDGET, WINDOW, SBLOCK, GROUPS = 2_048, 1_024, 16, 5

    def __init__(self, world: "StubWorld", num_blocks: int):
        self.world, self.pool = world, num_blocks - 1
        self.gb = 48 if world.compact else 32
        self.cap = math.ceil((self.WINDOW - 1 + 2 * self.BUDGET) / self.SBLOCK) + 1
        self.reqs: dict[str, dict] = {}
        self.preempted = False

    def reservation(self, p: int) -> int:
        return math.ceil(p / self.gb) + self.GROUPS * min(math.ceil(p / self.SBLOCK), self.cap)

    def held(self, r: dict) -> int:
        if r["c"] == 0:
            return 0
        h = r["hist"][-2:]
        inflight = sum(h) if any(n > 1 for n in h) else 1
        slide = min(math.ceil(r["c"] / self.SBLOCK), math.ceil((self.WINDOW - 1 + inflight) / self.SBLOCK) + 1)
        return math.ceil(r["c"] / self.gb) + self.GROUPS * slide

    def free(self) -> int:
        return self.pool - sum(self.held(r) for r in self.reqs.values() if r["state"] == "running")

    def add(self, rid: str, prompt: list[int], max_tokens: int) -> None:
        if not self.busy():
            self.preempted = False
        self.reqs[rid] = {"p": len(prompt), "m": max_tokens, "c": 0, "out": 0, "state": "waiting", "hist": []}

    def busy(self) -> bool:
        return any(r["state"] in ("waiting", "running") for r in self.reqs.values())

    def step(self) -> tuple[float, list[str]]:
        run = [k for k, r in self.reqs.items() if r["state"] == "running"]
        budget, sched, pre = self.BUDGET, {}, set()
        for k in run:
            if self.reqs[k]["c"] >= self.reqs[k]["p"]:
                sched[k] = 1
                budget -= 1
        for k in run:
            r = self.reqs[k]
            if r["c"] < r["p"] and budget > 0:
                sched[k] = min(r["p"] - r["c"], budget)
                budget -= sched[k]
                pre.add(k)
        for k, n in sched.items():
            self.reqs[k]["c"] += n
            self.reqs[k]["hist"].append(n)
        for k, r in self.reqs.items():
            if r["state"] != "waiting" or budget <= 0:
                continue
            if self.free() < self.reservation(r["p"]):
                self.world.counters["alloc_fail"] += 1
                break
            n = min(r["p"], budget)
            r.update(state="running", c=n, hist=[n])
            sched[k] = n
            budget -= n
            pre.add(k)
        done = []
        for k in sched:
            r = self.reqs[k]
            if r["c"] >= r["p"]:
                r["out"] += 1
                if r["out"] >= r["m"]:
                    r["state"] = "finished"
                    done.append(k)
        if self.world.plant == "preempt" and len(run) >= 2 and not self.preempted:
            self.world.counters["preemptions"] += 1
            self.preempted = True
        ctx = max((self.reqs[k]["c"] for k in sched), default=1)
        dec = sum(1 for k in sched if k not in pre)
        return (1.0 if pre else self.world.step_seconds(dec, ctx)), done

    def state(self) -> dict:
        rs = list(self.reqs.values())
        run = [r for r in rs if r["state"] == "running"]
        return {"running": len(run), "decoding": sum(1 for r in run if r["c"] >= r["p"]),
                "waiting": sum(1 for r in rs if r["state"] == "waiting"),
                "preemptions": self.world.counters["preemptions"], "alloc_fail": self.world.counters["alloc_fail"],
                "free": self.free()}

    def abort(self, ids: list[str]) -> None:
        for k in ids:
            if k in self.reqs and self.reqs[k]["state"] in ("waiting", "running"):
                self.reqs[k]["state"] = "aborted"
        self.reqs = {k: r for k, r in self.reqs.items() if r["state"] in ("waiting", "running")}


class StubWorld:
    """The --stub stand-in for the model and the engine. Answers: STOCK draws each item once (right with chance
    0.97 or 0.4); EMULV and COMPACT give the STOCK answer and scores bit for bit, except a planted difference;
    GROSS breaks most answers; NAIVE changes a few. The capacity phase runs on `StubEngine` (31B: global block 32 or
    48, 5 sliding groups of block 16), with a decode step time from a simple model."""

    def __init__(self, layout: str, plant: str, rehearsal: bool):
        self.layout, self.plant, self.rehearsal = layout, plant, rehearsal
        self.compact = layout == "compact" and plant != "no_compact"
        self.counters = {"fix_reads": 0, "fix_writes": 0, "pass_reads": 0, "pass_writes": 0, "alloc_fail": 0,
                         "preemptions": 0}

    def _right(self, item: dict, arm: str) -> bool:
        base = np.random.default_rng(seed_of("base", item["id"]))
        p = 0.97 if seed_of("p", item["id"]) % 5 else 0.4
        right = bool(base.random() < p)
        r = np.random.default_rng(seed_of(arm, item["id"]))
        if arm == "GROSS" and r.random() < 0.8:
            return False
        if arm == "NAIVE" and r.random() < 0.05:
            return not right
        if arm == "COMPACT" and self.plant == "compact_differs" and r.random() < 0.1:
            return not right
        return right

    def answer(self, arm: str, item: dict, kind: str) -> dict:
        a = "STOCK" if arm in ("EMULV", "DET") or (arm == "COMPACT" and self.plant != "compact_differs") else arm
        right = self._right(item, a)
        n = 20 + seed_of("len", item["id"]) % 40 if kind == "gsm8k" else 6
        ids = [(seed_of("tok", item["id"]) + i) % 50_000 for i in range(n)]
        if a != "STOCK":
            ids[n // 2:] = [x + 1 for x in ids[n // 2:]]
        if kind == "gsm8k":
            gold = int(float(item["gold"]))
            text = f"Steps. #### {gold if right else gold + 1 + seed_of(a, item['id']) % 7}"
        else:
            text = item["gold"] if right else f"wrong-{1000 + seed_of(a, item['id']) % 9000}"
        return {"text": text, "ids": ids, "finish": "stop"}

    def score(self, arm: str, ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ids) - 1
        lp = -np.abs(np.random.default_rng(seed_of("score", ids[:8], len(ids))).normal(0.4, 0.3, n))
        a = "STOCK" if arm in ("EMULV", "DET", "COMPACT") else arm
        if arm == "COMPACT" and self.plant == "compact_differs":
            lp = lp - 1e-3
        if arm == "COMPACT" and self.plant == "compact_near":       # a tiny float difference, the same answers
            lp = lp - 1e-4
        if a != "STOCK":
            lp = lp - {"GROSS": 2.0, "NAIVE": 0.01}.get(a, 0.0)
        return lp.astype(np.float32), np.asarray(ids[1:], dtype=np.int32), np.ones(n, dtype=np.int32)

    def step_seconds(self, batch: int, length: int) -> float:
        kv = (cl.ROW if self.compact else 1_024) / 1_024
        return 0.033 + 0.0004 * batch * (length / 32_768) * kv + (0.004 if self.plant == "slow" else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("stock", "compact"), required=True)
    ap.add_argument("--data", type=Path, default=E56 / "data_e56.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--model-id", default="google/gemma-4-31B-it")
    ap.add_argument("--revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    ap.add_argument("--start", type=float, default=time.time())
    ap.add_argument("--deadline", type=float, default=time.time() + 3_600)
    ap.add_argument("--reserve", type=float, default=60.0)
    ap.add_argument("--stock-capacity", type=Path, default=None)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-plant", default="",
                    help="no_compact, compact_differs, compact_near, no_fix, preempt, slow, unhealthy, step_error or "
                         "short_probe (the stub)")
    args = ap.parse_args()

    out, arrays = args.out, args.out / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    gsm, needle = data["gsm8k"], data["needle"]
    lengths = CAP_LENGTHS
    engine_kwargs = dict(ENGINE)
    if args.rehearsal:
        gsm, needle, lengths = gsm[:REH["gsm"]], needle[:REH["needle"]], REH["lengths"]
        engine_kwargs.update(kv_cache_memory_bytes=REH["kv_bytes"], max_model_len=lengths[-1] + 512)
    probe = data["health"][0]["ids"][:PROBE_TOKENS]
    compact = args.layout == "compact"
    rec = {"phase": "main", "layout": args.layout, "started_at": now(), "status": "running", "stub": args.stub,
           "rehearsal": args.rehearsal, "model_id": args.model_id, "revision_arg": args.revision,
           "snapshot_revision": Path(args.snapshot).name if args.snapshot else None, "items": [len(gsm), len(needle)],
           "arms": {}, "preflight": {}, "phase_counters": {}, "cut": [], "step_errors": [], "errors": []}
    run_path = out / "run_main.json"
    splice = None
    counters = None

    def save() -> None:
        if splice is not None:
            rec["evidence"] = splice.evidence()
        run_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")

    def on_term(signum, frame):
        raise Terminated("SIGTERM (the session timeout)")
    signal.signal(signal.SIGTERM, on_term)
    arm_seconds: list[float] = []

    def fits(name: str, factor: float = 1.5) -> bool:
        est = sum(arm_seconds) / len(arm_seconds) if arm_seconds else 0.0
        ok = time.time() + factor * est <= args.deadline - args.reserve
        if not ok:
            rec["cut"].append({"step": name, "estimate_s": round(est, 1), "at": now()})
            save()
        return ok

    def snap() -> dict:
        return dict(counters())

    def check_time() -> None:
        if time.time() > args.deadline - args.reserve:
            raise Terminated("the deadline in the capacity phase")

    code = 1
    cap_eng = None
    try:
        if args.stub:
            torch.set_num_threads(1)
            world = StubWorld(args.layout, args.stub_plant, args.rehearsal)
            counters = lambda: world.counters  # noqa: E731
            rec["engine"] = {"vllm": "STUB", **{k: engine_kwargs[k] for k in ("max_model_len", "max_num_seqs",
                                                                                "max_num_batched_tokens")}}
            rec["engine_kwargs"] = engine_kwargs
            rec["gpu"] = "STUB"
            num_blocks = engine_kwargs["kv_cache_memory_bytes"] // 2_621_440
            gb = 48 if world.compact else 32
            rec["kv"] = {"num_blocks": num_blocks, "pool_blocks": num_blocks, "free_blocks": num_blocks - 1,
                         "bytes_per_block": 2_621_440, "layout": "LBNHC", "tensors": 6,
                         "groups": [{"type": "SlidingWindowSpec", "layers": 10, "block_size": 16, "page_bytes": 262_144,
                                     "real_page_bytes": 262_144, "page_size_padded": None,
                                     "state_content_bytes": None, "num_kv_heads": 16, "head_size": 256,
                                     "sliding_window": 1_024}] * 5
                         + [{"type": "FullAttentionSpec", "layers": 10, "block_size": gb, "page_bytes": 262_144,
                             "real_page_bytes": gb * 4 * (640 if world.compact else 1_024) * 2,
                             "page_size_padded": 262_144 if world.compact else None,
                             "state_content_bytes": 1_280 if world.compact else None, "num_kv_heads": 4,
                             "head_size": 512, "sliding_window": None}]}
            rec["tensors"] = {"global": {"layer": 5, "shape": [num_blocks, 4, gb, 640 if world.compact else 1_024],
                                         "stride": [], "dtype": "torch.bfloat16"}}
            rec["spec"] = ({"applied": True, "layers": [f"layer.{i}" for i in range(10)], "block": 48,
                            "compact_page": 245_760, "sliding_page": 262_144} if world.compact else None)
            splice = ce.ViewsSplice()                  # the real hooks on stub modules (L-26)
            stub_mods = {layer: ce.StubAttention(*STUB_HEADS) for layer in STUB_LAYERS}
            for layer, m in stub_mods.items():
                splice.attach(layer, m, torch.full((512,), STUB_W), STUB_HEADS)
            rec["hooks"] = {"hooked": [{"layer": layer, "num_heads": STUB_HEADS[0], "num_kv_heads": STUB_HEADS[1],
                                        "head_size": 512} for layer in STUB_LAYERS], "problems": []}

            def stub_forward(t: int, seed: int) -> None:
                """One engine call of every global layer with t tokens, in layer order."""
                for layer, m in stub_mods.items():
                    m(*ce.stub_qkv(t, STUB_HEADS, STUB_W, seed * 100 + layer))

            def kernel_calls():
                """One engine step of the stub: 10 global and 50 sliding layers call the write and the read."""
                fix = compact and args.stub_plant != "no_fix"
                for k in ("reads", "writes"):
                    world.counters[f"fix_{k}" if fix else f"pass_{k}"] += 10
                    world.counters[f"pass_{k}"] += 50

            def set_mode(mode):
                if compact and mode == "OFF":
                    raise RuntimeError("mode OFF is not allowed with the compact cache")
                splice.set_mode(mode)

            def score(arm, seqs):
                kernel_calls()
                stub_forward(8, 1)
                if ce.STATE["sched"] is not None:        # a schedule that depends on the inputs only, as the engine's
                    ce.STATE["sched"].append(sorted(len(s) for s in seqs))
                return [world.score(arm, s) for s in seqs]

            def generate(arm, items, max_tokens, kind):
                if args.stub_plant == "step_error" and arm == "NAIVE":
                    raise RuntimeError("planted NAIVE failure")
                if ce.STATE["sched"] is not None:
                    ce.STATE["sched"].append(sorted(len(i["ids"]) for i in items))
                kernel_calls()
                stub_forward(8, 2)
                stub_forward(1, 3)
                res = {i["id"]: world.answer(arm, i, kind) for i in items}
                plp = {i["id"]: world.score(arm, i["ids"]) for i in items} if kind == "needle" else None
                return res, plp

            def probe_lps(mode):
                kernel_calls()
                set_mode(mode)
                splice.new_doc()
                stub_forward(len(probe) if args.stub_plant != "short_probe" else 8, 7)
                return world.score("STOCK" if mode in ("OFF", "VIEWS") else mode, probe)[0]

            def gap_evidence(modes):
                return splice.evidence()

            cap_eng = StubEngine(world, num_blocks)

            def warm(mode):
                set_mode(mode)
                kernel_calls()
                stub_forward(WARM_TOKENS, 9)
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            ce.install(args.layout)
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            import run_e55 as r55
            torch.backends.cuda.matmul.allow_tf32 = False
            counters = lambda: {k: ce.STATE[k] for k in ("fix_reads", "fix_writes", "pass_reads", "pass_writes",  # noqa: E731
                                                         "alloc_fail", "preemptions")}
            t0 = time.time()
            llm = LLM(model=args.snapshot or args.model_id, revision=None if args.snapshot else args.revision,
                      tokenizer=args.snapshot or args.model_id, **engine_kwargs)
            rec["engine_seconds"] = round(time.time() - t0, 1)
            signal.signal(signal.SIGTERM, on_term)
            rec["engine_kwargs"] = engine_kwargs
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["engine"] = r55.engine_record(llm)
            rec["spec"] = ce.STATE["spec"]
            rec["kv"] = ce.kv_record(llm)
            rec["tensors"] = llm.apply_model(ce.tensor_record)[0]
            tok = llm.get_tokenizer()
            got = [tok.encode(c["text"], add_special_tokens=False) for c in data["tok_check"]]
            rec["tok_check"] = {"ok": got == [c["ids"] for c in data["tok_check"]]}
            splice = ce.ViewsSplice()
            rec["hooks"] = llm.apply_model(lambda model: ce.register(model, splice))[0]
            save()
            sp_nll = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False,
                                    flat_logprobs=True)

            def set_mode(mode):
                if compact and mode == "OFF":
                    raise RuntimeError("mode OFF is not allowed with the compact cache")
                splice.set_mode(mode)

            def score(arm, seqs):
                outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], sp_nll, use_tqdm=False)
                return [r55.parse_prompt_logprobs(o.prompt_logprobs, s) for o, s in zip(outs, seqs)]

            def generate(arm, items, max_tokens, kind):
                extra = {"prompt_logprobs": 1, "flat_logprobs": True} if kind == "needle" else {}
                params = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=0, detokenize=True,
                                        stop_token_ids=[data["eot_id"]], **extra)
                outs = llm.generate([TokensPrompt(prompt_token_ids=i["ids"]) for i in items], params, use_tqdm=False)
                res = {i["id"]: {"text": o.outputs[0].text, "ids": list(o.outputs[0].token_ids),
                                 "finish": o.outputs[0].finish_reason} for i, o in zip(items, outs)}
                plp = ({i["id"]: r55.parse_prompt_logprobs(o.prompt_logprobs, i["ids"]) for i, o in zip(items, outs)}
                       if kind == "needle" else None)
                return res, plp

            def probe_lps(mode):
                set_mode(mode)
                splice.new_doc()
                return score(mode, [probe])[0][0]

            def gap_evidence(modes):
                return splice.evidence()

            cap_eng = VllmEngine(llm)

            def warm(mode):
                set_mode(mode)
                llm.generate([TokensPrompt(prompt_token_ids=probe[:WARM_TOKENS])], sp_nll, use_tqdm=False)

        def run_arm(arm: str, mode: str, items_gsm=None, items_needle=None) -> dict:
            """One arm: the scoring sequences, the GSM8K answers, the needle answers in groups. Saves the arrays and the
            outputs; records the counters of the phase (V2, V8)."""
            items_gsm = gsm if items_gsm is None else items_gsm
            items_needle = needle if items_needle is None else items_needle
            set_mode(mode)
            c0, t0, res, times, finite, sched = snap(), time.time(), {}, {}, True, {}
            try:                                       # A1 (R-82 F4): a schedule fingerprint per phase
                ce.STATE["sched"] = []
                scored = score(arm, [g["score_ids"] for g in items_gsm])
                sched["score"], ce.STATE["sched"] = ce.schedule_digest(ce.STATE["sched"]), []
                for g, (lp, top1, rank) in zip(items_gsm, scored):
                    a = g["answer_start"] - 1
                    finite &= bool(np.isfinite(lp[a:]).all())
                    res[f"{g['id']}_lp"], res[f"{g['id']}_top1"], res[f"{g['id']}_rank"] = lp[a:], top1[a:], rank[a:]
                times["score"] = round(time.time() - t0, 1)
                t1 = time.time()
                outs = {"gsm8k": generate(arm, items_gsm, GSM_MAX_TOKENS, "gsm8k")[0], "needle": {}}
                sched["gsm8k"], ce.STATE["sched"] = ce.schedule_digest(ce.STATE["sched"]), []
                times["gsm8k"] = round(time.time() - t1, 1)
                t1 = time.time()
                for s in range(0, len(items_needle), NEEDLE_GROUP):
                    got, plp = generate(arm, items_needle[s:s + NEEDLE_GROUP], NEEDLE_MAX_TOKENS, "needle")
                    outs["needle"].update(got)
                    for i, (lp, top1, rank) in (plp or {}).items():
                        finite &= bool(np.isfinite(lp[1_024:]).all())
                        res[f"{i}_plp"], res[f"{i}_prank"] = lp[1_024:], rank[1_024:]
                sched["needle"] = ce.schedule_digest(ce.STATE["sched"])
                times["needle"] = round(time.time() - t1, 1)
            finally:
                ce.STATE["sched"] = None
            np.savez(arrays / f"{arm}.npz", **res)
            (out / f"outputs_{arm}.json").write_text(json.dumps(outs) + "\n")
            c1 = snap()
            secs = time.time() - t0
            rec["arms"][arm] = {"mode": mode, "seconds": round(secs, 1), "times": times, "finite": finite,
                                "items": [len(items_gsm), len(items_needle)], "finished_at": now(),
                                "counters": {k: c1[k] - c0[k] for k in c1}, "schedule": sched}
            arm_seconds.append(secs)
            save()
            return outs

        def preflight(modes) -> dict:
            t0 = time.time()
            lps = {m: probe_lps(m) for m in modes}
            for m, lp in lps.items():
                np.save(arrays / f"probe_{m}.npy", lp)
            ev = gap_evidence(modes)
            return {"tokens": len(probe), "seconds": round(time.time() - t0, 1), "modes": list(modes),
                    "finite": {m: bool(np.isfinite(lp).all()) for m, lp in lps.items()},
                    **{k: ev[k] for k in ("gaps", "gaps_id", "seps", "hook_errors", "counts")}}

        row_rate: list[float] = []                     # seconds per (token of L x request) of the rows so far

        def fits_row(length: int, batch: int, name: str) -> bool:
            est = 1.2 * max(row_rate) * length * batch if row_rate else 180.0
            ok = time.time() + est <= args.deadline - args.reserve
            if not ok:
                rec["cut"].append({"step": name, "estimate_s": round(est, 1), "at": now()})
                save()
            return ok

        def timed_row(length: int, batch: int, probe: bool) -> dict:
            t0 = time.time()
            r = capacity_run(cap_eng, length, batch, probe, check_time)
            r["seconds"] = round(time.time() - t0, 1)
            row_rate.append(r["seconds"] / (length * batch))
            return r

        def capacity_phase(equal_batches: dict | None) -> dict:
            """C1 and C2 at every length first (the decision rows), then the OVH inputs (the A/A repeats in engine A,
            C3 in engine B), then the EMULV rows (engine A, reported) (Section 4.7, A1; the order is A2: the
            rehearsal cut a reported row for time)."""
            set_mode("VIEWS" if compact else "OFF")
            cap_seqs = int(engine_kwargs["max_num_seqs"])
            cap = {"lengths": list(lengths), "rows": {}, "constants": {
                "c1_decode": C1_DECODE, "c1_last": C1_LAST, "gap": CAP_GAP, "max_tokens": CAP_MAX_TOKENS,
                "skip": CAP_SKIP, "window": CAP_WINDOW, "min_window": CAP_MIN_WINDOW, "trim": CAP_TRIM,
                "probe_steps": PROBE_STEPS, "refuse_steps": REFUSE_STEPS}}

            def write() -> None:
                (out / "capacity.json").write_text(json.dumps(cap, indent=1) + "\n")
                save()

            for length in lengths:                             # pass 1: C1 and C2 (the decision rows)
                t0 = time.time()
                fp = footprint_run(cap_eng, length, check_time)
                fp["seconds"] = round(time.time() - t0, 1)
                b0 = start_batch(fp["free_idle"], fp["blocks"], fp["blocks_peak"], cap_seqs)
                rows, best, limit = search_batch(lambda b: timed_row(length, b, True), b0, cap_seqs,
                                                 lambda b: fits_row(length, b, f"capacity {length} batch {b}"))
                acc = rows[best] if best is not None else None
                cap["rows"][str(length)] = {"footprint": fp, "start_batch": b0, "c2": rows,
                                            "accepted_batch": acc["batch"] if acc else None, "limit": limit}
                write()
            for length in lengths:                             # pass 2: the A/A repeat (A) or C3 (B), for OVH
                row = cap["rows"][str(length)]
                acc_b = row["accepted_batch"]
                if equal_batches is None and acc_b and fits_row(length, acc_b, f"capacity {length} repeat"):
                    row["c2_repeat"] = timed_row(length, acc_b, False)               # the A/A of the rate
                eq = (equal_batches or {}).get(str(length))
                if eq:
                    same = [r for r in row["c2"] if r["batch"] == int(eq) and clean_row(r)]
                    if same:                                  # the clean C2 row at that batch is the C3 row
                        row["c3"] = {**{k: x for k, x in same[0].items() if k != "trace"}, "reused_c2_row": True}
                    elif fits_row(length, int(eq), f"capacity {length} C3"):
                        row["c3"] = timed_row(length, int(eq), False)
                write()
            if equal_batches is None:                          # pass 3: the EMULV rows (the hook cost, R-82 F3)
                for length in lengths:
                    row = cap["rows"][str(length)]
                    acc_b = row["accepted_batch"]
                    if acc_b and fits_row(length, acc_b, f"capacity {length} EMULV"):
                        set_mode("VIEWS")
                        try:
                            row["emulv"] = timed_row(length, acc_b, False)
                        finally:
                            set_mode("OFF")
                        write()
            return cap

        warm("VIEWS" if compact else "OFF")
        if splice is not None:
            splice.reset_evidence()
        rec["preflight"] = preflight(("VIEWS", "GROSS", "NAIVE") if compact else ("OFF", "VIEWS"))
        if splice is not None:
            splice.reset_evidence()
        rec["phase_counters"]["after_preflight"] = snap()
        save()
        problems = []
        if rec["hooks"]["problems"]:
            problems += rec["hooks"]["problems"]
        if compact and not (rec.get("spec") or {}).get("applied"):
            problems.append("the compact spec was not applied")
        if compact and rec["phase_counters"]["after_preflight"]["fix_reads"] == 0:
            problems.append("no V-view fix in the pre-flight")
        if not compact and rec["phase_counters"]["after_preflight"]["fix_reads"]:
            problems.append("a V-view fix in the stock engine")
        n_hooked = len(rec["hooks"]["hooked"])
        for m in [x for x in ("OFF", "VIEWS") if x in rec["preflight"]["modes"]]:
            gaps = rec["preflight"]["gaps"].get(m, {})
            if len(gaps) != n_hooked:
                problems.append(f"{m}: attention-level gaps for {len(gaps)} of {n_hooked} layers")
            elif max(gaps.values()) > 0.02:
                problems.append(f"{m}: an attention-level gap above 0.02")
        rec["preflight"]["problems"] = problems
        save()
        if problems and not args.rehearsal:
            raise PreflightError("; ".join(problems[:6]))
        if not compact:
            outs = run_arm("STOCK", "OFF")
            import analyze_e56 as an56
            if args.stub and args.stub_plant == "unhealthy":
                bad = ["planted: unhealthy"]
            else:
                arr = dict(np.load(arrays / "STOCK.npz"))
                sub = dict(data, gsm8k=gsm, needle=needle)
                bad, m = an56.health_problems(an56.grade(sub, outs), an56.ref_nll(arr, sub), sub)
                rec["health"] = {"problems": bad, **m}
            rec.setdefault("health", {})["problems"] = bad
            save()
            if bad and not args.rehearsal:
                raise HealthError("; ".join(bad))
            run_arm("EMULV", "VIEWS")
            run_arm("DET", "OFF", items_gsm=gsm[:DET_ITEMS], items_needle=[])
            rec["decision_arms_done"] = True
            save()
            rec["capacity"] = capacity_phase(None)
        else:
            run_arm("COMPACT", "VIEWS")
            rec["decision_arms_done"] = True
            save()
            eq = None
            if args.stock_capacity and args.stock_capacity.exists():
                sc = json.loads(args.stock_capacity.read_text())
                eq = {L: r["accepted_batch"] for L, r in sc.get("rows", {}).items() if r.get("accepted_batch")}
            rec["equal_batches"] = eq
            rec["capacity"] = capacity_phase(eq)
            for arm, kw in (("GROSS", {"items_gsm": gsm[:GROSS_GSM], "items_needle": needle[:GROSS_NEEDLE]}),
                            ("NAIVE", {})):
                if not fits(arm):
                    continue
                try:
                    run_arm(arm, arm, **kw)
                except Exception as exc:  # a reported arm does not stop the run
                    rec["step_errors"].append({"step": arm, "error": f"{type(exc).__name__}: {exc}"[:500],
                                               "at": now()})
                    save()
            set_mode("VIEWS")
        rec["phase_counters"]["end"] = snap()
        rec["status"] = "ok"
        code = 0
    except BaseException as exc:  # recorded: an error, a pre-flight or health stop, or SIGTERM
        rec["status"] = "error"
        rec["errors"].append(f"{type(exc).__name__}: {exc}"[:500])
        rec["traceback"] = traceback.format_exc()[-3000:]
        code = 3 if isinstance(exc, (PreflightError, HealthError)) else 1
    rec["finished_at"] = now()
    save()
    print(json.dumps({"layout": args.layout, "status": rec["status"], "arms": list(rec["arms"]),
                      "cut": [c["step"] for c in rec["cut"]], "step_errors": [e["step"] for e in rec["step_errors"]],
                      "errors": rec["errors"][:2]}))
    sys.exit(code)


if __name__ == "__main__":
    main()
