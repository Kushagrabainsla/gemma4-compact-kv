"""E-57: the compact K=V cache inside vLLM 0.33.0rc1.dev229+g3966924d5 (the V2 model runner, TRITON_ATTN, eager, in
process), as monkeypatches that the run installs before it builds the engine (README Section 4;
`dossier/DESIGN_E-57_VLLM_CODE_MAP_2026-11-04.md`).

Three parts:
- The KV cache spec. `GPUModelRunner.get_kv_cache_spec` returns, for each global layer of a K=V model, the stock
  `FullAttentionSpec` with `state_content_bytes` = 640 values and the B3 block of gate G1 (`compact_block`). The stock
  page unifier then pads the compact page to the sliding page. No allocator, unifier or scheduler code changes.
- The views. The Triton backend splits a cache row at `head_size`, so a 640-value row gives a correct K view (values 0
  to 511) and a 128-value "value cache". The wrappers of `triton_attn.unified_attention` and
  `triton_attn.triton_reshape_and_cache_flash` replace that value cache with the V view (the K view shifted by 128
  values). Stock rows pass through.
- The hooks. `ViewsSplice` changes the q, k and v of each global `Attention` call and the output: VIEWS (the compact
  arithmetic: the query in the K-view order with w, the two write operands, the output back to the model order), and
  the controls NAIVE and GROSS. In the stock engine VIEWS is EMULV, the bit-for-bit twin of the compact engine.

The vLLM imports are inside `install` and the record functions, so the Mac tests import this module without vLLM.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_model")]
import compact_layout as cl  # noqa: E402

MODES = ("OFF", "VIEWS", "NAIVE", "GROSS")
GAP_ROWS = 16                       # the query rows of the attention-level check (as E-55 V5)
STATE: dict = {"installed": None, "spec": None, "spec_calls": 0, "fix_reads": 0, "fix_writes": 0,
               "pass_reads": 0, "pass_writes": 0, "alloc_fail": 0, "preemptions": 0, "sched": None}


def schedule_digest(seq: list) -> dict:
    """The schedule fingerprint of one phase (A1, R-82 F4): the sorted scheduled-token counts of each non-empty step."""
    blob = json.dumps(seq, separators=(",", ":")).encode()
    return {"steps": len(seq), "tokens": int(sum(sum(s) for s in seq)), "sha": hashlib.sha256(blob).hexdigest()[:16]}


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def compact_specs(specs: dict, k_eq_v: bool, full_cls, sliding_cls) -> tuple[dict, dict]:
    """The compact specs of a K=V model: each `full_cls` spec (exact type) with head sizes 512 and no padding gets the
    B3 block and 640 values per head and token; the rest stays. Returns the new dict and a record."""
    info = {"applied": False, "layers": [], "k_eq_v": bool(k_eq_v)}
    sliding = [s for s in specs.values() if isinstance(s, sliding_cls)]
    if not k_eq_v or not sliding:
        info["reason"] = "not a K=V model" if not k_eq_v else "no sliding layer"
        return specs, info
    max_page = max(int(s.page_size_bytes) for s in sliding)
    new = dict(specs)
    for name, s in specs.items():
        if (type(s) is not full_cls or s.head_size != cl.HEAD or s.head_size_v != cl.HEAD
                or getattr(s, "page_size_padded", None) is not None
                or getattr(s, "state_content_bytes", None) is not None):
            continue
        nb = dtype_bytes(s.dtype)
        block = cl.compact_block(max_page, s.num_kv_heads, nb)
        new[name] = replace(s, block_size=block, state_content_bytes=cl.ROW * nb)
        info["layers"].append(name)
        info.update(block=block, kv_heads=int(s.num_kv_heads), dtype_bytes=nb, stock_block=int(s.block_size),
                    sliding_page=max_page, compact_page=block * int(s.num_kv_heads) * cl.ROW * nb)
    info["applied"] = bool(info["layers"])
    return new, info


def fix_value_cache(key_cache: torch.Tensor, value_cache: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """If the split of a compact row gave the 128-value tail as the value cache, return the V view: the K view
    shifted by 128 values, with the K view's shape and strides. Else return the value cache unchanged."""
    if (key_cache.shape[-1] == cl.HEAD and value_cache.shape[-1] == cl.NR and key_cache.stride(-1) == 1
            and value_cache.stride(-1) == 1 and value_cache.dtype == key_cache.dtype
            and value_cache.storage_offset() == key_cache.storage_offset() + cl.HEAD
            and tuple(value_cache.stride()[:-1]) == tuple(key_cache.stride()[:-1])):
        return key_cache.as_strided(key_cache.shape, key_cache.stride(), key_cache.storage_offset() + cl.NR), True
    return value_cache, False


def install(layout: str) -> dict:
    """Install the patches in this process before the engine starts. Both layouts wrap the two Triton entry points
    (stock rows pass through, so both engines pay the same wrapper cost); "compact" also wraps the spec method."""
    if layout not in ("stock", "compact"):
        raise ValueError(f"unknown layout {layout}")
    if STATE["installed"] is not None:
        raise RuntimeError("install() runs once per process")
    import vllm.v1.attention.backends.triton_attn as ta
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.worker.gpu import model_runner as mr

    orig_ua, orig_write = ta.unified_attention, ta.triton_reshape_and_cache_flash

    def unified_attention(*args, **kwargs):
        if "k" in kwargs and "v" in kwargs:
            v, fixed = fix_value_cache(kwargs["k"], kwargs["v"])
            if fixed:
                kwargs["v"] = v
        else:
            args = list(args)
            v, fixed = fix_value_cache(args[1], args[2])
            args[2] = v
        STATE["fix_reads" if fixed else "pass_reads"] += 1
        return orig_ua(*args, **kwargs)

    def triton_reshape_and_cache_flash(key, value, key_cache, value_cache, *rest, **kwargs):
        value_cache, fixed = fix_value_cache(key_cache, value_cache)
        STATE["fix_writes" if fixed else "pass_writes"] += 1
        return orig_write(key, value, key_cache, value_cache, *rest, **kwargs)

    ta.unified_attention = unified_attention
    ta.triton_reshape_and_cache_flash = triton_reshape_and_cache_flash
    orig_alloc, orig_preempt = KVCacheManager.allocate_slots, Scheduler._preempt_request

    def allocate_slots(self, *args, **kwargs):     # V8: a None result means the pool limited a step
        out = orig_alloc(self, *args, **kwargs)
        if out is None:
            STATE["alloc_fail"] += 1
        return out

    def _preempt_request(self, *args, **kwargs):
        STATE["preemptions"] += 1
        return orig_preempt(self, *args, **kwargs)

    orig_schedule = Scheduler.schedule

    def schedule(self, *args, **kwargs):           # A1 (R-82 F4): the schedule fingerprint while a phase records it
        out = orig_schedule(self, *args, **kwargs)
        if STATE["sched"] is not None and out.num_scheduled_tokens:
            STATE["sched"].append(sorted(int(n) for n in out.num_scheduled_tokens.values()))
        return out

    KVCacheManager.allocate_slots = allocate_slots
    Scheduler._preempt_request = _preempt_request
    Scheduler.schedule = schedule
    if layout == "compact":
        orig_spec = mr.GPUModelRunner.get_kv_cache_spec

        def get_kv_cache_spec(self):
            specs = orig_spec(self)
            cfg = self.vllm_config.model_config.hf_text_config
            new, info = compact_specs(specs, bool(getattr(cfg, "attention_k_eq_v", False)), FullAttentionSpec,
                                      SlidingWindowSpec)
            STATE["spec"] = info
            STATE["spec_calls"] += 1
            return new

        mr.GPUModelRunner.get_kv_cache_spec = get_kv_cache_spec
    STATE["installed"] = layout
    return dict(STATE)


class Indexer:
    """The dim permutations as device index tensors (one copy per device): list indexing would copy the index to the
    GPU on every call. The functions give the same values as `compact_layout` (a Mac test checks this)."""

    def __init__(self):
        self.cache: dict = {}

    def get(self, device) -> dict:
        key = str(device)
        if key not in self.cache:
            inv = [0] * cl.HEAD
            for pos, d in enumerate(cl.V_ORDER):
                inv[d] = pos
            mk = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)  # noqa: E731
            self.cache[key] = {"k": mk(cl.K_ORDER), "v": mk(cl.V_ORDER), "rot": mk(cl.ROT), "unrot": mk(cl.UNROT),
                               "inv": mk(inv)}
        return self.cache[key]

    def prep_query(self, q: torch.Tensor, w_u) -> torch.Tensor:
        ix = self.get(q.device)
        x = q.index_select(-1, ix["k"])
        out = torch.empty_like(x)
        out[..., :cl.NR] = x[..., :cl.NR]
        out[..., cl.NR:] = (x[..., cl.NR:].to(torch.promote_types(q.dtype, torch.float32)) * w_u).to(q.dtype)
        return out

    def gross_query(self, q: torch.Tensor) -> torch.Tensor:
        return q.index_select(-1, self.get(q.device)["k"])

    def write_operands(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.get(k.device)
        return (torch.cat([k.index_select(-1, ix["rot"]), v.index_select(-1, ix["unrot"])], dim=-1),
                v.index_select(-1, ix["v"]))

    def naive_operands(self, k: torch.Tensor, v: torch.Tensor, w_r) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.get(v.device)
        acc = torch.promote_types(v.dtype, torch.float32)
        k_rot = (v.index_select(-1, ix["rot"]).to(acc) * w_r).to(v.dtype)
        return torch.cat([k_rot, v.index_select(-1, ix["unrot"])], dim=-1), v.index_select(-1, ix["v"])

    def unpermute(self, o: torch.Tensor) -> torch.Tensor:
        return o.index_select(-1, self.get(o.device)["inv"])


class ViewsSplice:
    """The hooks of the global layers. `set_mode` selects OFF (stock arithmetic), VIEWS (the compact arithmetic),
    NAIVE or GROSS. `new_doc` marks the next call of each layer as the first chunk of a prompt: for the first such
    chunk of each mode with at least GAP_ROWS tokens, the output hook stores the relative gap between the engine
    output (model order) and two float32 references: of the changed q, k and v ("gaps") and of the stock q, k and v
    ("gaps_id"); "seps" is the gap between the two references. `counts` holds the tokens per mode and layer."""

    def __init__(self):
        self.mode = "OFF"
        self.ix = Indexer()
        self.modules: dict[int, object] = {}
        self.handles: dict[int, tuple] = {}
        self.w: dict[int, torch.Tensor] = {}
        self.wu: dict[int, torch.Tensor] = {}       # w on the unrotated dims (384), a device tensor (A1, R-82 B4)
        self.wr: dict[int, torch.Tensor] = {}       # w on the rotated dims (128)
        self.heads: dict[int, tuple[int, int, int]] = {}
        self.fresh: set[int] = set()
        self.reset_evidence()

    def reset_evidence(self) -> None:
        self.counts: dict[str, int] = {}
        self.gaps: dict[str, dict[int, float]] = {}
        self.gaps_id: dict[str, dict[int, float]] = {}
        self.seps: dict[str, dict[int, float]] = {}
        self.checked: set[tuple[str, int]] = set()
        self.ref: dict[int, tuple] = {}
        self.errors: list[str] = []

    def evidence(self) -> dict:
        js = lambda d: {m: {str(k): x for k, x in g.items()} for m, g in d.items()}  # noqa: E731
        return {"counts": dict(self.counts), "gaps": js(self.gaps), "gaps_id": js(self.gaps_id),
                "seps": js(self.seps), "hook_errors": list(self.errors)}

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode

    def new_doc(self) -> None:
        self.fresh = set(self.modules)

    def attach(self, layer: int, module, w: torch.Tensor, heads: tuple[int, int, int]) -> None:
        """Hook one global-layer Attention module. w: the k_norm weight (512 values)."""
        self.modules[layer], self.heads[layer] = module, heads
        wf = w.detach().to(torch.float32)
        ix = self.ix.get(wf.device)
        self.w[layer] = wf
        self.wu[layer], self.wr[layer] = wf.index_select(0, ix["unrot"]), wf.index_select(0, ix["rot"])
        self.handles[layer] = (module.register_forward_pre_hook(self._pre(layer), with_kwargs=True),
                               module.register_forward_hook(self._post(layer)))

    def detach(self) -> None:
        for pre, post in self.handles.values():
            pre.remove()
            post.remove()
        self.handles.clear()

    def transform(self, mode: str, layer: int, q3, k3, v3):
        """[T, H, 512] q, k, v (model order) -> what the cache and the kernel get. The w slices are device tensors made
        in `attach`: a Python-list index of a CUDA tensor copies the index to the GPU and syncs the stream (R-82 B4)."""
        w_u, w_r = self.wu[layer], self.wr[layer]
        if w_u.device != q3.device:
            self.wu[layer], self.wr[layer] = w_u, w_r = w_u.to(q3.device), w_r.to(q3.device)
        if mode == "VIEWS":
            return (self.ix.prep_query(q3, w_u),) + self.ix.write_operands(k3, v3)
        if mode == "NAIVE":
            return (self.ix.prep_query(q3, w_u),) + self.ix.naive_operands(k3, v3, w_r)
        if mode == "GROSS":
            return (self.ix.gross_query(q3),) + self.ix.write_operands(k3, v3)
        raise ValueError(mode)

    def _pre(self, layer: int):
        import splice_e55 as sp
        hq, hk, head = self.heads[layer]

        def pre(module, args, kwargs):
            try:
                q, k, v, put = sp.qkv_of(args, kwargs)
                if k is None or v is None:
                    return None
                t = k.shape[0]
                self.counts[f"{self.mode}|{layer}"] = self.counts.get(f"{self.mode}|{layer}", 0) + t
                fresh = layer in self.fresh
                self.fresh.discard(layer)
                if self.mode == "OFF" and (not fresh or ("OFF", layer) in self.checked or t < GAP_ROWS):
                    return None
                q3, k3, v3 = q.view(t, hq, head), k.view(t, hk, head), v.view(t, hk, head)
                if self.mode == "OFF":
                    q2, k2, v2 = q3, k3, v3
                else:
                    q2, k2, v2 = self.transform(self.mode, layer, q3, k3, v3)
                if fresh and t >= GAP_ROWS and (self.mode, layer) not in self.checked:
                    self.checked.add((self.mode, layer))
                    self.ref[layer] = (self.mode, q2[-GAP_ROWS:].clone(), k2.clone(), v2.clone(),
                                       q3[-GAP_ROWS:].clone(), k3.clone(), v3.clone(),
                                       float(getattr(getattr(module, "impl", None), "scale", 1.0)))
                if self.mode == "OFF":
                    return None
                return put(q2.reshape(q.shape), k2.reshape(k.shape), v2.reshape(v.shape))
            except Exception as exc:  # recorded; V4 fails
                self.errors.append(f"pre {self.mode} layer {layer}: {type(exc).__name__}: {exc}"[:300])
                raise
        return pre

    def _post(self, layer: int):
        import splice_e55 as sp
        hq, _, head = self.heads[layer]

        def post(module, args, output):
            o = output[0] if isinstance(output, (tuple, list)) else output
            mode = self.mode
            new = None
            if mode != "OFF":
                t = o.shape[0]
                new = self.ix.unpermute(o.view(t, hq, head)).reshape(o.shape)
            ref = self.ref.pop(layer, None)
            if ref is not None:
                try:
                    m, qr, k2, v2, q0, k0, v0, scale = ref
                    t = k2.shape[0]
                    qpos = torch.arange(t - GAP_ROWS, t, device=k2.device)
                    got = (o if new is None else new).reshape(t, hq, head)[-GAP_ROWS:].float()
                    want = sp.attend32(qr, k2, v2, qpos, scale)
                    if m != "OFF":
                        want = want.index_select(-1, self.ix.get(want.device)["inv"])
                    want0 = sp.attend32(q0, k0, v0, qpos, scale)
                    self.gaps.setdefault(m, {})[layer] = sp.rel(got, want)
                    self.gaps_id.setdefault(m, {})[layer] = sp.rel(got, want0)
                    self.seps.setdefault(m, {})[layer] = sp.rel(want, want0)
                except Exception as exc:  # recorded; V4 fails
                    self.errors.append(f"post {mode} layer {layer}: {type(exc).__name__}: {exc}"[:300])
            if new is None:
                return None
            return (new,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else new
        return post


def register(model, splice: ViewsSplice) -> dict:
    """Hook the global layers (`run_e55.find_global`) with their k_norm weights. Returns the hooked layers and the
    problems (V1)."""
    import run_e55 as r55
    rec, problems = [], []
    for name, mod in model.named_modules():         # A1 (R-82 F8): a shared global layer would read the compact
        if (type(mod).__name__ == "Gemma4Attention" and not getattr(mod, "is_sliding", True)   # cache unhooked
                and getattr(mod, "is_kv_shared_layer", False)):
            problems.append(f"{name}: a KV-shared global layer")
    for layer, m, parent in r55.find_global(model):
        if getattr(m, "kv_sharing_target_layer_name", None):
            problems.append(f"layer {layer}: the Attention module shares the KV cache of another layer")
            continue
        if int(m.head_size) != cl.HEAD:
            problems.append(f"layer {layer}: head size {m.head_size}, not {cl.HEAD}")
            continue
        wt = getattr(getattr(parent, "k_norm", None), "weight", None)
        if not isinstance(wt, torch.Tensor) or wt.numel() != cl.HEAD:
            problems.append(f"layer {layer}: no 512-value k_norm weight")
            continue
        heads = (int(m.num_heads), int(m.num_kv_heads), int(m.head_size))
        splice.attach(layer, m, wt, heads)
        wu = wt.detach().float().cpu()[cl.UNROT]
        rec.append({"layer": layer, "num_heads": heads[0], "num_kv_heads": heads[1], "head_size": heads[2],
                    "w_unrot_unique": int(torch.unique(wu).numel()), "w_unrot_min": float(wu.min()),
                    "w_unrot_max": float(wu.max()), "impl": type(getattr(m, "impl", None)).__name__})
    if not rec:
        problems.append("no hooked global layer")
    return {"hooked": rec, "problems": problems}


def engine_core(llm):
    """The in-process EngineCore (VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
    return llm.llm_engine.engine_core.engine_core


def kv_record(llm) -> dict:
    """The KV cache plan that the engine built: the pool blocks, the groups and the free blocks (V1, V3)."""
    sch = engine_core(llm).scheduler
    cfg = sch.kv_cache_config
    groups = []
    for g in cfg.kv_cache_groups:
        s = g.kv_cache_spec
        groups.append({"type": type(s).__name__, "layers": len(g.layer_names), "first_layer": g.layer_names[0],
                       "block_size": int(s.block_size), "page_bytes": int(s.page_size_bytes),
                       "real_page_bytes": int(getattr(s, "real_page_size_bytes", s.page_size_bytes)),
                       "page_size_padded": getattr(s, "page_size_padded", None),
                       "state_content_bytes": getattr(s, "state_content_bytes", None),
                       "num_kv_heads": getattr(s, "num_kv_heads", None), "head_size": getattr(s, "head_size", None),
                       "sliding_window": getattr(s, "sliding_window", None)})
    pool = sch.kv_cache_manager.block_pool
    return {"num_blocks": int(cfg.num_blocks), "pool_blocks": int(pool.num_gpu_blocks),
            "free_blocks": int(pool.get_num_free_blocks()), "groups": groups,
            "bytes_per_block": max((gr["page_bytes"] * gr["layers"] for gr in groups), default=None),
            "layout": cfg.kv_cache_layout, "tensors": len(cfg.kv_cache_tensors)}


def tensor_record(model) -> dict:
    """The bound KV cache tensor of the first global and the first sliding Attention module (V1: shape and strides)."""
    import run_e55 as r55
    out = {}
    glob = r55.find_global(model)
    if glob:
        layer, m, _ = glob[0]
        kv = m.kv_cache[0] if isinstance(m.kv_cache, (list, tuple)) else m.kv_cache
        out["global"] = {"layer": layer, "shape": list(kv.shape), "stride": list(kv.stride()), "dtype": str(kv.dtype)}
    for name, m in model.named_modules():
        if type(m).__name__ == "Attention" and getattr(m, "sliding_window", None) is not None:
            kv = m.kv_cache[0] if isinstance(m.kv_cache, (list, tuple)) else m.kv_cache
            out["sliding"] = {"name": name, "shape": list(kv.shape), "stride": list(kv.stride()),
                              "dtype": str(kv.dtype)}
            break
    return out


def free_blocks(llm) -> int:
    return int(engine_core(llm).scheduler.kv_cache_manager.block_pool.get_num_free_blocks())


def running_state(llm) -> dict:
    """The scheduler state for the capacity probe: running requests, those past their prompt, waiting requests, and
    the preemptions and failed allocations so far (the install() counters)."""
    sch = engine_core(llm).scheduler
    run = list(sch.running)
    decoding = sum(1 for r in run if r.num_computed_tokens >= r.num_prompt_tokens)
    counts = getattr(sch, "get_request_counts", None)
    waiting = counts()[1] if counts else len(sch.waiting)   # includes the waiting requests that hold blocks
    return {"running": len(run), "decoding": decoding, "waiting": int(waiting),
            "preemptions": STATE["preemptions"], "alloc_fail": STATE["alloc_fail"]}


class StubAttention(torch.nn.Module):
    """A stand-in for the vLLM `Attention` module on the Mac: forward(q, k, v) with [T, H x 512] inputs, causal GQA
    attention of the call's own tokens in BF16 (the hooks see the engine's call shapes; no paged cache)."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_size: int = cl.HEAD):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_size = num_heads, num_kv_heads, head_size

    def forward(self, q, k, v):
        t, hq, hk, d = q.shape[0], self.num_heads, self.num_kv_heads, self.head_size
        q3, k3, v3 = q.view(t, hq, d), k.view(t, hk, d), v.view(t, hk, d)
        g = hq // hk
        s = torch.einsum("thd,lhd->htl", q3.float(), k3.float().repeat_interleave(g, dim=1))
        s = s.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1)[None], float("-inf"))
        o = torch.einsum("htl,lhd->thd", torch.softmax(s, -1), v3.float().repeat_interleave(g, dim=1))
        return o.to(q.dtype).reshape(t, -1)


def stub_qkv(t: int, heads: tuple[int, int, int], w: float, seed: int):
    """Random q, k and v with the model's structure: K_unrot = bf16(w V_unrot), the rotated dims differ."""
    hq, hk, d = heads
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(t, hk, d, generator=g).to(torch.bfloat16)
    k = (v.float() * w).to(torch.bfloat16)
    k[..., cl.ROT] = (torch.randn(t, hk, cl.NR, generator=g) * 0.5).to(torch.bfloat16)
    q = (torch.randn(t, hq, d, generator=g) * 8).to(torch.bfloat16)
    return q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)
