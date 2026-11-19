"""E-70 stage 1 (README Section 2): the NIXL patch for the compact K=V page, and the P and D hash hooks.

- `byte_view`: for a layer view whose block holds fewer content bytes than its block stride (the padded compact global
  page), a 2D uint8 view (num_blocks, block_stride) at the same address; any other tensor is returned unchanged.
- `install_nixl_patch`: wraps `NixlBaseConnectorWorker.register_kv_caches` so NIXL gets the byte views.
- `install_capture`: keeps the original manager views (for the hashes), with or without the patch.
- `install_p_hook`: on P, hashes each finished request's blocks in `request_finished_all_groups` (before the blocks are
  released for the transfer).
- `install_d_hook`: on D, hashes each request's blocks when its NIXL load ends (`_update_waiting_for_remote_kv`, the
  hit read before the call, as E-69).

No vLLM import at module level, so the Mac tests import this file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "gemma4_kv_offload")]
import e69_hooks as hk  # noqa: E402

STATE: dict = {"patched": [], "skipped": [], "kv": {}, "sched": None, "groups": None, "p": {}, "d": {},
               "errors": [], "installed": set()}


def byte_view(t: torch.Tensor) -> tuple[torch.Tensor, str]:
    """The NIXL view of one layer tensor and the reason. A block-major view with a contiguous block whose content is
    shorter than its block stride gets a (num_blocks, stride) uint8 view at the same address (the padding is at the end
    of each block); everything else is unchanged."""
    if t.ndim < 2:
        return t, "flat"
    stride_b = t.stride(0) * t.element_size()
    content_b = t[0].numel() * t.element_size()
    if not t[0].is_contiguous() or content_b >= stride_b:
        return t, "unchanged"
    storage = t.untyped_storage()
    offset_b = t.storage_offset() * t.element_size()
    if offset_b + t.shape[0] * stride_b > storage.nbytes():
        return t, "padding beyond the storage"
    v = torch.empty(0, dtype=torch.uint8, device=t.device).set_(storage, offset_b, (t.shape[0], stride_b),
                                                                (stride_b, 1))
    assert v.data_ptr() == t.data_ptr() and v[0].nbytes == stride_b and v[0].is_contiguous()
    return v, "byte view"


def patch_views(kv_caches: dict) -> dict:
    out = {}
    for name, t in kv_caches.items():
        v, why = byte_view(t) if isinstance(t, torch.Tensor) else (t, "not a tensor")
        out[name] = v
        if why == "byte view":
            STATE["patched"].append([name, list(t.shape), list(t.stride()), str(t.dtype), list(v.shape)])
        elif why not in ("unchanged",):
            STATE["skipped"].append([name, why])
    return out


def install_capture_and_patch(patch: bool) -> None:
    """Keep the original views for the hashes; with `patch`, give NIXL the byte views."""
    if "worker" in STATE["installed"]:
        return
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    orig = NixlBaseConnectorWorker.register_kv_caches

    def register_kv_caches(self, kv_caches, *args, **kwargs):
        STATE["kv"] = dict(kv_caches)
        return orig(self, patch_views(kv_caches) if patch else kv_caches, *args, **kwargs)

    NixlBaseConnectorWorker.register_kv_caches = register_kv_caches
    STATE["installed"].add("worker")


def group_info(sch) -> tuple[dict, dict, set]:
    layers, bsz, sliding = {}, {}, set()
    for g, grp in enumerate(sch.kv_cache_config.kv_cache_groups):
        layers[g] = [STATE["kv"][n] for n in grp.layer_names if n in STATE["kv"]]
        bsz[g] = int(grp.kv_cache_spec.block_size)
        if getattr(grp.kv_cache_spec, "sliding_window", None):
            sliding.add(g)
    return layers, bsz, sliding


def _hash_groups(block_ids_by_group, limit_tokens: int | None) -> dict:
    torch.cuda.synchronize()
    if STATE["groups"] is None:
        STATE["groups"] = group_info(STATE["sched"])
    layers, bsz, _ = STATE["groups"]
    snap = hk.snapshot(layers, {g: list(b) for g, b in enumerate(block_ids_by_group)}, 0, bsz, limit_tokens)
    return {g: {i: h for i, (_, h) in blocks.items()} for g, blocks in snap.groups.items()}


def install_p_hook() -> None:
    """P: the hashes of each finished request's blocks, before the blocks are released for the transfer."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlBaseConnector
    orig = NixlBaseConnector.request_finished_all_groups

    def request_finished_all_groups(self, request, block_ids, *args, **kwargs):
        try:
            if STATE["sched"] is not None:
                STATE["p"][request.request_id] = {"groups": _hash_groups(block_ids, None),
                                                  "num_computed": int(request.num_computed_tokens),
                                                  "prompt": int(request.num_prompt_tokens)}
        except Exception as e:  # noqa: BLE001 (a hook error is recorded, never raised into the engine)
            STATE["errors"].append(f"p: {type(e).__name__}: {e}"[:300])
        return orig(self, request, block_ids, *args, **kwargs)

    NixlBaseConnector.request_finished_all_groups = request_finished_all_groups


def install_d_hook() -> None:
    """D: the hashes of each request's blocks that lie fully inside the loaded range, when its NIXL load ends."""
    from vllm.v1.core.sched.scheduler import Scheduler
    orig = Scheduler._update_waiting_for_remote_kv

    def _update_waiting_for_remote_kv(self, request, *args, **kwargs):
        rid = request.request_id
        hit = int(request.num_computed_tokens)
        failed = rid in getattr(self, "failed_recving_kv_req_ids", set())
        out = orig(self, request, *args, **kwargs)
        try:
            if rid in STATE["d"]:              # stage 2: a later load is an offload reload, not the NIXL load
                return out
            ids = self.kv_cache_manager.get_block_ids(rid)
            STATE["d"][rid] = {"groups": _hash_groups(ids, hit), "hit": hit, "failed": failed,
                               "t_loaded": time.time()}                        # A1 (R-114 N10): report only
        except Exception as e:  # noqa: BLE001
            STATE["errors"].append(f"d: {type(e).__name__}: {e}"[:300])
        return out

    Scheduler._update_waiting_for_remote_kv = _update_waiting_for_remote_kv


def compare(p_groups: dict, d_groups: dict, sliding: set) -> dict:
    """The byte check of one request: every logical block that D hashed and P hashed too."""
    out = {"checked": 0, "mismatch": 0, "global": 0, "sliding": 0, "d_only": 0, "bad": []}
    for g, blocks in d_groups.items():
        pg = p_groups.get(g, p_groups.get(str(g), {}))
        for i, h in blocks.items():
            ph = pg.get(i, pg.get(str(i)))
            if ph is None:
                out["d_only"] += 1
                continue
            out["checked"] += 1
            out["sliding" if int(g) in sliding else "global"] += 1
            if ph != h:
                out["mismatch"] += 1
                if len(out["bad"]) < 32:
                    out["bad"].append([int(g), int(i)])
    return out
