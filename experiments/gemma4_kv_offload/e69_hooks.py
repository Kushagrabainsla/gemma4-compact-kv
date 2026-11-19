"""E-69 (README Sections 5 and 6, as amended by A1): the in-process hooks of the engine test. No vLLM import at module
level, so the Mac tests import this file without vLLM.

- `block_hashes`: one int64 hash per block id over the content bytes of every layer of a KV cache group (the manager
  views that the connector copies, so the page padding is not read).
- `ReloadCheck` (V-RELOAD): the (block id, hash) pairs of a request's completed blocks at its preemption (A1 B3: only
  blocks that end at or below num_computed_tokens - num_in_flight_tokens), and again after its CPU load. A logical
  block inside the loaded range is compared only if its block id changed (A1 B3: a stale block with the same id would
  hash equal even if the load did nothing).
- `EventLog` (A1 B4): one record per connector lookup (`PrefixCacheStats.record`), for the per-event decode coverage.
- `StatsSum` (V3 and the secondary metrics): the sum over steps of the stats of `Scheduler.make_stats`.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

import torch

NULL_BLOCK = 0                      # vLLM block id 0 is the null block (sliding blocks out of the window)
MAX_CHECKS = 24                     # A1 F1: the first 24 reloads of an arm are checked
MIN_DECODE = 512                    # A1 B2: a decode event has at least 512 generated tokens at the preemption
_W: dict = {}


def _weights(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _W:
        _W[key] = (torch.arange(n, device=device, dtype=torch.int64) % 1021) + 1
    return _W[key]


def block_hashes(layers: list[torch.Tensor], block_ids: list[int]) -> dict[int, int]:
    """For the given block ids (null blocks skipped), the sum over layers of a position-weighted int32 sum of the block
    content. Each layer tensor is indexed by block on dim 0 (a view of any dtype)."""
    ids = sorted({b for b in block_ids if b != NULL_BLOCK})
    if not ids or not layers:
        return {}
    total = None
    for t in layers:
        idx = torch.tensor(ids, device=t.device, dtype=torch.long)
        raw = t.index_select(0, idx).contiguous().view(len(ids), -1).view(torch.uint8)
        if raw.shape[1] % 4:
            raw = torch.nn.functional.pad(raw, (0, 4 - raw.shape[1] % 4))
        x = raw.view(torch.int32).to(torch.int64)
        h = (x * _weights(x.shape[1], x.device)).sum(dim=1)
        total = h if total is None else total + h
    return dict(zip(ids, (int(v) for v in total.cpu())))


@dataclass
class Snapshot:
    """One request at one time: its token count and, per group, {logical block index: (block id, hash)}."""
    tokens: int
    groups: dict[int, dict[int, tuple[int, int]]] = field(default_factory=dict)


def snapshot(layers_by_group: dict[int, list[torch.Tensor]], block_ids_by_group: dict[int, list[int]], tokens: int,
             block_size: dict[int, int], limit: int | None = None) -> Snapshot:
    """The completed blocks of a request: logical blocks that end at or below `limit` tokens (all if None)."""
    s = Snapshot(tokens=tokens)
    for g, ids in block_ids_by_group.items():
        keep = {i: b for i, b in enumerate(ids)
                if b != NULL_BLOCK and (limit is None or (i + 1) * block_size[g] <= limit)}
        hashes = block_hashes(layers_by_group.get(g, []), list(keep.values()))
        s.groups[g] = {i: (b, hashes[b]) for i, b in keep.items() if b in hashes}
    return s


def compare(before: Snapshot, after: Snapshot, hit: int, block_size: dict[int, int], sliding: set[int],
            recomputed: tuple[int, int] | None = None, first: dict | None = None) -> dict:
    """V-RELOAD for one reload: the logical blocks fully inside [0, hit), with a hash in both snapshots and a new id.
    A3: a block fully inside `recomputed` = [previous hit, previous n_pre) of the same request was recomputed by the
    prefill path after the previous reload, while the CPU tier keeps its first (decode) copy (`prepare_store` stores
    only new keys).
    A3 as changed by R-112 F1: such a block passes if it equals the GPU copy at this preemption or the FIRST hash seen
    for that block of the request (`first`, the decode version that the CPU tier keeps); a third value, or a block
    without a first hash, is a mismatch (fail safe). Blocks that pass by the first version are counted apart."""
    out = {"hit": hit, "n_pre": before.tokens, "checked": 0, "same_id": 0, "mismatch": 0, "global": 0, "sliding": 0,
           "recomputed": 0, "recomputed_first_version": 0, "recomputed_third": 0, "bad": [], "bad_recomputed": []}
    lo, hi = recomputed if recomputed else (0, 0)
    first = first or {}
    for g, after_g in after.groups.items():
        bs = block_size[g]
        for i, (bid, h) in after_g.items():
            old = before.groups.get(g, {}).get(i)
            if (i + 1) * bs > hit or old is None:
                continue
            if old[0] == bid:
                out["same_id"] += 1
                continue
            if i * bs >= lo and (i + 1) * bs <= hi:
                out["recomputed"] += 1
                if old[1] == h:
                    pass
                elif first.get((g, i)) == h:
                    out["recomputed_first_version"] += 1
                    if len(out["bad_recomputed"]) < 64:
                        out["bad_recomputed"].append([g, i])
                else:                                       # a third value or no first hash: decides (fail safe)
                    out["recomputed_third"] += 1
                    out["mismatch"] += 1
                    if len(out["bad"]) < 64:
                        out["bad"].append([g, i, "in_range"])
                continue
            out["checked"] += 1
            out["sliding" if g in sliding else "global"] += 1
            if old[1] != h:
                out["mismatch"] += 1
                if len(out["bad"]) < 64:
                    out["bad"].append([g, i])
    out["valid"] = out["global"] > 0 and out["sliding"] > 0       # README Section 6: one global and one sliding block
    return out


class ReloadCheck:
    """Bookkeeping of V-RELOAD across preemptions and loads of one arm."""

    def __init__(self, max_checks: int = MAX_CHECKS):
        self.max_checks = max_checks
        self.last: dict[str, tuple[int, int]] = {}      # A3: (hit, n_pre) of the last load of each request
        self.n_pre: dict[str, int] = {}                 # A3: the length at the last preemption of each request
        self.first: dict[str, dict] = {}                # R-112 F1: the first hash seen of each (group, block index)
        self.pending: dict[str, Snapshot] = {}
        self.results: list[dict] = []
        self.preempt_seen = 0
        self.loads_seen = 0
        self.skipped = 0
        self.failed_loads = 0

    def want(self) -> bool:
        return len(self.results) + len(self.pending) < self.max_checks

    def on_preempt(self, req_id: str, snap_fn, n_pre: int | None = None) -> None:
        self.preempt_seen += 1
        if n_pre is not None:
            self.n_pre[req_id] = int(n_pre)               # A3: the length at every preemption, checked or not
        if req_id in self.pending or self.want():       # preempted again before its reload: keep the newest bytes
            s = snap_fn()
            self.pending[req_id] = s
            self.n_pre[req_id] = s.tokens
            seen = self.first.setdefault(req_id, {})
            for g, blocks in s.groups.items():
                for i, (_, h) in blocks.items():
                    seen.setdefault((g, i), h)

    def on_loaded(self, req_id: str, hit: int, snap_fn, block_size: dict[int, int], sliding: set[int],
                  failed: bool = False, corrupt_fn=None) -> None:
        self.loads_seen += 1
        before = self.pending.pop(req_id, None)
        if failed:
            self.failed_loads += 1
        prev = self.last.get(req_id)
        if req_id in self.n_pre:
            self.last[req_id] = (hit, self.n_pre[req_id])
        if before is None:
            self.skipped += 1
            return
        if corrupt_fn is not None:                       # A1 B3: the planted control (rehearsal arm C-CX only)
            corrupt_fn()
        self.results.append({"req": req_id, "failed_load": failed, "recomputed_range": prev,
                             **compare(before, snap_fn(), hit, block_size, sliding, prev, self.first.get(req_id))})

    def note_load(self, req_id: str, hit: int) -> None:
        """A3: a load that is not checked still moves the recomputed range of the request."""
        self.loads_seen += 1
        if req_id in self.n_pre:
            self.last[req_id] = (int(hit), self.n_pre[req_id])

    def summary(self) -> dict:
        valid = [r for r in self.results if r["valid"]]
        checked = sum(r["checked"] for r in valid)
        same = sum(r["same_id"] for r in valid)
        return {"checked_reloads": len(valid), "invalid_reloads": len(self.results) - len(valid),
                "mismatched_blocks": sum(r["mismatch"] for r in valid), "checked_blocks": checked,
                "same_id_blocks": same, "same_id_share": same / (same + checked) if same + checked else None,
                "recomputed_blocks": sum(r["recomputed"] for r in valid),
                "recomputed_first_version": sum(r["recomputed_first_version"] for r in valid),
                "recomputed_third": sum(r["recomputed_third"] for r in valid),
                # R-112 F3: the A1 count (every changed block decides), printed beside the A3 word
                "a1_strict_mismatch": sum(r["mismatch"] + r["recomputed_first_version"] for r in valid),
                "preempt_seen": self.preempt_seen, "loads_seen": self.loads_seen, "skipped_loads": self.skipped,
                "failed_loads": self.failed_loads, "pending_at_end": len(self.pending), "results": self.results}


class EventLog:
    """A1 B4: the connector lookups (`PrefixCacheStats.record`): tokens queried, hits, and the preempted flag."""

    def __init__(self):
        self.events: list[dict] = []

    def add(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        self.events.append({"tokens": int(num_tokens), "hits": int(num_hits), "preempted": bool(preempted)})

    def decode_coverage(self, prompt_len: int, min_decode: int = MIN_DECODE) -> dict:
        """A1 B2: per re-admission of a preempted request with at least `min_decode` generated tokens, the share of the
        generated part that was reloaded, c = (hits - P) / (tokens - P), clipped to [0, 1]."""
        pre = [e for e in self.events if e["preempted"]]
        dec = [e for e in pre if e["tokens"] - prompt_len >= min_decode]
        cs = [min(1.0, max(0.0, (e["hits"] - prompt_len) / (e["tokens"] - prompt_len))) for e in dec]
        return {"preempted_events": len(pre), "decode_events": len(dec), "median_c": st.median(cs) if cs else None,
                "min_c": min(cs) if cs else None, "max_c": max(cs) if cs else None,
                "zero_hit_events": sum(e["hits"] == 0 for e in pre),
                "hit_rate_tokens": (sum(e["hits"] for e in pre) / sum(e["tokens"] for e in pre)) if pre else None}


class StatsSum:
    """The sum over steps of the connector prefix-cache stats and the offload transfer stats (report and V3)."""

    def __init__(self):
        self.steps = 0
        self.none_steps = 0
        self.prefix = {k: 0 for k in ("requests", "queries", "hits", "preempted_requests", "preempted_queries",
                                      "preempted_hits")}
        self.kv: dict[str, float] = {}

    def add(self, stats) -> None:
        if stats is None:
            self.none_steps += 1
            return
        self.steps += 1
        p = getattr(stats, "connector_prefix_cache_stats", None)
        if p is not None:
            for k in self.prefix:
                self.prefix[k] += int(getattr(p, k, 0) or 0)
        kv = getattr(stats, "kv_connector_stats", None) or {}
        if isinstance(kv.get("data"), dict) and isinstance(kv.get("types"), dict):
            # A2: the offload payload (offloading/metrics.py): {"types": {name: kind}, "data": {name: {labels: v}}}.
            for name, by_label in kv["data"].items():
                kind = kv["types"].get(name)
                for val in by_label.values():
                    if kind == "histogram":
                        self.kv[name + "_count"] = self.kv.get(name + "_count", 0.0) + len(val)
                        self.kv[name + "_sum"] = self.kv.get(name + "_sum", 0.0) + _num(val)
                    elif kind == "gauge":
                        self.kv[name] = _num(val)
                    else:
                        self.kv[name] = self.kv.get(name, 0.0) + _num(val)
            return
        for name, val in kv.items():
            self.kv[name] = self.kv.get(name, 0.0) + _num(val)

    def summary(self) -> dict:
        lb = self.kv.get("vllm:kv_offload_load_bytes", 0.0)
        lt = self.kv.get("vllm:kv_offload_load_time", 0.0)
        return {"steps": self.steps, "none_steps": self.none_steps, "prefix": self.prefix, "kv": self.kv,
                "load_gb_per_s": lb / lt / 1e9 if lt > 0 else None}


def _num(val) -> float:
    """A stats value can be a number, a list of samples or a dict of fields."""
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (list, tuple)):
        return float(sum(_num(v) for v in val))
    if isinstance(val, dict):
        return float(sum(_num(v) for v in val.values()))
    return 0.0
