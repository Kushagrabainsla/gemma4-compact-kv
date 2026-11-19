"""The compact K=V row of the Gemma 4 global layers as an engine stores it (E-57; the views layout of E-52).

One row per KV head and token holds [K_rot (128) | V_unrot (384) | V_rot (128)] = 640 values. The K view is values 0
to 511 ([K_rot | V_unrot]) and the V view is values 128 to 639 ([V_unrot | V_rot]). A stock attention kernel reads
the two strided views as its K and V caches. The query takes the K-view order, and its unrotated part carries w (the
k_norm weight), because dot(q_u, w V_u) = dot(q_u w, V_u). RoPE pairs dim i with dim i + 256, so the query
permutation comes after RoPE (a gather at run time). The attention output comes in the V-view order
[unrotated | rotated]; `unpermute_output` or a permutation of the o_proj input columns (`o_proj_columns`) restores the
model order.

The two views overlap on V_unrot. Writing key' = [K_rot | V_unrot] into the K view and value' = [V_unrot | V_rot] into
the V view puts the same values into the shared region, so the stock cache-write op can fill the row.

The page plan is B3 of gate G1: the global block is the largest multiple of 16 tokens whose compact page fits in the
sliding page, and the vLLM page unifier pads the compact page to the sliding page (31B and 26B-A4B: 48 tokens; 12B: 96).
"""

from __future__ import annotations

import math

import torch

HEAD = 512                   # the global head size of Gemma 4 12B, 26B-A4B and 31B
ROTARY = 0.25                # p-RoPE: 128 of the 512 dims rotate


def rot_dims(head: int = HEAD, rotary: float = ROTARY) -> tuple[list[int], list[int]]:
    """The rotated and unrotated dims (neox halves): for head 512, 0..63 and 256..319 rotate (as splice_e55)."""
    half = int(head * rotary) // 2
    rot = list(range(half)) + list(range(head // 2, head // 2 + half))
    rs = set(rot)
    return rot, [d for d in range(head) if d not in rs]


ROT, UNROT = rot_dims()
NR, NU = len(ROT), len(UNROT)            # 128 and 384
ROW = NR + NU + NR                       # 640
K_ORDER = ROT + UNROT                    # the dims of the K view, in row order
V_ORDER = UNROT + ROT                    # the dims of the V view, in row order


def build_row(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[T, H, 512] k and v (model order) -> the compact row [T, H, 640]."""
    return torch.cat([k[..., ROT], v[..., UNROT], v[..., ROT]], dim=-1)


def key_view(row: torch.Tensor) -> torch.Tensor:
    return row[..., :HEAD]


def value_view(row: torch.Tensor) -> torch.Tensor:
    return row[..., NR:]


def write_operands(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The key and value that a stock cache-write op stores into the K and V views: [K_rot | V_unrot] and
    [V_unrot | V_rot]. Both carry the same V_unrot values for the shared region."""
    return torch.cat([k[..., ROT], v[..., UNROT]], dim=-1), torch.cat([v[..., UNROT], v[..., ROT]], dim=-1)


def _w_unrot(w, ref: torch.Tensor):
    """w as a scalar or as a tensor of the unrotated dims (a 512-value k_norm weight is indexed; 384 values are used
    as they are)."""
    if isinstance(w, torch.Tensor):
        w = w.to(device=ref.device, dtype=torch.promote_types(ref.dtype, torch.float32))
        return w[UNROT] if w.shape[-1] == HEAD else w
    return w


def prep_query(q: torch.Tensor, w) -> torch.Tensor:
    """[T, Hq, 512] q (model order, after RoPE) -> the K-view order with w on the unrotated part, rounded once to the
    dtype of q (the same rounding as the E-55 and E-56 BF16C arm). w: a scalar, or a per-dim tensor (exact for any w,
    because RoPE is the identity on the unrotated dims)."""
    acc = torch.promote_types(q.dtype, torch.float32)   # float32 for BF16 (as E-55 and E-56), float64 for float64
    return torch.cat([q[..., ROT], (q[..., UNROT].to(acc) * _w_unrot(w, q)).to(q.dtype)], dim=-1)


def gross_query(q: torch.Tensor) -> torch.Tensor:
    """The GROSS control (E-57): the K-view order without w. The unrotated scores grow by 1 / w (about 16)."""
    return torch.cat([q[..., ROT], q[..., UNROT]], dim=-1)


def naive_operands(k: torch.Tensor, v: torch.Tensor, w) -> tuple[torch.Tensor, torch.Tensor]:
    """The NAIVE control (E-57): the cache keeps w * V as the key on every dim (the 50% "K equals V" design), so the
    K view holds w V_rot (no rotation) in place of K_rot. The V view is unchanged."""
    acc = torch.promote_types(v.dtype, torch.float32)
    wr = w.to(device=v.device, dtype=acc)[ROT] if isinstance(w, torch.Tensor) and w.shape[-1] == HEAD else w
    k_rot = (v[..., ROT].to(acc) * wr).to(v.dtype)
    return torch.cat([k_rot, v[..., UNROT]], dim=-1), torch.cat([v[..., UNROT], v[..., ROT]], dim=-1)


def unpermute_output(o: torch.Tensor) -> torch.Tensor:
    """[T, Hq, 512] output in the V-view order -> the model order."""
    out = torch.empty_like(o)
    out[..., UNROT] = o[..., :NU]
    out[..., ROT] = o[..., NU:]
    return out


def o_proj_columns(num_heads: int, head: int = HEAD) -> list[int]:
    """The input-column order of o_proj that takes the V-view output of every head: W_o[:, cols] @ o_vview equals
    W_o @ o_model."""
    return [h * head + d for h in range(num_heads) for d in V_ORDER]


def page_bytes(block_tokens: int, kv_heads: int, values_per_head: int, dtype_bytes: int) -> int:
    """The bytes of one KV page: tokens x heads x values per head and token x bytes per value."""
    return block_tokens * kv_heads * values_per_head * dtype_bytes


def compact_block(sliding_page: int, kv_heads: int, dtype_bytes: int, multiple: int = 16) -> int:
    """B3 (gate G1): the largest multiple of `multiple` tokens whose compact page fits in the sliding page."""
    per_token = kv_heads * ROW * dtype_bytes
    b = multiple * (sliding_page // (multiple * per_token))
    if b <= 0:
        raise ValueError(f"no compact block fits a sliding page of {sliding_page} B")
    return b


def sliding_blocks(window: int, block: int, in_flight: int = 1) -> int:
    """The blocks that one sliding group holds for a request: the formula of the vLLM sliding spec,
    ceil((window - 1 + in_flight) / block) + 1 (decode: in_flight 1 or 2; the startup bound: 2 x the batched tokens)."""
    return math.ceil((window - 1 + in_flight) / block) + 1


def blocks_per_request(tokens: int, global_block: int, sliding_block: int, sliding_groups: int, window: int = 1_024,
                       in_flight: int = 1) -> int:
    """The pool blocks of one request: ceil(tokens / global block) + the sliding groups' blocks (one pool block holds
    one page of every layer of a group, so a group counts once)."""
    return math.ceil(tokens / global_block) + sliding_groups * min(sliding_blocks(window, sliding_block, in_flight),
                                                                  math.ceil(tokens / sliding_block) + 1)


def request_kv_gib(tokens: int, compact: bool, dtype_bytes: int = 2, global_layers: int = 10, global_heads: int = 4,
                   sliding_layers: int = 50, sliding_heads: int = 16, sliding_head: int = 256,
                   window: int = 1_024) -> float:
    """The ideal KV bytes of one request in GiB for Gemma 4 31B (no page padding): the global layers keep every token;
    the sliding layers keep the last `window` tokens (the hybrid manager frees the rest)."""
    per_head = ROW if compact else 2 * HEAD
    glob = global_layers * global_heads * per_head * dtype_bytes * tokens
    slide = sliding_layers * sliding_heads * 2 * sliding_head * dtype_bytes * min(tokens, window)
    return (glob + slide) / 2**30
