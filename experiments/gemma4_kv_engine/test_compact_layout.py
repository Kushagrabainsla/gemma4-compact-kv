"""Tests of the compact K=V row (E-57). Run: uv run --no-project --with numpy --with torch --with pytest python -B -m
pytest -q -p no:cacheprovider experiments/gemma4_kv_engine/test_compact_layout.py"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_model")]
import compact_layout as cl  # noqa: E402


def attention(q, k, v, causal=True):
    """Reference GQA attention in the dtype of the inputs: q [T, Hq, D], k and v [T, Hk, D] -> [T, Hq, Dv]."""
    t, hq, _ = q.shape
    hk = k.shape[1]
    kk = k.repeat_interleave(hq // hk, dim=1)
    vv = v.repeat_interleave(hq // hk, dim=1)
    s = torch.einsum("thd,shd->hts", q, kk)
    if causal:
        s = s.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    return torch.einsum("hts,shd->thd", torch.softmax(s, dim=-1), vv)


def model_kv(t=24, hk=2, w=0.0615234375, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(t, hk, cl.HEAD, generator=g, dtype=torch.float64)
    k = (v * w).clone()
    k[..., cl.ROT] = torch.randn(t, hk, cl.NR, generator=g, dtype=torch.float64)   # the rotated dims differ
    return k.to(dtype), v.to(dtype)


def test_dims_and_orders():
    assert cl.ROT == list(range(64)) + list(range(256, 320)) and (cl.NR, cl.NU, cl.ROW) == (128, 384, 640)
    assert sorted(cl.K_ORDER) == list(range(512)) and sorted(cl.V_ORDER) == list(range(512))


def test_compact_attention_is_exact_in_float64():
    w = 0.0615234375
    k, v = model_kv(w=w)
    q = torch.randn(24, 8, cl.HEAD, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    ref = attention(q, k, v)
    row = cl.build_row(k, v)
    out = cl.unpermute_output(attention(cl.prep_query(q, w), cl.key_view(row), cl.value_view(row)))
    assert float((out - ref).norm() / ref.norm()) < 1e-12


def test_two_view_write_fills_the_row():
    k, v = model_kv()
    row = torch.zeros(k.shape[0], k.shape[1], cl.ROW, dtype=k.dtype)
    kw, vw = cl.write_operands(k, v)
    cl.key_view(row)[...] = kw
    cl.value_view(row)[...] = vw
    assert torch.equal(row, cl.build_row(k, v))
    assert torch.equal(kw[..., cl.NR:], vw[..., :cl.NU])          # the shared region gets the same values twice


def test_o_proj_fold_restores_the_model_order():
    g = torch.Generator().manual_seed(2)
    o = torch.randn(5, 4, cl.HEAD, generator=g, dtype=torch.float64)
    w_o = torch.randn(64, 4 * cl.HEAD, generator=g, dtype=torch.float64)
    o_vview = o[..., cl.V_ORDER]
    assert torch.allclose(w_o[:, cl.o_proj_columns(4)] @ o_vview.reshape(5, -1).T, w_o @ o.reshape(5, -1).T)
    assert torch.equal(cl.unpermute_output(o_vview), o)


def test_views_are_strided_with_a_unit_last_stride():
    row = torch.empty(3, 16, 4, cl.ROW, dtype=torch.bfloat16)        # [blocks, block tokens, KV heads, 640]
    for view in (cl.key_view(row), cl.value_view(row)):
        assert view.shape[-1] == 512 and view.stride(-1) == 1 and view.stride(-2) == cl.ROW
        assert not view.is_contiguous()
    assert cl.value_view(row).data_ptr() - cl.key_view(row).data_ptr() == cl.NR * row.element_size()


def test_bf16_compact_scores_match_the_e56_emulation():
    """The compact path computes the same dot products as the E-55 and E-56 BF16C arm (only the order of the terms
    differs), so E-56 measured this arithmetic."""
    import splice_e55 as sp
    w = 0.0615234375
    k, v = model_kv(t=16, w=w, dtype=torch.bfloat16)
    q = torch.randn(16, 8, cl.HEAD, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    rot, unrot = sp.rot_dims(cl.HEAD)
    q2, k2, _ = sp.transform("BF16C", q, k, v, w, {"ks": 1.0, "vs": 1.0}, rot, unrot)
    emul = torch.einsum("thd,shd->hts", q2.float(), k2.float().repeat_interleave(4, dim=1))
    qc, kc = cl.prep_query(q, w), cl.key_view(cl.build_row(k, v))
    comp = torch.einsum("thd,shd->hts", qc.float(), kc.float().repeat_interleave(4, dim=1))
    assert torch.allclose(comp, emul, rtol=1e-5, atol=1e-4)


def test_page_plan_and_request_memory():
    assert cl.page_bytes(64, 4, cl.ROW, 2) / cl.page_bytes(64, 16, 512, 2) == 0.3125         # the 3.2 ratio
    sliding_page = cl.page_bytes(16, 16, 512, 2)                                              # 262,144 B
    assert cl.compact_block(sliding_page, 4, 2) == 48                                         # B3 of gate G1
    assert cl.page_bytes(48, 4, cl.ROW, 2) == 245_760 < sliding_page                         # padded by 6.25%
    for tokens, std, comp in ((32_768, 3.28, 2.34), (65_536, 5.78, 3.91), (131_072, 10.78, 7.03)):
        assert abs(cl.request_kv_gib(tokens, False) - std) < 0.01 and abs(cl.request_kv_gib(tokens, True) - comp) < 0.01
