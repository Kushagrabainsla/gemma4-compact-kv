"""Mac tests of the E-57 engine patch (no vLLM, no GPU). Run: uv run --no-project --with numpy --with torch --with
pytest python -B -m pytest -q -p no:cacheprovider experiments/gemma4_kv_engine/"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_model")]
import compact_engine as ce  # noqa: E402
import compact_layout as cl  # noqa: E402

W31 = 0.0615234375                       # a k_norm value of the 31B (E-55 and E-56 read 0.0600 to 0.0654)


@dataclass(frozen=True, kw_only=True)
class FakeSpec:
    """The fields of the vLLM AttentionSpec that the patch reads (kv_cache_interface.py:552-617)."""
    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: torch.dtype = torch.bfloat16
    page_size_padded: int | None = None
    state_content_bytes: int | None = None

    @property
    def page_size_bytes(self) -> int:
        row = self.state_content_bytes or (self.head_size + self.head_size_v) * ce.dtype_bytes(self.dtype)
        return self.page_size_padded or self.block_size * self.num_kv_heads * row


@dataclass(frozen=True, kw_only=True)
class FakeFull(FakeSpec):
    pass


@dataclass(frozen=True, kw_only=True)
class FakeSliding(FakeSpec):
    sliding_window: int = 1_024


def model_specs(global_heads: int, sliding_heads: int, n_global: int = 2, n_sliding: int = 4) -> dict:
    specs = {f"g{i}": FakeFull(block_size=16, num_kv_heads=global_heads, head_size=512, head_size_v=512)
             for i in range(n_global)}
    specs.update({f"s{i}": FakeSliding(block_size=16, num_kv_heads=sliding_heads, head_size=256, head_size_v=256)
                  for i in range(n_sliding)})
    return specs


@pytest.mark.parametrize("name, gh, sh, block", [("31B", 4, 16, 48), ("26B-A4B", 2, 8, 48), ("12B", 1, 8, 96)])
def test_compact_specs_follow_b3(name, gh, sh, block):
    specs = model_specs(gh, sh)
    new, info = ce.compact_specs(specs, True, FakeFull, FakeSliding)
    assert info["applied"] and info["block"] == block and sorted(info["layers"]) == ["g0", "g1"]
    sliding_page = 16 * sh * 512 * 2
    for n in ("g0", "g1"):
        s = new[n]
        assert (s.block_size, s.state_content_bytes, s.head_size, s.head_size_v) == (block, 1_280, 512, 512)
        assert s.page_size_bytes <= sliding_page < s.page_size_bytes + 16 * gh * 1_280   # pads by less than 16 tokens
    assert all(new[f"s{i}"] is specs[f"s{i}"] for i in range(4))


def test_compact_specs_leave_other_models_alone():
    specs = model_specs(4, 16)
    assert ce.compact_specs(specs, False, FakeFull, FakeSliding)[0] is specs
    padded = dict(specs, g0=FakeFull(block_size=16, num_kv_heads=4, head_size=512, head_size_v=512,
                                     page_size_padded=262_144))
    new, info = ce.compact_specs(padded, True, FakeFull, FakeSliding)
    assert new["g0"] is padded["g0"] and info["layers"] == ["g1"]


def lbnhc_cache(blocks: int, block: int, heads: int, row: int, page_elems: int | None = None) -> torch.Tensor:
    """One layer's [B, H, N, C] view with the LBNHC strides that `compute_layout_strides` gives (N outside H; a padded
    page sets the block stride)."""
    page = page_elems or block * heads * row
    raw = torch.zeros(blocks * page, dtype=torch.bfloat16)
    return torch.as_strided(raw, (blocks, heads, block, row), (page, row, heads * row, 1))


def triton_split(kv_cache: torch.Tensor, head_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """The split of `TritonAttentionImpl.forward` and `do_kv_cache_update` (triton_attn.py:655-657, 828)."""
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache


def write(key, value, key_cache, value_cache, slots):
    """The semantics of `triton_reshape_and_cache_flash`: token i goes to block slots[i] // N, offset slots[i] % N."""
    n = key_cache.shape[1]
    for i, s in enumerate(slots):
        key_cache[s // n, s % n] = key[i]
        value_cache[s // n, s % n] = value[i]


def gather(cache, table, length):
    n = cache.shape[1]
    return torch.stack([cache[table[j // n], j % n] for j in range(length)])          # [L, H, D]


def paged_attention(q, key_cache, value_cache, table, length):
    """The semantics of `unified_attention` for one sequence (causal, the queries are the last T positions)."""
    k, v = gather(key_cache, table, length).float(), gather(value_cache, table, length).float()
    t, hq, _ = q.shape
    g = hq // k.shape[1]
    s = torch.einsum("thd,lhd->htl", q.float(), k.repeat_interleave(g, dim=1))
    pos = torch.arange(length - t, length)
    s = s.masked_fill(torch.arange(length)[None, None, :] > pos[None, :, None], float("-inf"))
    return torch.einsum("htl,lhd->thd", torch.softmax(s, dim=-1), v.repeat_interleave(g, dim=1)).to(q.dtype)


def model_qkv(t=40, hq=8, hk=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(t, hk, 512, generator=g).to(torch.bfloat16)
    k = (v.float() * W31).to(torch.bfloat16)
    k[..., cl.ROT] = (torch.randn(t, hk, cl.NR, generator=g) * 0.5).to(torch.bfloat16)
    q = (torch.randn(t, hq, 512, generator=g) * 8).to(torch.bfloat16)
    return q, k, v


def run_layout(layout: str, mode: str, seed=0):
    """One sequence of 40 tokens through the cache write and the read, as the engine does: the stock row (1,024
    values, block 32) or the compact row (640 values, block 48, page padded to the stock page)."""
    q, k, v = model_qkv(seed=seed)
    ix, w = ce.Indexer(), torch.full((512,), W31)
    if mode == "OFF":
        q2, k2, v2 = q, k, v
    elif mode == "VIEWS":
        q2, (k2, v2) = ix.prep_query(q, w[cl.UNROT]), ix.write_operands(k, v)
    elif mode == "NAIVE":
        q2, (k2, v2) = ix.prep_query(q, w[cl.UNROT]), ix.naive_operands(k, v, w[cl.ROT])
    else:
        q2, (k2, v2) = ix.gross_query(q), ix.write_operands(k, v)
    if layout == "stock":
        cache = lbnhc_cache(6, 32, 2, 1_024)
        table, slots = [4, 1], [4 * 32 + i if i < 32 else 32 + i - 32 for i in range(40)]
    else:
        cache = lbnhc_cache(6, 48, 2, 640, page_elems=32 * 2 * 1_024)         # the padded page of B3
        table, slots = [3], [3 * 48 + i for i in range(40)]
    key_cache, value_cache = triton_split(cache)
    fixes = 0
    for _ in range(2):                               # the write and the read each fix the split
        vc, fixed = ce.fix_value_cache(key_cache, value_cache)
        fixes += fixed
    write(k2, v2, key_cache, vc, slots)
    out = paged_attention(q2[-8:], key_cache, vc, table, 40)
    if mode != "OFF":
        out = ix.unpermute(out)
    return out, fixes, (q, k, v)


def test_fix_value_cache_only_changes_compact_rows():
    comp = lbnhc_cache(3, 48, 4, 640)
    kc, vc = triton_split(comp)
    fixed, ok = ce.fix_value_cache(kc, vc)
    assert ok and fixed.shape == kc.shape and fixed.stride() == kc.stride()
    assert fixed.data_ptr() - kc.data_ptr() == cl.NR * 2
    comp.normal_()
    assert torch.equal(fixed, comp.transpose(1, 2)[..., cl.NR:])
    for row, head in ((1_024, 512), (512, 256)):                  # a stock global row and a sliding row
        kc, vc = triton_split(lbnhc_cache(3, 16, 4, row), head)
        assert ce.fix_value_cache(kc, vc) == (vc, False)


def test_compact_engine_equals_its_emulation_bit_for_bit():
    """EMULV (the stock cache with the VIEWS hooks) and COMPACT (the compact cache, the same hooks) read the same
    values in the same order, so their outputs are identical; both match the stock arithmetic to BF16 error."""
    comp, fixes, (q, k, v) = run_layout("compact", "VIEWS")
    emul, fixes_s, _ = run_layout("stock", "VIEWS")
    stock, _, _ = run_layout("stock", "OFF")
    assert fixes == 2 and fixes_s == 0
    assert torch.equal(comp, emul)
    ref = paged_attention_ref(q, k, v)
    assert float((stock.float() - ref).norm() / ref.norm()) < 1e-2
    assert float((comp.float() - ref).norm() / ref.norm()) < 1e-2


def paged_attention_ref(q, k, v):
    """The float64 attention of the last 8 queries over 40 keys (the model arithmetic)."""
    t, hq, _ = q.shape
    g = hq // k.shape[1]
    s = torch.einsum("thd,lhd->htl", q[-8:].double(), k.double().repeat_interleave(g, dim=1))
    pos = torch.arange(t - 8, t)
    s = s.masked_fill(torch.arange(t)[None, None, :] > pos[None, :, None], float("-inf"))
    return torch.einsum("htl,lhd->thd", torch.softmax(s, dim=-1), v.double().repeat_interleave(g, dim=1)).float()


def test_controls_change_the_output():
    views = run_layout("compact", "VIEWS")[0].float()
    for mode, low in (("NAIVE", 1e-2), ("GROSS", 1e-1)):
        out = run_layout("compact", mode)[0].float()
        assert float((out - views).norm() / views.norm()) > low, mode


def test_indexer_equals_compact_layout():
    q, k, v = model_qkv()
    ix, w = ce.Indexer(), torch.full((512,), W31)
    assert torch.equal(ix.prep_query(q, w[cl.UNROT]), cl.prep_query(q, W31))
    assert torch.equal(ix.prep_query(q, w[cl.UNROT]), cl.prep_query(q, w))
    for a, b in zip(ix.write_operands(k, v), cl.write_operands(k, v)):
        assert torch.equal(a, b)
    for a, b in zip(ix.naive_operands(k, v, w[cl.ROT]), cl.naive_operands(k, v, w)):
        assert torch.equal(a, b)
    assert torch.equal(ix.gross_query(q), cl.gross_query(q))
    o = torch.randn(5, 8, 512).to(torch.bfloat16)
    assert torch.equal(ix.unpermute(o), cl.unpermute_output(o))


def test_hooks_keep_the_stock_output_and_record_the_gaps():
    m = ce.StubAttention(8, 2)
    sp = ce.ViewsSplice()
    sp.attach(5, m, torch.full((512,), W31), (8, 2, 512))
    q, k, v = model_qkv()
    flat = lambda x: x.reshape(x.shape[0], -1)  # noqa: E731
    outs = {}
    for mode in ce.MODES:
        sp.set_mode(mode)
        sp.new_doc()
        outs[mode] = m(flat(q), flat(k), flat(v)).float()
    ref = outs["OFF"]
    assert float((outs["VIEWS"] - ref).norm() / ref.norm()) < 1e-2
    assert float((outs["GROSS"] - ref).norm() / ref.norm()) > 1e-1
    ev = sp.evidence()
    assert set(ev["gaps"]) == set(ce.MODES) and not ev["hook_errors"]
    assert ev["gaps"]["VIEWS"]["5"] < 1e-2 and ev["seps"]["VIEWS"]["5"] < 1e-2 and ev["seps"]["GROSS"]["5"] > 1e-1
    assert ev["counts"] == {f"{mode}|5": 40 for mode in ce.MODES}
    sp.detach()
    assert torch.equal(m(flat(q), flat(k), flat(v)).float(), ref)


def test_transform_does_not_index_with_python_lists(monkeypatch):
    """A1 (R-82 B4): the hook transform uses device tensors only (a list index of a CUDA tensor syncs the stream), and
    it equals the layout functions."""
    sp = ce.ViewsSplice()
    sp.attach(5, ce.StubAttention(8, 2), torch.full((512,), W31), (8, 2, 512))
    q, k, v = model_qkv()
    orig = torch.Tensor.__getitem__

    def guard(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        assert not any(isinstance(x, list) for x in keys), "a Python-list index in the hook transform"
        return orig(self, key)

    monkeypatch.setattr(torch.Tensor, "__getitem__", guard)
    got = {m: sp.transform(m, 5, q, k, v) for m in ("VIEWS", "NAIVE", "GROSS")}
    monkeypatch.undo()
    want = {"VIEWS": (cl.prep_query(q, W31),) + cl.write_operands(k, v),
            "NAIVE": (cl.prep_query(q, W31),) + cl.naive_operands(k, v, torch.full((512,), W31)),
            "GROSS": (cl.gross_query(q),) + cl.write_operands(k, v)}
    for m in want:
        assert all(torch.equal(a, b) for a, b in zip(got[m], want[m])), m


def test_register_reports_a_kv_shared_global_layer():
    """A1 (R-82 F8): a KV-shared global layer would read the compact cache without the hooks; V1 reports it."""
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_size, self.num_heads, self.num_kv_heads = 512, 8, 2

    class Gemma4Attention(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.is_sliding, self.is_kv_shared_layer, self.attn = False, shared, Attention()
            self.k_norm = torch.nn.Module()
            self.k_norm.weight = torch.nn.Parameter(torch.full((512,), W31))

    class Layer(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.self_attn = Gemma4Attention(shared)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Layer(False), Layer(True)])
    out = ce.register(model, ce.ViewsSplice())
    assert [h["layer"] for h in out["hooked"]] == [0]
    assert out["problems"] == ["layers.1.self_attn: a KV-shared global layer"]


def test_schedule_digest_tells_schedules_apart():
    a = [[1, 1, 2_046], [1, 1, 1]]
    assert ce.schedule_digest(a) == ce.schedule_digest([list(x) for x in a])
    assert ce.schedule_digest(a)["sha"] != ce.schedule_digest([[1, 2_047], [1, 1, 1]])["sha"]
    assert ce.schedule_digest(a)["steps"] == 2 and ce.schedule_digest(a)["tokens"] == 2_051


def test_capacity_arithmetic_reproduces_the_e56_log():
    """E-56 (A100, stock BF16): 14.58 GiB of KV, 2,621,440 B per pool block, max_num_batched_tokens 2,048 with two
    batches in flight, max_model_len 10,240 -> the log line 3.10x."""
    blocks = int(14.58 * 2**30 // 2_621_440)
    per_req = cl.blocks_per_request(10_240, 32, 16, 5, in_flight=2 * 2_048)
    assert per_req == 320 + 5 * 321 and round(blocks / per_req, 2) == 3.10
    for tokens, ratio in ((32_768, 1.338), (65_536, 1.403)):
        stock = cl.blocks_per_request(tokens, 32, 16, 5)
        comp = cl.blocks_per_request(tokens, 48, 16, 5)
        assert abs(stock / comp - ratio) < 1e-3
    assert cl.sliding_blocks(1_024, 16) == 65 and cl.sliding_blocks(1_024, 16, 2) == 66
    assert math.isclose(cl.compact_block(262_144, 4, 2) * 4 * 640 * 2 / 262_144, 0.9375)
