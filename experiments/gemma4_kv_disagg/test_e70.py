"""E-70 stage 1 Mac tests (README Section 12): the byte view, the NIXL region rule (a copy of the pinned branch), the
byte check, the word, the manifest, and the stub session end to end."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_e70 as an  # noqa: E402
import e70_stage as stg  # noqa: E402
import nixl_compact as nx  # noqa: E402

NB, PAGE = 6, 128                      # 6 blocks of a 128-byte padded page; 2 layers share one buffer


def shared_buffer():
    """One int8 buffer, block-major, with two layer slots per block row (as the hybrid allocator overlays groups)."""
    return torch.zeros(NB * 2 * PAGE, dtype=torch.int8)


def stock_view(buf, slot: int):
    """A stock layer view [B, H=2, N=4, C=8] bf16: 128 content bytes per 256-byte row slot."""
    base = buf.view(torch.bfloat16)
    return base.as_strided((NB, 2, 4, 8), (2 * PAGE // 2, 32, 8, 1), slot * PAGE // 2)


def compact_view(buf, slot: int):
    """A compact layer view [B, H=1, N=6, C=10] bf16: 120 content bytes in a 128-byte padded page."""
    base = buf.view(torch.bfloat16)
    return base.as_strided((NB, 1, 6, 10), (2 * PAGE // 2, 60, 10, 1), slot * PAGE // 2)


def nixl_region_specs(cache: torch.Tensor, num_blocks: int, physical_page_size: int, registration_len: int):
    """A copy of the pinned non-MLA, non-Mamba attention branch (nixl/base_worker.py:1700-1774 at 3966924d5),
    without the packed-MLA arms (not used by Gemma 4)."""
    if cache.ndim == 1:
        block_stride = physical_page_size
    else:
        block_stride = cache.stride(0) * cache.element_size()
    storage_is_block_major = 0 <= registration_len - num_blocks * block_stride < block_stride
    page_contiguous = cache.ndim > 1 and cache[0].is_contiguous() and cache[0].nbytes == physical_page_size
    if storage_is_block_major and not page_contiguous:
        return [("storage", block_stride, block_stride)]
    if storage_is_block_major:
        return [(cache.data_ptr(), physical_page_size, block_stride)]
    segment_bytes = num_blocks * block_stride
    if cache.nbytes % segment_bytes != 0:
        raise AssertionError("KV cache view cannot be partitioned into NIXL regions")
    num_segments = cache.nbytes // segment_bytes
    region_block_len = block_stride if num_segments > 1 else physical_page_size
    return [(cache.data_ptr() + k * segment_bytes, region_block_len, block_stride) for k in range(num_segments)]


def test_byte_view_refuses_rows_that_run_past_the_storage():
    """The guard: in a row-interleaved buffer, the padded rows of the second slot would end past the buffer, so the
    view is left unchanged (vLLM's views are region-major: a layer's NB pages are consecutive; next test)."""
    buf = shared_buffer()
    c = compact_view(buf, 1)
    v, why = nx.byte_view(c)
    assert why == "padding beyond the storage" and v is c
    v0, why0 = nx.byte_view(compact_view(buf, 0))
    assert why0 == "byte view" and tuple(v0.shape) == (NB, 2 * PAGE) and v0.data_ptr() == buf.data_ptr()


def test_the_nixl_rule_rejects_the_compact_view_and_accepts_the_byte_view():
    """The real geometry: one buffer of NB rows, each row holds one page per layer group slot; a layer's view has its
    page at a fixed offset and a stride of the whole row. Here a row is ONE page (as in vLLM: the groups overlay the
    same bytes), so the stock view fills its page and the compact view does not."""
    buf = torch.zeros(NB * PAGE, dtype=torch.int8)
    base = buf.view(torch.bfloat16)
    stock = base.as_strided((NB, 2, 4, 8), (PAGE // 2, 32, 8, 1), 0)          # 128 content bytes, stride 128
    compact = base.as_strided((NB, 1, 6, 10), (PAGE // 2, 60, 10, 1), 0)       # 120 content bytes, stride 128
    reg_len = buf.untyped_storage().nbytes() * 10                              # the shared buffer is larger
    assert nixl_region_specs(stock, NB, PAGE, reg_len) == [(stock.data_ptr(), PAGE, PAGE)]
    assert nx.byte_view(stock)[1] == "unchanged"
    with pytest.raises(AssertionError, match="cannot be partitioned"):
        nixl_region_specs(compact, NB, PAGE, reg_len)
    v, why = nx.byte_view(compact)
    assert why == "byte view"
    assert nixl_region_specs(v, NB, PAGE, reg_len) == [(compact.data_ptr(), PAGE, PAGE)]


def test_byte_view_bytes_cover_the_content_first():
    buf = torch.zeros(NB * PAGE, dtype=torch.int8)
    base = buf.view(torch.bfloat16)
    compact = base.as_strided((NB, 1, 6, 10), (PAGE // 2, 60, 10, 1), 0)
    compact[3].fill_(1.5)
    v, _ = nx.byte_view(compact)
    assert torch.equal(v[3, :120].view(torch.bfloat16), compact[3].reshape(-1))   # the content, then the padding
    assert int(v[3, 120:].abs().sum()) == 0


def test_patch_views_records_only_the_compact_layers():
    nx.STATE["patched"].clear()
    buf = torch.zeros(NB * PAGE, dtype=torch.int8)
    base = buf.view(torch.bfloat16)
    kv = {"glob": base.as_strided((NB, 1, 6, 10), (PAGE // 2, 60, 10, 1), 0),
          "slid": base.as_strided((NB, 2, 4, 8), (PAGE // 2, 32, 8, 1), 0)}
    out = nx.patch_views(kv)
    assert out["slid"] is kv["slid"] and out["glob"].dtype == torch.uint8
    assert [p[0] for p in nx.STATE["patched"]] == ["glob"]


def test_compare():
    p = {0: {1: 10, 2: 11}, 5: {0: 1, 1: 2, 2: 3}}
    d = {0: {1: 10, 2: 11, 3: 99}, 5: {0: 1, 1: 7}}
    c = nx.compare({str(g): {str(i): h for i, h in b.items()} for g, b in p.items()}, d, {0})
    assert c["checked"] == 4 and c["mismatch"] == 1 and c["sliding"] == 2 and c["global"] == 2 and c["d_only"] == 1
    assert c["bad"] == [[5, 1]]


def mounted_tree(root: Path) -> Path:
    for local, remote in stg.mounts():
        dst = root / remote.removeprefix(stg.CODE + "/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())
    return root


def test_manifest_is_current_and_cost_cap():
    for local, _ in stg.mounts():
        assert local.exists(), local
    assert json.loads(stg.MANIFEST_LOCAL.read_text()) == stg.make_manifest()
    assert 2.4 < stg.cost_cap() < 2.6


def test_stub_session_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("STUB_CORRUPT", raising=False)
    code = mounted_tree(tmp_path / "code")
    out = stg.session(str(code), str(tmp_path / "res"), "e70-stub", stub=True)
    rec = out["record"]
    assert rec["manifest_ok"] and rec["ok"], rec
    assert "E-70: PATCH-WORKS | GAP-CONFIRMED" in rec["decision"], rec["decision"]
    top = stg.unpack(out["tar"], tmp_path / "copy")
    assert (top / "summary.json").exists() and (top / "COMPACT" / "d_run.json").exists()
    assert stg.session(str(code), str(tmp_path / "res"), "e70-stub", stub=True)["record"]["restart"]


def test_stub_session_finds_a_changed_block(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CORRUPT", "1")
    code = mounted_tree(tmp_path / "code")
    rec = stg.session(str(code), str(tmp_path / "res"), "e70-bad", stub=True)["record"]
    assert "INVALID (ENV" in rec["decision"]          # the stock pair also gets the planted change: V3 fails first


def test_lookup_matches_a_suffixed_request_id():
    """A1 (R-114 B1): the scheduler's id can be "<id>-<8 chars>"; the output keeps "<id>"."""
    import run_e70
    store = {"3-a1b2c3d4": "x", "13-ffff0000": "y"}
    assert run_e70.lookup(store, "3") == "x" and run_e70.lookup(store, "13") == "y"
    assert run_e70.lookup({"3": "z"}, "3") == "z" and run_e70.lookup(store, "4") is None


def test_sampling_ignores_eos():
    """A1 (R-114 B2): every request decodes the full count (the chat answers end early otherwise)."""
    import run_e70
    kw = run_e70.sampling_kwargs(32, {"do_remote_decode": True})
    assert kw["ignore_eos"] is True and kw["max_tokens"] == 32 and kw["extra_args"]["kv_transfer_params"]
    assert "extra_args" not in run_e70.sampling_kwargs(32, None)


def test_stop_rule_reads_the_role_status(tmp_path):
    """A1 (R-114 B3): a role that wrote status ok counts as ok whatever its exit code."""
    (tmp_path / "STOCK").mkdir()
    (tmp_path / "STOCK" / "d_run.json").write_text(json.dumps({"status": "ok"}))
    assert stg.role_status(tmp_path, "STOCK", "d") == "ok" and stg.role_status(tmp_path, "STOCK", "p") == "missing"


def test_stage2_stub_session(tmp_path, monkeypatch):
    monkeypatch.delenv("STUB_CORRUPT", raising=False)
    code = mounted_tree(tmp_path / "code")
    rec = stg.session(str(code), str(tmp_path / "res"), "e70s2-stub", stub=True, pairs=stg.STAGE2_PAIRS)["record"]
    assert rec["stage"] == 2 and "E-70 stage 2: RELOAD-WORKS" in rec["decision"], rec["decision"]
    monkeypatch.setenv("STUB_CORRUPT", "1")
    rec = stg.session(str(code), str(tmp_path / "res"), "e70s2-bad", stub=True, pairs=stg.STAGE2_PAIRS)["record"]
    assert "INVALID (V3" in rec["decision"], rec["decision"]          # the stock pair fails its NIXL check first


def test_stage2_engine_events_and_stats():
    """A1 of stage 2 (R-116 B1, B2, F2)."""
    import run_e70
    kw = run_e70.d2_engine()
    assert kw["max_model_len"] >= 7_667 + run_e70.S2_TOKENS and kw["disable_log_stats"] is False
    assert kw["prefix_cache_retention_interval"] is None
    kids = kw["kv_transfer_config"]["kv_connector_extra_config"]["connectors"]
    assert [k["kv_connector"] for k in kids] == ["NixlConnector", "OffloadingConnector"]
    assert kids[1]["kv_connector_extra_config"]["offload_prompt_only"] is False
    assert not run_e70.event_due([], 256) and not run_e70.event_due([300, 10], 256) and run_e70.event_due([300, 256], 256)
    from types import SimpleNamespace
    import e69_hooks as hk
    ms = run_e70.MultiStatsSum(hk.StatsSum())
    payload = {"types": {"vllm:kv_offload_load_bytes": "counter", "vllm:kv_offload_load_time": "counter"},
               "data": {"vllm:kv_offload_load_bytes": {(): 4e9}, "vllm:kv_offload_load_time": {(): 0.2}}, "info": None}
    ms.add(SimpleNamespace(connector_prefix_cache_stats=None,
                           kv_connector_stats={"NixlConnector": {"num_xfers": 3}, "OffloadingConnector": payload}))
    assert abs(ms.summary()["load_gb_per_s"] - 20.0) < 1e-9


def test_stage2_word_rule():
    import analyze_e70s2 as a2

    def d(**kw):
        base = {"status": "ok", "gpu": "NVIDIA L40S", "manifest_ok": True, "spec": {"applied": True},
                "kv": {"layout": "LBHNC"}, "n_complete": 8, "answers": [{"gold_in_text": True}] * 8,
                "checks": [{"checked": 5, "mismatch": 0, "global": 2, "sliding": 3}] * 8,
                "counters": {"preemptions": 40},
                "coverage": {"median_c": 0.97, "min_c": 0.92, "zero_hit_events": 0, "decode_events": 24},
                "reload": {"checked_reloads": 24, "mismatched_blocks": 0}, "stats": {}}
        base["checks"] = [dict(c, hit=7666, prompt=7666) for c in base["checks"]]
        base.update(kw)
        return base
    pairs = {"STOCK2": {"p": d(), "d": d()}, "COMPACT2": {"p": d(), "d": d()}}
    assert a2.word(pairs)["word"] == "RELOAD-WORKS"
    pairs["COMPACT2"]["d"] = d(coverage={"median_c": 0.0, "min_c": 0.0, "zero_hit_events": 0, "decode_events": 24})
    assert a2.word(pairs)["word"] == "RELOAD-PROMPT-ONLY"
    pairs["COMPACT2"]["d"] = d(coverage={"median_c": 0.5, "min_c": 0.2, "zero_hit_events": 0, "decode_events": 24})
    assert a2.word(pairs)["word"] == "RELOAD-PARTIAL"
    # A1 (R-116 F1): zero-hit reloads hidden behind a high median are not WORKS, and not PROMPT-ONLY
    pairs["COMPACT2"]["d"] = d(coverage={"median_c": 0.97, "min_c": 0.0, "zero_hit_events": 3, "decode_events": 24})
    assert a2.word(pairs)["word"] == "RELOAD-PARTIAL"
    pairs["COMPACT2"]["d"] = d(n_complete=5)
    assert a2.word(pairs)["word"].startswith("INVALID (too few")
    pairs["COMPACT2"]["d"] = d(checks=[{"checked": 0, "mismatch": 0, "global": 0, "sliding": 0, "hit": 7666,
                                        "prompt": 7666}] * 8)
    assert a2.word(pairs)["word"].startswith("INVALID (too few")
    pairs["COMPACT2"]["d"] = d(reload={"checked_reloads": 24, "mismatched_blocks": 2})
    assert a2.word(pairs)["word"] == "RELOAD-FAILS"
    pairs["COMPACT2"]["d"] = {"status": "failed", "gpu": "NVIDIA L40S", "manifest_ok": True,
                              "spec": {"applied": True}, "layout_env": "LBHNC", "error": "x",
                              "traceback": "kv_offload/cpu/gpu_worker.py"}
    assert a2.word(pairs)["word"] == "RELOAD-FAILS"


def test_word_rule_cases():
    def role(**kw):
        base = {"status": "ok", "gpu": "NVIDIA L40S", "manifest_ok": True, "spec": {"applied": True},
                "kv": {"layout": "LBHNC"}, "ref": [{"tokens": [1, 2], "gold_in_text": True}] * 8,
                "answers": [{"tokens": [1, 2], "gold_in_text": True}] * 8, "n_complete": 8,
                "checks": [{"checked": 5, "mismatch": 0, "global": 2, "sliding": 3}] * 8}
        base.update(kw)
        return base
    pairs = {"STOCK": {"p": role(), "d": role()},
             "COMPACT-NOPATCH": {"p": {"status": "failed", "manifest_ok": True,
                                       "error": "AssertionError: KV cache view cannot be partitioned into NIXL regions"},
                                 "d": None},
             "COMPACT": {"p": role(), "d": role()}}
    out = an.word(pairs)
    assert out["word"] == "PATCH-WORKS" and out["gap"] == "GAP-CONFIRMED"
    pairs["COMPACT"]["d"] = role(checks=[{"checked": 5, "mismatch": 1, "global": 2, "sliding": 3}] * 8)
    assert an.word(pairs)["word"] == "PATCH-FAILS"
    pairs["COMPACT"]["d"] = {"status": "failed", "manifest_ok": True, "error": "x"}
    assert an.word(pairs)["word"] == "PATCH-FAILS"
    pairs["STOCK"]["d"] = {"status": "failed", "manifest_ok": True, "error": "UCX"}
    assert an.word(pairs)["word"].startswith("INVALID (ENV")
