"""S3 tests of E-58 (the layouts, the reference read, the stub benchmark and the frozen analysis words). Run with:
uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p no:cacheprovider
experiments/gemma4_kv_microbench/"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import analyze_e58 as an  # noqa: E402
import bench_e58 as b  # noqa: E402


def filled(batch=2, length=100, seed=0):
    x = (torch.randn(batch, length, b.HK, b.ROW_C, generator=torch.Generator().manual_seed(seed)) * 0.5).to(
        torch.bfloat16)
    lays = {lo: b.Layout(lo, batch, length, "cpu") for lo in b.LAYOUTS}
    for lay in lays.values():
        lay.fill(x)
    return x, lays


def test_layout_pages_strides_and_views():
    _, lays = filled()
    assert {lo: lay.page * 2 for lo, lay in lays.items()} == {"S32": 262_144, "S48": 393_216, "C32": 262_144,
                                                              "C48": 262_144}
    for lo, lay in lays.items():
        kc, vc = lay.views()
        assert kc.shape[1:] == (lay.block, b.HK, b.D) and vc.shape == kc.shape and vc.stride() == kc.stride()
        offset = vc.storage_offset() - kc.storage_offset()
        assert offset == (b.D if lay.kind == "stock" else b.cl.NR)       # V at +512 (stock) or +128 (the V view)
    assert int(lays["C48"].slots[1]) == (lays["C48"].per_req + 100 // 48) * 48 + 100 % 48


def test_every_layout_holds_the_same_logical_k_and_v():
    x, lays = filled()
    for lay in lays.values():
        kc, vc = lay.views()
        for r, t in ((0, 0), (0, 47), (1, 48), (1, 99)):
            blk = int(lay.block_table[r, t // lay.block])
            assert torch.equal(kc[blk, t % lay.block], x[r, t, :, :b.D])
            assert torch.equal(vc[blk, t % lay.block], x[r, t, :, 128:])


def test_reference_read_matches_float_attention_and_is_layout_independent():
    x, lays = filled()
    q = (torch.randn(2, b.HQ, b.D, generator=torch.Generator().manual_seed(1)) * 0.1).to(torch.bfloat16)
    cu, seqused = torch.arange(3, dtype=torch.int32), torch.full((2,), 100, dtype=torch.int32)
    outs = {}
    for lo, lay in lays.items():
        kc, vc = lay.views()
        o = torch.empty(2, b.HQ, b.D, dtype=torch.bfloat16)
        b.ref_read(**b.read_kwargs(q, kc, vc, o, lay, 100, cu, seqused, (None, None, None), None))
        outs[lo] = o
    for lo in b.LAYOUTS:
        assert torch.equal(outs[lo], outs["S32"]), lo
    k = x[..., :b.D].float().repeat_interleave(8, dim=2)
    v = x[..., 128:].float().repeat_interleave(8, dim=2)
    p = torch.softmax(torch.einsum("bhd,blhd->bhl", q.float(), k), dim=-1)
    want = torch.einsum("bhl,blhd->bhd", p, v)
    assert float((outs["S32"].float() - want).abs().max()) < 2e-2


def test_stub_benchmark_and_analysis(tmp_path):
    run = subprocess.run([sys.executable, "-B", str(HERE / "bench_e58.py"), "--out", str(tmp_path), "--stub",
                          "--layers", "2", "--rounds", "2"], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-2_000:] + run.stderr[-2_000:]
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert rec["status"] == "ok" and sorted(rec["shapes"]) == ["D32", "D64"]
    for sh in rec["shapes"].values():
        assert not sh["errors"]
        assert set(sh["R"]) == set(b.LAYOUTS) and all(len(v) == 2 for v in sh["R"].values())
        assert set(sh["P"]) == {"S32 raw", "S32 wrapped", "C48 raw", "C48 wrapped"}
        assert set(sh["H"]) == {"NONE", "OFF", "VIEWS", "EAGER_FN"} and set(sh["X"]) == {"STOCK", "EMULV", "COMPACT"}
        assert len(sh["v1"]) == 6 and all(v["identical"] for v in sh["v1"].values())
    out = subprocess.run([sys.executable, "-B", str(HERE / "analyze_e58.py"), "--res", str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2_000:]
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("E-58: REHEARSAL") and "| D32 cache:" in last and "scratch guard: 0 values" in last
    assert "guard_values_changed" in rec["scratch_guard"] and rec["scratch_guard"]["outputs_identical"]


def record(views=-0.2, block=0.05, wrapper=0.1, hook_cpu=2.4, hook_graph=0.2, d64_views=-0.1, gpu="NVIDIA A100-SXM4-80GB",
           spread=0.0, busy=True, v1=0.0, loop=10.0):
    """A synthetic bench.json with known per-step differences (ms)."""
    def cpu(t):
        return {"cpu": t, "busy": busy, "loop_max_ms": loop}

    def shape(vw):
        base = 5.0
        r = {"S32": base, "S48": base + 0.02, "C32": base + vw, "C48": base + vw + block}
        return {"errors": [], "v1": {k: {"identical": v1 == 0, "max_abs": v1} for k in
                                     ("S32", "S48", "C32", "C48", "C48_wrapped", "X_COMPACT_vs_EMULV")},
                "R": {lo: [t, t * (1 + spread), t] for lo, t in r.items()}, "W": {"S32": 0.05, "C48": 0.05},
                "sleep_ms": 70.0,
                "P": {"S32 raw": cpu(1.0), "S32 wrapped": cpu(1.2), "C48 raw": cpu(1.0), "C48 wrapped": cpu(1.2 + wrapper)},
                "H": {"NONE": {**cpu(0.1), "gpu": 0.0, "graph": 0.0},
                      "OFF": {**cpu(0.2), "gpu": 0.0, "graph": 0.0},
                      "VIEWS": {**cpu(0.1 + hook_cpu), "gpu": hook_cpu, "graph": hook_graph},
                      "EAGER_FN": {**cpu(1.0), "gpu": 1.0, "graph": hook_graph}},
                "X": {k: {**cpu(1.0), "gpu": 6.0, "wall": 7.0, "graph": 5.0} for k in ("STOCK", "EMULV", "COMPACT")},
                "bytes_distinct": {lo: 10 ** 9 for lo in b.LAYOUTS}}
    return {"status": "ok", "meta": {"gpu": gpu, "vllm": an.VLLM, "rehearsal": False, "stub": False},
            "shapes": {"D32": shape(views), "D64": shape(d64_views)}}


def words(tmp_path, **kw) -> dict:
    (tmp_path / "bench.json").write_text(json.dumps(record(**kw)))
    return an.summarize(tmp_path)


def test_analysis_words(tmp_path):
    s = words(tmp_path)
    assert s["word"] == "GO" and s["cache_words"] == ["CACHE-NONE"] and s["hook_word"] == "HOOK-CPU", s
    assert s["hook_explains_e57"] is True and s["cache_cost_open"] is True
    assert words(tmp_path, views=0.8)["cache_words"] == ["CACHE-VIEWS"]
    assert words(tmp_path, block=0.6)["cache_words"] == ["CACHE-BLOCK"]
    assert words(tmp_path, wrapper=0.7)["cache_words"] == ["CACHE-WRAPPER"]
    assert words(tmp_path, views=-1.0)["cache_faster"] is True
    # A1 (R-84 B1): four disjoint hook words from C (CPU) and G (graph GPU).
    assert words(tmp_path, hook_cpu=2.4, hook_graph=0.9)["hook_word"] == "HOOK-MIXED"
    assert words(tmp_path, hook_cpu=0.4, hook_graph=0.9)["hook_word"] == "HOOK-GPU"
    s = words(tmp_path, hook_cpu=1.5, hook_graph=0.2)
    assert s["hook_word"] == "HOOK-CPU" and s["hook_explains_e57"] is False
    s = words(tmp_path, hook_cpu=0.4)
    assert s["hook_word"] == "HOOK-SMALL" and s["word"] == "PIVOT"
    assert words(tmp_path, views=1.6, d64_views=1.7)["word"] == "KILL"
    assert words(tmp_path, views=1.6, d64_views=0.2)["word"] == "GO"


def test_analysis_validity(tmp_path):
    assert words(tmp_path, v1=0.5)["word"] == "INVALID"
    assert words(tmp_path, v1=0.001)["word"] == "GO"                     # A1: not identical, but within 0.002
    assert words(tmp_path, v1=0.005)["word"] == "INVALID"
    assert words(tmp_path, busy=False)["word"] == "INVALID"
    assert words(tmp_path, loop=40.0)["word"] == "INVALID"               # A1 (R-84 F3): a loop above half the sleep
    assert words(tmp_path, spread=0.2)["word"] == "INVALID"
    assert words(tmp_path, spread=0.05)["word"] == "INVALID"              # A1 (R-84 B3): V4 at 3%
    assert words(tmp_path, spread=0.02)["word"] == "GO"
    assert words(tmp_path, gpu="NVIDIA A40")["word"] == "INVALID"
    rec = record()
    rec["meta"]["rehearsal"] = True
    rec["meta"]["gpu"] = "NVIDIA A40"
    (tmp_path / "bench.json").write_text(json.dumps(rec))
    assert an.summarize(tmp_path)["word"] == "REHEARSAL"
    rec = copy.deepcopy(record())
    rec["shapes"]["D32"]["errors"] = [{"arm": "X COMPACT", "error": "boom"}]
    (tmp_path / "bench.json").write_text(json.dumps(rec))
    assert an.summarize(tmp_path)["validity"]["V5"]
