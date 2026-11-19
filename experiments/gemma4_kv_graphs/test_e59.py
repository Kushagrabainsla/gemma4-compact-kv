"""S3 tests of E-59 (the plan and scratch rules, the window, the trace summary, the token check, the stub session, the
planted failures and the frozen decision words). Run with:
uv run --no-project --with numpy --with torch --with pytest python -B -m pytest -q -p no:cacheprovider
experiments/gemma4_kv_graphs/"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import analyze_e59 as an  # noqa: E402
import run_e59 as r  # noqa: E402


def test_plan_rules():
    assert r.parse_rows("STOCK:3, emulv:3,COMPACT") == [("STOCK", 3), ("EMULV", 3), ("COMPACT", 3)]
    assert r.plan_problems("stock", True, "OFF", [("STOCK", 3), ("STOCK", 3)]) == []
    assert r.plan_problems("stock", False, "OFF", [("STOCK", 3), ("EMULV", 3)]) == []      # eager can switch
    assert any("capture mode" in p for p in r.plan_problems("stock", True, "OFF", [("EMULV", 3)]))
    assert any("needs the capture mode VIEWS" in p for p in r.plan_problems("compact", False, "OFF", [("COMPACT", 3)]))
    assert any("compact layout" in p for p in r.plan_problems("stock", False, "VIEWS", [("COMPACT", 3)]))
    assert any("batch 5" in p for p in r.plan_problems("stock", False, "OFF", [("STOCK", 5)]))


def test_scratch_rule():
    ok = {"rows": 8, "headdim": 256, "seq_threshold_3D": 8, "shape": [8, 32, 16, 512]}    # widened (A1, R-86 B1)
    assert r.scratch_problems([ok], 4) == []
    narrow = dict(ok, shape=[8, 32, 16, 256])                       # the stock scratch: the size-8 capture overflows
    assert any("width 256" in p for p in r.scratch_problems([narrow], 3))
    assert any("rows 2" in p for p in r.scratch_problems([dict(ok, rows=2, shape=[2, 32, 16, 512])], 3))
    assert r.scratch_problems([], 3) == ["no Triton metadata builder found"]


def test_runner_builders_widened():
    """A2 (rehearsal 1): the builders come from the runner's attn_groups and get a 512-wide scratch before the
    capture; a wide scratch stays as it is."""
    import torch

    class B:
        def __init__(self, width):
            self.softmax_segm_output = torch.empty(8, 2, 16, width)
            self.seq_threshold_3D, self.headdim, self.block_size = 8, 256, 16

    class G:
        def __init__(self, *bs):
            self.metadata_builders = list(bs)

    class R:
        attn_groups = [[G(B(256)), G(B(256))], [G(B(512))]]

    runner = R()
    assert len(r.builders_of(runner)) == 3 and r.builders_of(object()) == []
    assert r.widen_builders(runner, "t") == 2 and r.widen_builders(runner, "t") == 0
    rec = r.scratch_record(runner)
    assert [x["shape"] for x in rec] == [[8, 2, 16, 512]] * 3 and r.scratch_problems(rec, 4) == []
    assert rec[0]["class"].endswith("B")


def steps_of(seconds, batch=3, prefill=5, probe=0, replays=1):
    st = [{"running": 1, "decoding": 0, "seconds": 1.0, "held": 10, "replays": 0} for _ in range(prefill)]
    st += [{"running": batch, "decoding": batch, "seconds": s, "held": 20, "cpu": s / 4, "replays": replays}
           for s in seconds]
    st += [{"running": batch, "decoding": batch, "seconds": 9.0, "held": 20, "probe": True, "replays": 1}
           for _ in range(probe)]
    return st


def test_window_stats():
    secs = [0.5, 0.5, 0.5] + [0.040] * 60 + [0.050] * 20 + [0.30]
    w = r.window_stats(steps_of(secs, probe=6), 3, skip=3, window=128)
    assert w["window_steps"] == 81 and w["full_steps"] == 84             # the profiled steps are not in the run
    assert abs(w["median_ms"] - 40.0) < 1e-9 and w["dropped"] == 1      # 0.30 > 5 x the median
    assert w["replays"] == 81 and abs(w["slow_share"] - 21 / 81) < 1e-3
    assert r.window_stats(steps_of([0.04] * 200), 3, window=128)["window_steps"] == 128


def test_trace_summary():
    tr = {"traceEvents": [
        {"cat": "kernel", "name": "kernel_unified_attention_3d", "ts": 0, "dur": 10},
        {"cat": "kernel", "name": "ampere_bf16_s16816gemm", "ts": 5, "dur": 10},        # overlaps the first
        {"cat": "kernel", "name": "index_select_kernel", "ts": 30, "dur": 10},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0, "dur": 1},
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 1, "dur": 1}]}
    s = r.summarize_trace(tr, 2)
    assert s["kernels"] == 3 and abs(s["busy_ms"] - 0.0125) < 1e-9 and abs(s["span_ms"] - 0.02) < 1e-9
    assert abs(s["gap_ms"] - 0.0075) < 1e-9 and s["graph_launches_per_step"] == 0.5
    assert s["category_ms"]["attention"] == 0.005 and s["category_ms"]["gemm"] == 0.005


def test_token_check():
    a = {"tokens": {"0": list(range(200)), "1": list(range(150))}}
    assert an.same_tokens(a, copy.deepcopy(a), 128)[0]
    b = copy.deepcopy(a)
    b["tokens"]["1"][140] = -1
    ok, why = an.same_tokens(a, b, 128)
    assert not ok and "position 140" in why
    assert not an.same_tokens(a, {"tokens": {"0": list(range(100)), "1": list(range(150))}}, 128)[0]


def run_session(res: Path, plant: str = "") -> str:
    env = dict(os.environ, STUB="1", CODE=str(HERE.parent), RES=str(res), BOOT_LOG="/nonexistent", STUB_PLANT=plant)
    out = subprocess.run(["sh", str(HERE / "session_e59.sh")], env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2_000:] + out.stderr[-2_000:]
    return (res / "decision.txt").read_text().strip().splitlines()[-1]


@pytest.fixture(scope="module")
def stub_res(tmp_path_factory):
    res = tmp_path_factory.mktemp("e59") / "res"
    return res, run_session(res)


def test_stub_session(stub_res):
    res, last = stub_res
    assert last.startswith("E-59: REHEARSAL") and last.endswith("failed checks: none"), last
    s = an.summarize(res)
    assert all(v == "ok" for v in s["status"].values()) and s["v1"] == {"graph": "3 prompts equal",
                                                                      "eager": "3 prompts equal"}
    n = s["numbers"]
    assert n["OG"] < 0.02 and 0.03 < n["HE"] < 0.07 and n["GS"] < 0 and 1.2 < n["TG"] < 1.45
    assert s["hook_word"] == "HOOK-FIXED" and s["cache_word"] == "CACHE-NOREPRO"
    g1 = json.loads((res / "G1" / "run_e59.json").read_text())
    assert g1["hooks"]["attached_at"] == "load_model" and g1["rows"][0]["profile"]["kernels"] > 0
    assert all(row["replays"] >= row["window_steps"] for row in g1["rows"])
    e1 = json.loads((res / "E1" / "run_e59.json").read_text())
    assert [w["mode"] for w in e1["warm"]] == ["OFF", "VIEWS"] and all(row["replays"] == 0 for row in e1["rows"])
    assert (res / "code_sha.txt").read_text().count("\n") == 9
    assert [w["replays"] > 0 for w in g1["warm"]] == [True]                # A1 (R-86 B3)


def engine(tmp: Path, name: str, layout: str, graphs: int, capture: str, rows: str, plant: str = "") -> int:
    cmd = [sys.executable, "-B", str(HERE / "run_e59.py"), "--engine", name, "--layout", layout, "--graphs",
           str(graphs), "--capture", capture, "--rows", rows, "--out", str(tmp / name), "--stub"]
    if plant:
        cmd += ["--stub-plant", plant]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300).returncode


def test_planted_failures(tmp_path):
    assert engine(tmp_path, "G1", "stock", 1, "OFF", "STOCK:3", "scratch_small") == 3
    rec = json.loads((tmp_path / "G1" / "run_e59.json").read_text())
    assert rec["status"] == "preflight" and any("scratch" in p for p in rec["preflight"]) and not rec["rows"]
    assert engine(tmp_path, "X", "compact", 0, "OFF", "COMPACT:3") == 3                 # a plan error
    assert engine(tmp_path, "G2", "stock", 1, "VIEWS", "EMULV:3") == 0
    assert engine(tmp_path, "G3", "compact", 1, "VIEWS", "COMPACT:3", "graph_drops_hooks") == 0
    assert engine(tmp_path, "G4", "stock", 1, "OFF", "STOCK:3", "no_replay") == 3       # A1 (R-86 B3)
    g4 = json.loads((tmp_path / "G4" / "run_e59.json").read_text())
    assert g4["preflight"] == ["no CUDA graph replay in the warm-up"] and not g4["rows"]
    s = an.summarize(tmp_path)
    assert any("graph: COMPACT against EMULV" in p for p in s["validity"]["V1"])
    assert any("G1: status preflight" in p for p in s["validity"]["V4"]) and s["validity"]["V3"]


def test_session_stops_after_g1(tmp_path):
    last = run_session(tmp_path / "res", "scratch_small")
    log = (tmp_path / "res" / "session.log").read_text()
    assert "G1: PREFLIGHT STOP" in log and all(f"{e}: SKIPPED (G1 did not end ok)" in log for e in an.ENGINES[1:])
    assert "V3" in last and "V4" in last


def as_main_run(src: Path, dst: Path, scale: dict | None = None, edit=None) -> Path:
    """A copy of the stub records as if they came from the main run, with the row medians of some engines scaled."""
    shutil.copytree(src, dst)
    for e in an.ENGINES:
        p = dst / e / "run_e59.json"
        rec = json.loads(p.read_text())
        rec.update(stub=False, rehearsal=False, gpu="NVIDIA A100-SXM4-80GB", engine={"vllm": an.VLLM},
                   snapshot=f"/root/hf/hub/models--google--gemma-4-31B-it/snapshots/{an.REVISION}")
        for row in rec["rows"]:
            f = (scale or {}).get((e, row["index"]), (scale or {}).get(e, 1.0))
            row["median_ms"] = round(row["median_ms"] * f, 4)
        if edit:
            edit(e, rec)
        p.write_text(json.dumps(rec))
    return dst


def test_decision_words(stub_res, tmp_path):
    res, _ = stub_res
    assert an.summarize(as_main_run(res, tmp_path / "go"))["word"] == "GO"
    s = an.summarize(as_main_run(res, tmp_path / "kill", {"G3": 1.07}))
    assert s["word"] == "KILL" and s["numbers"]["OG"] >= 0.05
    s = an.summarize(as_main_run(res, tmp_path / "pivot", {"G3": 1.045}))
    assert s["word"] == "PIVOT", s["numbers"]
    s = an.summarize(as_main_run(res, tmp_path / "noisy", {"G4": 1.05}))
    assert s["word"] == "INVALID" and s["validity"]["V5"]
    s = an.summarize(as_main_run(res, tmp_path / "hook", {"G2": 1.06}))
    assert s["hook_word"] in ("HOOK-REDUCED", "HOOK-STAYS")
    s = an.summarize(as_main_run(res, tmp_path / "cache", {"E2": 1.06, "G3": 1.045}))
    assert s["cache_word"] == "CACHE-REAL", s["numbers"]
    bad = as_main_run(res, tmp_path / "gpu")
    rec = json.loads((bad / "G1" / "run_e59.json").read_text())
    rec["gpu"] = "NVIDIA A40"
    (bad / "G1" / "run_e59.json").write_text(json.dumps(rec))
    assert an.summarize(bad)["word"] == "INVALID"


def test_amended_words(stub_res, tmp_path):
    res, _ = stub_res
    s = an.summarize(as_main_run(res, tmp_path / "fast", {"G3": 0.96}))
    assert s["word"] == "GO" and s["flags"] == ["FASTER", "CACHE-GAIN"], (s["flags"], s["numbers"])
    s = an.summarize(as_main_run(res, tmp_path / "weak", {("G1", 2): 1.025, ("G3", 1): 1.032, ("G3", 2): 1.04}))
    assert s["word"] == "PIVOT-WEAK", s["numbers"]

    def bad_b4(e, rec):
        if e == "G3":
            rec["rows"][2]["end"] = "preempted"
    s = an.summarize(as_main_run(res, tmp_path / "b4", edit=bad_b4))
    assert s["word"] == "GO" and s["numbers"]["TG"] is None

    def replays(frac):
        def f(e, rec):
            if e == "G1":
                rec["rows"][0]["replays"] = int(frac * rec["rows"][0]["window_steps"])
        return f
    assert an.summarize(as_main_run(res, tmp_path / "rp95", edit=replays(0.95)))["word"] == "GO"
    s = an.summarize(as_main_run(res, tmp_path / "rp80", edit=replays(0.8)))
    assert s["word"] == "INVALID" and s["validity"]["V2"]
