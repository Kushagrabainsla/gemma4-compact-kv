"""E-65 tests (README Section 12): the grid, the stub benchmark and its analysis, the W-M model, the checks, the
prefill term, the poolsim prefill chunks, the E-64 regression and the label rule."""
from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import analyze_e65 as an  # noqa: E402
import bench_e65 as be  # noqa: E402
import e65_model as em  # noqa: E402
import poolsim as ps  # noqa: E402


def test_grid():
    plan, core = be.grid()
    assert len(core) == 33 and len(plan) == 35 and plan[0] == be.TIE and be.TIE in core
    assert all(b * ln <= be.MAX_TOKENS for b, ln in core)
    assert plan[-2:] == list(be.EXTRA) and not set(be.EXTRA) & set(core)
    toks = [b * ln for b, ln in plan[1:-2]]
    assert toks == sorted(toks)                                   # A1 (R-101 B4): the largest points last
    assert (64, 1_024) in core and (16, 32_768) in core and (4, 131_072) in core and (16, 65_536) not in core
    assert all(b * ln <= 2 ** 17 for b, ln in be.REH_POINTS) and be.REH_POINTS[0] == be.TIE


def test_sliding_cache_window_blocks():
    import torch
    sc = be.SlidingCache(2, 4_096, torch.device("cpu"))
    assert sc.block_table.shape == (2, 257) and sc.first_real == 3_072 // 16 - 1
    assert int(sc.block_table[0, 0]) == 0 and int(sc.block_table[1, sc.first_real - 1]) == 0
    real = sc.block_table[:, sc.first_real:]
    assert len(set(real.flatten().tolist())) == real.numel() and int(real.min()) == 1
    assert sc.num_blocks == 1 + 2 * (257 - sc.first_real)
    small = be.SlidingCache(1, 512, torch.device("cpu"))
    assert small.first_real == 0 and int(small.block_table.min()) == 1


def stub_bench(tmp_path: Path, plant: str = "") -> dict:
    out = tmp_path / ("bench" + plant)
    cmd = [sys.executable, str(HERE / "bench_e65.py"), "--out", str(out), "--stub"]
    if plant:
        cmd += ["--stub-plant", plant]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads((out / "bench.json").read_text())


def test_stub_bench_and_analysis(tmp_path):
    bench = stub_bench(tmp_path)
    assert bench["status"] == "ok" and not bench["errors"]
    assert len(bench["points"]) == len(bench["meta"]["plan"]) == 9 and len(bench["meta"]["core"]) == 8
    assert all(p["v1"]["identical"] for p in bench["points"])
    seq = {(p["batch"], p["length"]) for p in bench["points"] if p["sequential"]}
    assert seq == {(4, 96), (2, 192), (4, 192)}                  # A1: one layout at a time above 300 tokens
    assert all(len(p["SL"]) == 3 and len(p["H"]) == 3 for p in bench["points"])
    res = an.analyze(bench)
    assert res["word"] == "REHEARSAL" and not res["failed"]
    assert set(res["fit"]["G_S32"]) == {"48", "96", "192"} and set(res["fit"]["SL"]) == {"48", "96", "192"}
    m = an.Model(res["fit"])
    assert m.step("compact", 3, 100) > 0 and m.step("stock", 3, 100) > 0
    planted = an.analyze(stub_bench(tmp_path, "compact_fast"))
    assert planted["Phi"] is None or planted["Phi"] >= (res["Phi"] or -1)
    assert all(r["read_ratio"] == 0.625 or abs(r["read_ratio"] - 0.625) < 1e-9 for r in planted["table"])


def synthetic_fit(kfun) -> dict:
    fit: dict = {}
    for lo, k in (("S32", 1.0), ("C48", 0.7)):
        fit[f"G_{lo}"] = {}
        for ln in be.L_GRID:
            bs = [b for b in be.B_GRID if b * ln <= be.MAX_TOKENS]
            fit[f"G_{lo}"][str(ln)] = {"B": bs, "s": [kfun(k, b, ln) for b in bs]}
    fit["SL"] = {str(ln): {"B": list(be.B_GRID), "s": [1e-4 * b for b in be.B_GRID]} for ln in be.L_GRID}
    fit.update(H={"B": list(be.B_GRID), "s": [2e-4] * len(be.B_GRID)}, t_other_s=0.04, t_other_profile_s=0.04)
    return fit


def test_model_interpolation_exact_for_bilinear_tables():
    f = synthetic_fit(lambda k, b, ln: k * (1e-4 + 1e-9 * b * ln))      # bilinear in (B, L) within each cell
    m = an.Model(f)
    for b, ln in ((3, 20_000), (12, 3_000), (5, 40_000)):
        want = 0.04 + 2 * (1e-4 + 1e-9 * b * ln) + 10 * 1e-4 * b
        assert abs(m.step("stock", b, ln) - want) < 1e-12
        assert abs(m.step("compact", b, ln) - (want - 2 * 0.3 * (1e-4 + 1e-9 * b * ln) + 2 * 2e-4)) < 1e-12
    # Past the last B of a row: linear extrapolation; below L = 1,024: the 1,024 row.
    assert abs(m.g("S32", 16, 65_536) - (1e-4 + 1e-9 * 16 * 65_536)) < 1e-12
    assert m.g("S32", 4, 300) == m.g("S32", 4, 1_024)


def test_checks_fail(tmp_path):
    bench = stub_bench(tmp_path)
    bad = copy.deepcopy(bench)
    bad["points"][0]["v1"] = {"identical": False, "max_abs": 0.01, "out_scale": 0.1}
    assert "V1" in an.analyze(bad)["failed"]
    bad = copy.deepcopy(bench)
    bad["points"][0]["v1"] = {"identical": True, "max_abs": 0.0, "out_scale": 0.0}   # A1 (R-101 F4): zero output
    assert "V1" in an.analyze(bad)["failed"]
    bad = copy.deepcopy(bench)
    bad["meta"]["stub"] = False
    bad["meta"]["rehearsal"] = True
    big = max(bad["points"], key=lambda p: p["tokens"])
    big["tokens"] = 10 ** 6
    big["G"]["S32"] = [1.0, 1.0, 1.1]
    assert "V3" in an.analyze(bad)["failed"]
    bad = copy.deepcopy(bench)
    bad["meta"]["stub"], bad["meta"]["rehearsal"] = False, True
    for p in bad["points"]:
        p["G"] = {lo: [1.0, 1.0, 1.0] for lo in ("S32", "C48")}
        p["SL"], p["H"] = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    bad["points"][0]["SL"] = [1.0, 1.0, 1.2]                       # A1 (R-101 F3): V3 on SL
    assert "V3" in an.analyze(bad)["failed"]
    bad = copy.deepcopy(bench)
    bad["points"].pop(0)                                           # a core point
    assert "V5" in an.analyze(bad)["failed"]
    ok = copy.deepcopy(bench)
    ok["points"].pop()                                             # the extra point: extrapolated, not INVALID
    res = an.analyze(ok)
    assert "V5" not in res["failed"] and res["checks"]["extrapolated"] == [(4, 192)]
    main = copy.deepcopy(bench)
    main["meta"].update(stub=False, rehearsal=False, gpu="NVIDIA A40", vllm="x")
    assert "V2" in an.analyze(main)["failed"]


def test_prefill_term():
    pf = {"a_s": 0.6, "b_s_per_1024": 0.04}
    assert abs(em.prefill_s([(2_048, 0)], pf) - (0.6 + 0.04)) < 1e-12
    assert abs(em.prefill_s([(1_024, 4_096)], pf) - 0.5 * (0.6 + 0.04 * 4.5)) < 1e-12


def test_poolsim_prefill_chunks_sum():
    rng = random.Random(3)
    for decode_only in (False, True):
        sim = ps.PoolSim(6_000, "compact", decode_only=decode_only)
        for _ in range(40):
            p, m = rng.randint(256, 40_000), rng.randint(16, 400)
            if sim.reservation(p + m, False) <= 6_000:
                sim.add(p, m)
        while (sim.waiting or sim.running) and sim.stats["steps"] < 200_000:
            d = sim.step()
            assert sum(n for n, _ in d["prefill_chunks"]) == d["prefill_tokens"]
            assert all(0 <= c0 for _, c0 in d["prefill_chunks"])


def test_e64_cell_unchanged():
    # The E-65 poolsim change must not move any E-64 number: one stored cell is recomputed exactly.
    import e64
    res = json.loads((HERE.parent / "gemma4_kv_capacity_model/results/e64_2026-11-10.json").read_text())
    row = next(r for r in res["cells"] if r["scenario"] == "U" and r["workload"] == "D3" and r["pool"] == 6_727
               and r["seed"] == 0)
    new = e64.run_cell("U", "D3", 6_727, 0)
    for k in ("decode_per_step_stock", "decode_per_step_compact", "r_A", "r_B"):
        assert new[k] == row[k], k


def test_model_cell_runs_and_labels():
    fit = synthetic_fit(lambda k, b, ln: k * (1e-4 + 1e-9 * b * ln))
    m = an.Model(fit)
    pf = {"stock": {"a_s": 0.6, "b_s_per_1024": 0.04}, "compact": {"a_s": 0.62, "b_s_per_1024": 0.042}}
    row = em.run_cell("D", "D3", 8_724, 0, m, pf, 0.04, horizon=3_000, warmup=500)
    assert row["r_M"] > 0 and row["r_step"] > 0
    assert em.label(1.25) == "GO" and em.label(1.2499) == "MID" and em.label(1.0999) == "KILL"
    rows = []
    for s in (0, 1):
        for i, pool in enumerate((1, 2, 3)):
            rows.append({"seed": s, "pool": pool, "included": True, "r_step": 1.3,
                         **{f"r_{a}": (1.3 if s == 0 else 1.2) for a in em.ARMS}})
    assert em.workload_label(rows, seeds=(0, 1))["label"] == "MID"
    assert em.workload_label([r for r in rows if r["seed"] == 0], seeds=(0,))["label"] == "GO"
