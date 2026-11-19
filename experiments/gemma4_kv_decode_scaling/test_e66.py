"""E-66 tests (README_E66.md Section 6): the default path is unchanged, the queue accounting, the hit rules."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import analyze_e65 as an  # noqa: E402
import e64  # noqa: E402
import e65_model as em  # noqa: E402
import e66  # noqa: E402
import poolsim as ps  # noqa: E402


def test_v1_none_mode_reproduces_stored_bracket_cell():
    res = json.loads((HERE / "results/bracket/e65_bracket_r0.625.json").read_text())
    row = next(r for r in res["cells"] if r["scenario"] == "D" and r["workload"] == "M2" and r["pool"] == 8_724
               and r["seed"] == 0)
    fit = json.loads((HERE / "results/bracket/fit_r0.625.json").read_text())["fit"]
    pf = json.loads((HERE / "prefill_fit.json").read_text())
    new = em.run_cell("D", "M2", 8_724, 0, an.Model(fit), pf, fit["t_other_profile_s"])
    for k in ("decode_per_step_stock", "decode_per_step_compact", "preemptions_stock", "preemptions_compact"):
        assert new[k] == row[k], k                     # the simulation is exact
    for k in ("r_M", "r_M_pf_equal", "r_M_gemm"):      # Model.step caches per 64-token bucket: order moves the 6th digit
        assert abs(new[k] - row[k]) < 1e-4, k


def run_to_end(mode: str, decode_only: bool, check=None) -> ps.PoolSim:
    rng = random.Random(4)
    sim = ps.PoolSim(5_000, "compact", decode_only=decode_only, prefix_cache=mode)
    for _ in range(40):
        p, m = rng.randint(2_000, 40_000), rng.randint(200, 6_000)
        if sim.reservation(p + m, False) <= 5_000:
            sim.add(p, m)
    n = len(sim.waiting)
    while (sim.waiting or sim.running) and sim.stats["steps"] < 500_000:
        sim.step()
        assert sim.free() >= 0
        if check:
            check(sim)
    assert len(sim.done) == n
    return sim


def run_pressure(mode: str, decode_only: bool, layout: str = "stock") -> ps.PoolSim:
    """Five long-output requests in a small pool: preemptions happen (the E-63 double-preemption workload)."""
    sim = ps.PoolSim(3_000, layout, decode_only=decode_only, prefix_cache=mode)
    for _ in range(5):
        sim.add(500, 40_000)
    while (sim.waiting or sim.running) and sim.stats["steps"] < 500_000:
        sim.step()
        assert sim.free() >= 0
    assert len(sim.done) == 5 and sim.stats["preemptions"] > 0
    return sim


def test_v2_conservation_all_modes():
    for mode in ("engine", "global", "perfect"):
        for d in (False, True):
            sim = run_to_end(mode, d)
            assert sim.stats["decoded"] == sum(r.m for r in sim.done)
            for lay in ("stock", "compact"):
                sim = run_pressure(mode, d, lay)
                assert sim.stats["decoded"] == 5 * 40_000


def test_v4_perfect_hits_every_readmission():
    for d in (False, True):
        sim = run_pressure("perfect", d)
        assert sim.stats["hits"] == sim.stats["readmits"] == sim.stats["preemptions"]


def test_v4_hit_rules():
    sim = ps.PoolSim(5_000, "stock", prefix_cache="engine")
    r = ps.Req(rid=0, base=1_000, m=10, p=1_100, qstart=100, nhash=400, gl=35, cpre=1_099)
    sim.popped = 100
    assert sim.evicted(r) == 0 and sim.hit_tokens(r) == (1_099 // 32) * 32
    sim.popped = 101
    assert sim.evicted(r) == 1 and sim.hit_tokens(r) == 0                # one reused block removes the hit
    sim.prefix_cache = "global"
    sim.popped = 100 + 365                                                # all 365 sliding blocks reused, none global
    assert sim.hit_tokens(r) == (1_099 // 32) * 32
    sim.popped = 100 + 375                                                # 10 global blocks reused
    assert sim.hit_tokens(r) == 25 * 32
    sim.prefix_cache = "perfect"
    assert sim.hit_tokens(r) == (1_099 // 32) * 32
    sim.prefix_cache = "none"
    assert sim.hit_tokens(r) == 0


def test_v5_engine_hit_only_before_any_reuse():
    def check(sim):
        for q in sim.waiting:
            if q.qstart >= 0 and sim.evicted(q) > 0:
                assert sim.hit_tokens(q) == 0
    run_to_end("engine", True, check)


def test_v3_queue_head_never_passes_the_tail():
    def check(sim):                                    # A1 (R-103 B1): ranges inside the queue, with no overlap
        ranges = sorted((q.qstart, q.qstart + q.nhash) for q in sim.waiting if q.qstart >= 0)
        for (_, e0), (s1, _) in zip(ranges, ranges[1:]):
            assert e0 <= s1
        if ranges:
            assert ranges[-1][1] <= sim.popped + sim.free()
    rng = random.Random(11)
    for d in (False, True):
        sim = ps.PoolSim(3_000, "compact", decode_only=d, prefix_cache="global")
        for _ in range(12):
            sim.add(rng.randint(400, 3_000), rng.randint(5_000, 30_000))
        while (sim.waiting or sim.running) and sim.stats["steps"] < 500_000:
            sim.step()
            check(sim)
        assert sim.stats["preemptions"] > 0


def test_v6_reference_hit_search():
    import refsim
    r = refsim.Ref(10_000, 32, True)
    r.grow(1, 0, 2_048, 2_048)                         # one chunk: every sliding block is held
    r.grow(1, 2_048, 4_096, 2_048)
    for c in range(4_096, 4_096 + 64):
        r.grow(1, c, c + 1, 1)                         # decode: only the window stays
    c = 4_096 + 64
    assert r.find_hit(1, c + 1) == c // 32 * 32        # nothing reused: the aligned full prefix
    r.free_request(1)
    assert r.find_hit(1, c + 1) == c // 32 * 32        # freed but not reused: still a hit
    while r.free:                                      # reuse every block
        r.alloc()
    assert r.find_hit(1, c + 1) == 0


def test_model_cell_runs():
    fits, pf = e66.load()
    models = {t: an.Model(f) for t, f in fits.items()}
    row = e66.run_cell("engine", "D", "D3", 8_724, 0, models, fits, pf, horizon=4_000, warmup=500)
    assert row["r_r0625_M"] > 0 and row["r_r1_M"] > 0 and row["readmits_stock"] >= row["hits_stock"]
    assert e64.POOLS and e66.labels_for
