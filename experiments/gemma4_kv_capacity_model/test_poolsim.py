"""E-63 tests: V1 (tie to E-57), V2 (conservation), V3 (determinism), V4 (limit), and rule unit tests."""
import random

import poolsim as ps

E57_BLOCKS = 6_143


def test_v1_e57_footprints():
    st, co = ps.PoolSim(E57_BLOCKS, "stock"), ps.PoolSim(E57_BLOCKS, "compact")
    assert st.steady(32_768) == 1_349 and co.steady(32_768) == 1_008
    assert st.steady(65_536) == 2_373 and abs(co.steady(65_536) - 1_690) <= 1


def test_v1_e57_batches_closed_form():
    assert ps.closed_form_batch(E57_BLOCKS, "stock", 32_768) == 3
    assert ps.closed_form_batch(E57_BLOCKS, "compact", 32_768) == 4
    assert ps.closed_form_batch(E57_BLOCKS, "stock", 65_536) == 2
    assert ps.closed_form_batch(E57_BLOCKS, "compact", 65_536) == 2


def test_v1_e57_batches_simulated():
    for layout, prompt, b in (("stock", 32_768, 3), ("compact", 32_768, 4), ("stock", 65_536, 2),
                              ("compact", 65_536, 2)):
        out = ps.run_saturated(E57_BLOCKS, layout, [(prompt, 256)] * 8, max_num_seqs=16)
        assert out["max_decoding"] == b, (layout, prompt, out)


E57_PEAKS = {("stock", 32_768): 2_602, ("compact", 32_768): 2_262, ("stock", 65_536): 3_626,
             ("compact", 65_536): 2_944}


def test_v1_e57_prefill_peaks():
    # A1 F3: one request of L - 64 prompt tokens (the reviewer's probe); the measured E-57 peaks within 6 blocks.
    for (layout, L), peak in E57_PEAKS.items():
        out = ps.run_saturated(E57_BLOCKS, layout, [(L - 64, 64)])
        assert abs(out["peak_single"] - peak) <= 6, (layout, L, out["peak_single"], peak)


def test_double_preemption_prompt():
    # A1 B2: a request preempted twice after decoding must prefill base + output, not base + 2 x output.
    sim = ps.PoolSim(3_000, "stock")
    for _ in range(5):          # found by a small search: one request is preempted twice
        sim.add(500, 40_000)
    seen = 0
    while (sim.waiting or sim.running) and sim.stats["steps"] < 400_000:
        sim.step()
        for r in sim.waiting:
            assert r.p == r.base + r.out          # a waiting victim prefills its prompt plus its output so far
        for r in sim.waiting + sim.running:
            assert r.base <= r.p <= r.base + r.out <= r.base + r.m or r.out == 0
        seen = max([seen] + [r.npre for r in sim.waiting + sim.running + sim.done])
    assert seen >= 2 and len(sim.done) == 5


def test_stream_same_window_and_bound_share():
    import itertools
    import random

    def gen():
        rng = random.Random(5)
        return ((rng.randint(2_000, 60_000), rng.randint(16, 500)) for _ in itertools.count())
    a = ps.run_stream(8_000, "stock", gen, horizon=3_000, warmup=500)
    b = ps.run_stream(8_000, "stock", gen, horizon=3_000, warmup=500)
    assert a == b
    assert 0.0 <= a["pool_bound_share"] <= 1.0 and a["decode_per_step"] > 0


def test_sync_scheduling_no_self_preemption():
    # A2: with one chunk in flight (cap 193), a lone request must not outgrow its own reservation.
    out = ps.run_saturated(2_000, "stock", [(16_384, 256)] * 3, cap_batches=1, max_steps=50_000)
    assert out["finished"] == 3 and out["preemptions"] == 0


def test_stream_watermark_no_starvation():
    # A3: with a watermark, a request that fits the pool but not pool - watermark must be skipped, not wait forever.
    import itertools

    big = (62_000, 300)
    pool = ps.PoolSim(4_000, "stock").reservation(sum(big)) + 20   # fits the pool, not pool - the 1% watermark

    def gen():   # every other request is the big one
        return (big if i % 2 == 0 else (4_000, 300) for i in itertools.count())
    out = ps.run_stream(pool, "stock", gen, horizon=2_000, warmup=0, watermark=0.01)
    assert out["skipped"] > 0 and out["decode_per_step"] > 0


def test_v2_conservation_mixed():
    rng = random.Random(1)
    reqs = [(rng.randint(256, 70_000), rng.randint(16, 600)) for _ in range(60)]
    for layout in ("stock", "compact"):
        sim = ps.PoolSim(6_000, layout)
        for p, m in reqs:
            sim.add(p, m)
        while sim.waiting or sim.running:
            sim.step()
            assert sim.free() >= 0
        assert len(sim.done) == len(reqs)
        assert sim.stats["decoded"] == sum(m for _, m in reqs)


def test_v3_determinism():
    rng = random.Random(2)
    reqs = [(rng.randint(256, 40_000), rng.randint(16, 400)) for _ in range(40)]
    a = ps.run_saturated(8_000, "compact", reqs)
    b = ps.run_saturated(8_000, "compact", reqs)
    assert a == b


def test_v4_limit_ratio():
    for prompt in (16_384, 32_768, 65_536, 131_072):
        st, co = ps.PoolSim(10**6, "stock"), ps.PoolSim(10**6, "compact")
        foot = st.steady(prompt) / co.steady(prompt)
        r = ps.closed_form_batch(10**6, "compact", prompt) / ps.closed_form_batch(10**6, "stock", prompt)
        assert abs(r / foot - 1) < 0.02


def test_reservation_and_hol():
    sim = ps.PoolSim(3_000, "stock")
    assert sim.reservation(32_768) == 1_024 + 5 * 321
    sim.add(32_768, 4)      # the head does not fit next to the first big one
    sim.add(32_768, 4)
    sim.add(1_000, 4)       # a small one behind it would fit
    for _ in range(40):
        sim.step()
    assert sim.stats["hol_stops"] > 0


def test_preemption_recompute():
    # Two requests that fit at admission but outgrow a tight pool during decode.
    # A recomputed request must still fit the empty pool: 2 g + 650 > pool >= g + 1,605 for its g global blocks.
    sim = ps.PoolSim(3_000, "stock")
    sim.add(1_000, 40_000)
    sim.add(1_000, 40_000)
    while (sim.waiting or sim.running) and sim.stats["steps"] < 200_000:
        sim.step()
        assert sim.free() >= 0
    assert sim.stats["preemptions"] > 0 and len(sim.done) == 2
    assert sim.stats["decoded"] == 80_000


# --- E-64 -----------------------------------------------------------------------------------------------------------

def test_e64_decode_only_no_prefill_and_batch():
    sim = ps.PoolSim(8_724, "compact", decode_only=True)
    b = 8_724 // sim.steady(32_768)
    for _ in range(b + 2):
        sim.add(32_768, 64)
    first = sim.step()                      # admissions only: the KV arrives computed
    assert first["prefill_tokens"] == 0 and first["decoding"] == 0 and len(sim.running) == b
    second = sim.step()
    assert second["decoding"] == b and second["prefill_tokens"] == 0
    assert all(c == 32_768 for c in second["decode_ctx"])


def test_e64_decode_only_conservation():
    rng = random.Random(9)
    reqs = [(rng.randint(256, 70_000), rng.randint(16, 3_000)) for _ in range(60)]
    sim = ps.PoolSim(6_000, "stock", decode_only=True)
    for p, m in reqs:
        if sim.reservation(p + m) <= 6_000:
            sim.add(p, m)
    n = len(sim.waiting)
    while (sim.waiting or sim.running) and sim.stats["steps"] < 500_000:
        sim.step()
        assert sim.free() >= 0
    assert len(sim.done) == n


def test_e64_default_path_reproduces_e63():
    # The E-64 changes must not move any E-63 number: one stored MIX cell is recomputed.
    import json
    from pathlib import Path

    import sweep
    res = json.loads((Path(__file__).parent / "results/e63_2026-11-10.json").read_text())
    row = next(r for r in res["mix"] if r["workload"] == "M1" and r["pool"] == 11_314)
    out = ps.run_stream(11_314, "stock", sweep.gen_factory("M1"), sweep.HORIZON, sweep.WARMUP)
    assert out["decode_per_step"] == row["decode_per_step_stock"]
    assert out["decode_per_second"] == row["decode_per_second_stock"]


def test_e64_time_model_v7():
    import e64
    v = e64.checks()
    assert v["V7"] and v["V8"] and v["V8b"]


def test_e64_decode_node_local_reprefill():
    # A1 (R-99 B1): a request preempted on a decode node prefills again locally through the token budget.
    sim = ps.PoolSim(3_000, "stock", decode_only=True)
    for _ in range(5):
        sim.add(500, 40_000)
    pre_tokens, saw_pre = 0, False
    while (sim.waiting or sim.running) and sim.stats["steps"] < 400_000:
        d = sim.step()
        saw_pre |= sim.stats["preemptions"] > 0
        if saw_pre:
            pre_tokens += d["prefill_tokens"]
        assert sim.free() >= 0
    assert sim.stats["preemptions"] > 0 and pre_tokens > 0 and len(sim.done) == 5
