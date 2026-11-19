"""E-67 checks V2 to V5 (README_E67.md Section 6). V1 is `test_e66.py`; V6 is `ref_e67.py`."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import poolsim as ps  # noqa: E402

BIG = 1e15


def arrived(layout: str, base: int, out: int, prompt_only: bool = True) -> tuple[ps.PoolSim, ps.Req]:
    """A request that arrived from a prefill node, decoded `out` tokens, and was preempted."""
    s = ps.PoolSim(100_000, layout, decode_only=True, offload=(prompt_only, BIG))
    r = ps.Req(rid=0, base=base, m=10**6, p=base)
    s.offload_arrive(r)
    for c in range(base, base + out):
        s.offload_store(r, c, c + 1)
    r.npre, r.out, r.p = 1, out, base + out
    return s, r


def pressure(layout: str, offload) -> ps.PoolSim:
    s = ps.PoolSim(3_000, layout, decode_only=True, offload=offload)
    for _ in range(5):
        s.add(500, 40_000)
    for _ in range(400_000):
        if not (s.waiting or s.running):
            break
        s.step()
        assert s.free() >= 0, "V2: held blocks exceed the pool"
        assert s.cpu_blocks <= s.cpu_cap, "V3: the CPU tier is over its capacity"
        assert sum(q.ob for q in s.cpu.values()) == s.cpu_blocks, "V3: the block counts do not sum"
    return s


def test_v2_v3_conservation_and_finish():
    for lay in ("stock", "compact"):
        for off in ((True, 200e9), (False, 200e9), (False, 2e9)):
            s = pressure(lay, off)
            assert len(s.done) == 5 and not s.running and not s.waiting
            assert s.stats["preemptions"] > 0


def test_v3_eviction_oldest_first_and_finish_renews():
    s = ps.PoolSim(100_000, "stock", decode_only=True, offload=(True, 100 * ps.BLOCK_BYTES))
    a, b, c = (ps.Req(rid=i, base=320, m=9, p=320) for i in range(3))     # 10 + 5 x 20 = 110 blocks without a cap
    s.cpu_cap = 250
    s.offload_arrive(a)
    s.offload_arrive(b)
    s.cpu_renew(a)                                                       # a finished: b is now the oldest
    s.offload_arrive(c)
    assert b.rid not in s.cpu and a.rid in s.cpu and c.rid in s.cpu
    assert b.ob == b.og == b.os1 == 0 and b.ohole
    rank = b.orank                                                       # A1 B1: a re-store keeps the old rank
    s.offload_arrive(b)
    assert b.orank == rank and b.rid not in s.cpu and a.rid in s.cpu and c.rid in s.cpu


def test_v4_prompt_only_alignment_on_arrival():
    # Stock: a prompt with base mod 32 <= 14 hits at the aligned point; a larger remainder does not.
    for base, want in ((32_000, 32_000), (32_014, 32_000), (32_015, 0), (32_031, 0)):
        s, r = arrived("stock", base, 100)
        assert s.offload_hit(r) == want, base
    # Compact: base mod 48 <= 14.
    assert arrived("compact", 48_014, 100)[0].offload_hit(arrived("compact", 48_014, 100)[1]) == 48_000
    s, r = arrived("compact", 48_015, 100)
    assert s.offload_hit(r) == 0


def test_v4_local_prefill_stores_all_sliding_blocks():
    s, r = arrived("stock", 32_031, 100)
    assert s.offload_hit(r) == 0
    for c0 in range(0, 32_031, 2_048):                                   # a miss: the request prefills locally
        s.offload_store(r, c0, min(c0 + 2_048, r.p))
    assert r.os0 == 0
    assert s.offload_hit(r) == 32_000


def test_v4_short_prompt_needs_all_blocks_from_zero():
    s, r = arrived("stock", 320, 100)
    assert r.os0 == 0 and s.offload_hit(r) == 320


def test_v4_no_output_yet_hits_at_p_minus_one():
    s, r = arrived("stock", 32_000, 0)                                   # A1 F2: h = p - 1, not rounded down
    assert s.offload_hit(r) == 31_999


def test_v4_not_preempted_or_absent_no_hit():
    s, r = arrived("stock", 32_000, 100)
    r.npre = 0
    assert s.offload_hit(r) == 0
    r.npre = 1
    s.cpu.pop(r.rid)
    assert s.offload_hit(r) == 0


def test_v5_store_all_hits_after_the_window():
    for lay, gb in (("stock", 32), ("compact", 48)):
        for base in (32_031, 48_015, 40_001):
            for out in (2_000, 3_333):
                s, r = arrived(lay, base, out, prompt_only=False)
                assert s.offload_hit(r) == min(r.p - 1, r.p // gb * gb), (lay, base, out)


def test_run_cell_smoke():
    import analyze_e65 as an
    import e66
    import e67
    fits, pf = e66.load()
    models = {t: an.Model(f) for t, f in fits.items()}
    row = e67.run_cell("all200", "M2", 8_724, 0, models, fits, pf, horizon=4_000, warmup=500)
    assert row["r_r0625_M_L20"] > 0 and row["link_use_L20_stock"] >= 0
    assert row["r_r0625_M_Linf"] >= row["r_r0625_M_L10"] * 0.5


def test_load_blocks_and_hole():
    s = ps.PoolSim(100_000, "stock", decode_only=True, offload=(False, BIG))
    r = ps.Req(rid=0, base=1_024, m=9, p=1_024)
    s.offload_arrive(r)
    r.ohole, before = True, r.ob
    s.offload_store(r, 1_024, 1_040)
    assert r.ob == before
    s.offload_store(r, 0, 16)
    assert not r.ohole
