"""E-69 Mac tests (README Section 12, as amended by A1): the hash, V-RELOAD, the event log, the stats sum, the word,
the manifest and the stub session end to end."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_e69 as an  # noqa: E402
import e69_hooks as hk  # noqa: E402
import e69_stage as stg  # noqa: E402

BS = {0: 16, 1: 16}


def kv(nb=12, shape=(2, 3, 8), dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((nb, *shape), generator=g).to(dtype)


def test_hash_skips_null_and_sees_one_changed_value():
    t = kv()
    h = hk.block_hashes([t], [0, 3, 5])
    assert set(h) == {3, 5}
    t2 = t.clone()
    t2[5, 1, 2, 3] += 1
    h2 = hk.block_hashes([t2], [3, 5])
    assert h2[3] == h[3] and h2[5] != h[5]


def test_hash_of_a_moved_block_is_equal():
    a, b = kv(seed=1), kv(seed=2)
    b[7] = a[3]
    assert hk.block_hashes([a], [3])[3] == hk.block_hashes([b], [7])[7]


def test_hash_reads_only_the_view_not_the_padding():
    base = torch.zeros(12, 2, 3, 10, dtype=torch.bfloat16)
    view = base[..., :8]                                   # content 8 of 10 per row; the rest is padding
    view.copy_(kv())
    h = hk.block_hashes([view], [4])
    base[4, :, :, 8:] = 7.0                                # change only the padding
    assert hk.block_hashes([view], [4]) == h


def test_snapshot_limit_keeps_only_completed_blocks():
    s = hk.snapshot({0: [kv()]}, {0: [1, 2, 3]}, tokens=40, block_size=BS, limit=33)
    assert set(s.groups[0]) == {0, 1}                       # block 2 ends at 48 > 33 (in flight)


def test_reload_compare_match_mismatch_range_and_same_id():
    glob, slid = kv(seed=3), kv(seed=4)
    before = hk.snapshot({0: [glob], 1: [slid]}, {0: [1, 2, 3], 1: [0, 5, 6]}, 48, BS)
    g2, s2 = kv(seed=9), kv(seed=10)
    g2[7], g2[8] = glob[1], glob[2]
    s2[9] = slid[5]
    ids_after = {0: [7, 8, 11], 1: [0, 9, 10]}
    r = hk.compare(before, hk.snapshot({0: [g2], 1: [s2]}, ids_after, 32, BS), 32, BS, {1})
    assert r["checked"] == 3 and r["mismatch"] == 0 and r["valid"]          # global 0, 1 and sliding 1
    s2[9, 0, 0, 0] += 1
    r = hk.compare(before, hk.snapshot({0: [g2], 1: [s2]}, ids_after, 32, BS), 32, BS, {1})
    assert r["mismatch"] == 1
    same = hk.compare(before, before, 32, BS, {1})                         # same ids: stale blocks are not counted
    assert same["checked"] == 0 and same["same_id"] == 3 and not same["valid"]


def test_reload_bookkeeping_caps_skips_and_plants():
    rc = hk.ReloadCheck(max_checks=2)
    snap_a = lambda: hk.Snapshot(tokens=32, groups={0: {0: (1, 5)}, 1: {0: (2, 6)}})  # noqa: E731
    snap_b = lambda: hk.Snapshot(tokens=32, groups={0: {0: (3, 5)}, 1: {0: (4, 6)}})  # noqa: E731
    for r in ("a", "b", "c"):
        rc.on_preempt(r, snap_a)
    assert set(rc.pending) == {"a", "b"}
    rc.on_loaded("c", 16, snap_b, BS, {1})
    rc.on_loaded("a", 16, snap_b, BS, {1})
    planted = []
    rc.on_loaded("b", 16, lambda: hk.Snapshot(32, {0: {0: (3, 99)}, 1: {0: (4, 6)}}), BS, {1},
                 corrupt_fn=lambda: planted.append(1))
    s = rc.summary()
    assert planted == [1] and s["skipped_loads"] == 1 and s["checked_reloads"] == 2 and s["mismatched_blocks"] == 1


def test_event_log_decode_coverage():
    log = hk.EventLog()
    log.add(8_192, 0, False)                               # a new request: not counted
    log.add(8_192 + 600, 8_192 + 576, True)                # c = 576 / 600
    log.add(8_192 + 1_000, 8_192 - 16, True)               # prompt only: c = 0
    log.add(8_192 + 100, 8_192 + 96, True)                 # under 512 generated: not a decode event
    c = log.decode_coverage(8_192)
    assert c["preempted_events"] == 3 and c["decode_events"] == 2
    assert abs(c["max_c"] - 0.96) < 1e-9 and c["min_c"] == 0.0


def test_stats_sum():
    ss = hk.StatsSum()
    p = SimpleNamespace(requests=1, queries=100, hits=10, preempted_requests=2, preempted_queries=200,
                        preempted_hits=180)
    ss.add(SimpleNamespace(connector_prefix_cache_stats=p,
                           kv_connector_stats={"vllm:kv_offload_load_bytes": [1e9, 1e9],
                                               "vllm:kv_offload_load_time": 0.5}))
    ss.add(None)
    s = ss.summary()
    assert s["prefix"]["preempted_hits"] == 180 and s["none_steps"] == 1 and abs(s["load_gb_per_s"] - 4.0) < 1e-9


def test_stats_sum_reads_the_offload_payload():
    """A2: the real payload of the pinned offloading/metrics.py (`to_dict` of OffloadingConnectorStats)."""
    ss = hk.StatsSum()
    payload = {"types": {"vllm:kv_offload_load_bytes": "counter", "vllm:kv_offload_load_time": "counter",
                         "vllm:kv_offload_load_size": "histogram", "vllm:kv_offload_cpu_cache_usage_perc": "gauge"},
               "data": {"vllm:kv_offload_load_bytes": {(): 3e9}, "vllm:kv_offload_load_time": {(): 0.5},
                        "vllm:kv_offload_load_size": {(): [1e9, 2e9]},
                        "vllm:kv_offload_cpu_cache_usage_perc": {(): 0.3}},
               "info": None}
    for _ in range(2):
        ss.add(SimpleNamespace(connector_prefix_cache_stats=None, kv_connector_stats=payload))
    s = ss.summary()
    assert s["kv"]["vllm:kv_offload_load_bytes"] == 6e9 and s["kv"]["vllm:kv_offload_load_size_count"] == 4
    assert s["kv"]["vllm:kv_offload_cpu_cache_usage_perc"] == 0.3 and abs(s["load_gb_per_s"] - 6.0) < 1e-9


def arm(status="ok", preempt=20, reloads=12, mismatch=0, med=0.98, dec=12, compact=True, eq=None, eq_load=0.0):
    return {"status": status, "gpu": "NVIDIA A100 80GB PCIe", "manifest_ok": True, "layout": "LBHNC",
            "prompt_len": 8_192, "eq_tokens": eq or [[1, 2, 3]], "eq_load_bytes": eq_load, "stats_preemptions": preempt,
            "counters": {"preemptions": preempt, "fix_reads": 5 if compact else 0},
            "stats": {"none_steps": 0, "steps": 100, "kv": {"vllm:kv_offload_load_bytes": 1.0}, "load_gb_per_s": 9.0},
            "coverage": {"median_c": med, "decode_events": dec},
            "reload": {"checked_reloads": reloads, "mismatched_blocks": mismatch, "same_id_share": 0.0}}


def arms(**cc):
    return {"C-C": arm(**cc), "S-C": arm(compact=False), "S-R": None, "C-R": None, "C-C0": None, "C-CX": None,
            "C-C64": None}


def test_word_rule():
    assert an.word(arms())["word"] == "CONNECTOR-WORKS (A3)"
    assert an.word(arms(med=0.05))["word"] == "CONNECTOR-PROMPT-ONLY (A3)"
    assert an.word(arms(med=0.5))["word"] == "CONNECTOR-PARTIAL (A3)"
    assert an.word(arms(mismatch=1))["word"] == "CONNECTOR-FAILS (A3)"
    assert an.word(arms(preempt=5))["word"] == "INVALID (V4)"
    assert an.word(arms(dec=4))["word"] == "INVALID (V4)"
    assert an.word(arms(reloads=0))["word"] == "INVALID (zero checked reloads)"
    assert an.word(arms(eq_load=5.0))["word"] == "INVALID (V5)"
    a = arms()
    a["C-C"]["eq_tokens"] = [[9]]
    assert an.word(a)["word"] == "INVALID (V5)"
    a = arms()
    a["S-C"] = None
    assert an.word(a)["word"].startswith("INVALID (S-C")
    a = arms()
    a["C-C"] = {"status": "failed", "tries": 2, "error": "AssertionError", "traceback": "kv_offload/cpu/gpu_worker.py"}
    assert an.word(a)["word"] == "CONNECTOR-FAILS (A3)"
    a["C-C"] = {"status": "failed", "tries": 2, "error": "RuntimeError: /dev/shm too small", "traceback": ""}
    assert an.word(a)["word"].startswith("INVALID (ENV")
    a["C-C"] = {"status": "failed", "tries": 1, "error": "x", "traceback": "kv_offload"}
    assert an.word(a)["word"].startswith("INVALID (C-C not complete")


def test_rehearsal_verify():
    a = arms()
    a["C-C0"], a["C-CX"] = arm(med=0.0), arm(mismatch=2)
    a["C-C64"] = {"status": "ok", "start_only": True, "engine_seconds": 120.0}
    v = an.word(a, rehearsal=False)
    assert v["word"] == "INVALID (V1)" or v["word"] == "CONNECTOR-WORKS (A3)"
    for r in a.values():
        if r:
            r["gpu"] = "NVIDIA L40S"
    out = an.word(a, rehearsal=True)
    assert out["word"] == "REHEARSAL | CONNECTOR-WORKS (A3)" and out["verify"]["all_pass"]
    a["C-C64"]["engine_seconds"] = 900.0
    assert not an.word(a, rehearsal=True)["verify"]["all_pass"]


def mounted_tree(root: Path) -> Path:
    for local, remote in stg.mounts():
        dst = root / remote.removeprefix(stg.CODE + "/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())
    return root


def test_manifest_is_current_and_every_file_exists():
    for local, _ in stg.mounts():
        assert local.exists(), local
    assert json.loads(stg.MANIFEST_LOCAL.read_text()) == stg.make_manifest()


def test_stub_session_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(stg.ARM_MIN, "reh", 0)
    code = mounted_tree(tmp_path / "code")
    out = stg.session(str(code), str(tmp_path / "res"), "e69-stub", "reh", stub=True)
    rec = out["record"]
    assert rec["manifest_ok"] and rec["ok"], rec
    assert all(rec["arms"][a]["exit"] == 0 for a in stg.ARM_ORDER["reh"]), rec["arms"]
    assert "REHEARSAL | CONNECTOR-WORKS" in rec["decision"] and "all_pass: True" in rec["decision"], rec["decision"]
    top = stg.unpack(out["tar"], tmp_path / "copy")
    assert (top / "summary.json").exists() and (top / "C-C" / "run_e69.json").exists()
    again = stg.session(str(code), str(tmp_path / "res"), "e69-stub", "reh", stub=True)
    assert again["record"]["restart"]                                  # the restart guard


def test_wrong_code_stops_before_the_session(tmp_path):
    code = mounted_tree(tmp_path / "code")
    (code / "gemma4_kv_offload" / "run_e69.py").write_text("# changed\n")
    out = stg.session(str(code), str(tmp_path / "res"), "e69-bad", "reh", stub=True)
    assert not out["record"]["manifest_ok"] and not out["record"]["ok"] and not out["record"]["arms"]


def test_cost_caps():
    assert 5.4 < stg.cost_cap("main") < 5.7 and 2.7 < stg.cost_cap("reh") < 2.9


def test_a3_recomputed_range_is_counted_apart():
    """A3: after a reload at hit 32 of a request preempted at 47, the tail [32, 47) is recomputed on the GPU while the
    CPU keeps the first copy of block 2 (tokens 32 to 48 were full at the first preemption only if n_pre >= 48; here a
    later block). A difference inside [prev hit, prev n_pre) is reported apart; one outside still decides."""
    rc = hk.ReloadCheck(max_checks=8)
    s1 = hk.Snapshot(64, {0: {0: (1, 10), 1: (2, 11), 2: (3, 12), 3: (4, 13)}, 1: {2: (5, 20), 3: (6, 21)}})
    rc.on_preempt("r", lambda: s1, 64)
    rc.on_loaded("r", 32, lambda: hk.Snapshot(32, {0: {0: (7, 10), 1: (8, 11)}, 1: {1: (9, 30)}}), BS, {1})
    # second preemption at 96; blocks 2, 3 (tokens 32 to 64) were recomputed after the first reload
    s2 = hk.Snapshot(96, {0: {i: (20 + i, 100 + i) for i in range(6)}, 1: {4: (40, 50), 5: (41, 51)}})
    rc.on_preempt("r", lambda: s2, 96)
    # the CPU copy of block 2 is the first version (hash 12, seen at the first preemption): it passes, counted apart
    after = hk.Snapshot(96, {0: {0: (60, 100), 1: (61, 101), 2: (62, 12), 3: (63, 103), 4: (64, 104), 5: (65, 105)},
                             1: {4: (70, 50), 5: (71, 51)}})
    rc.on_loaded("r", 96, lambda: after, BS, {1})
    last = rc.results[-1]
    assert last["recomputed_range"] == (32, 64)
    assert last["recomputed"] == 2 and last["recomputed_first_version"] == 1 and last["mismatch"] == 0
    assert rc.summary()["a1_strict_mismatch"] == 1 and rc.summary()["mismatched_blocks"] == 0
    after.groups[0][2] = (62, 102)                    # the range is now (96, 96): restore block 2, change block 5
    after.groups[0][5] = (65, 7)                                       # a change outside the range still counts
    rc.on_preempt("r", lambda: s2, 96)
    rc.on_loaded("r", 96, lambda: after, BS, {1})
    assert rc.results[-1]["mismatch"] == 1


def test_a3_third_value_in_the_range_decides():
    """R-112 F1, F4: inside the recomputed range, a value that is neither the GPU copy nor the first version is a
    mismatch; so is a block with no first hash (fail safe)."""
    rc = hk.ReloadCheck(max_checks=8)
    rc.on_preempt("r", lambda: hk.Snapshot(64, {0: {0: (1, 10), 1: (2, 11), 2: (3, 12), 3: (4, 13)}, 1: {2: (5, 20)}}),
                  64)
    rc.on_loaded("r", 32, lambda: hk.Snapshot(32, {0: {0: (7, 10), 1: (8, 11)}, 1: {1: (9, 30)}}), BS, {1})
    rc.on_preempt("r", lambda: hk.Snapshot(96, {0: {i: (20 + i, 100 + i) for i in range(6)}, 1: {5: (41, 51)}}), 96)
    after = hk.Snapshot(96, {0: {0: (60, 100), 1: (61, 101), 2: (62, 777), 3: (63, 103), 4: (64, 104), 5: (65, 105)},
                             1: {5: (71, 51)}})
    rc.on_loaded("r", 96, lambda: after, BS, {1})
    r = rc.results[-1]
    assert r["recomputed_third"] == 1 and r["mismatch"] == 1 and r["bad"] == [[0, 2, "in_range"]]
    rc.first["r"].pop((0, 3))                                          # no first hash for block 3
    rc.on_preempt("r", lambda: hk.Snapshot(96, {0: {i: (20 + i, 100 + i) for i in range(6)}, 1: {5: (41, 51)}}), 96)
    rc.last["r"] = (32, 64)
    after.groups[0][2], after.groups[0][3] = (62, 102), (63, 555)
    rc.on_loaded("r", 96, lambda: after, BS, {1})
    assert rc.results[-1]["mismatch"] == 1 and rc.results[-1]["recomputed_third"] == 1


def test_a3_unchecked_load_moves_the_range():
    rc = hk.ReloadCheck(max_checks=0)
    rc.on_preempt("r", lambda: None, 80)
    rc.note_load("r", 64)
    assert rc.last["r"] == (64, 80) and rc.skipped == 0 and not rc.results
