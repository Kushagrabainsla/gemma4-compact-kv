"""E-71 Mac tests (A1): the request sequence equals the model's, the engine settings fit the pool, the word rule, the
watchdogs fit the arm limits, the manifest and the cost caps, and the stub session end to end."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_capacity_model"), str(EXP / "gemma4_kv_decode_scaling")]
import analyze_e71 as an  # noqa: E402
import e71_stage as stg  # noqa: E402
import run_e71  # noqa: E402

PRED = json.loads((HERE / "prediction_interval.json").read_text())


def test_sequence_equals_the_model_sequence():
    import e64
    import e71_predict as ep
    import poolsim as ps
    e64.draw = ep.draw
    g = e64.gen_factory("D3H", 0)
    seen = []
    orig = ps.PoolSim.add

    def add(self, p, m):
        seen.append((p, m))
        return orig(self, p, m)
    ps.PoolSim.add = add
    try:
        ps.run_stream(6_143, "stock", lambda: ((p, m) for p, m in g() if p + m <= PRED["max_len"]), 2_000, 100,
                      decode_only=True, offload=(False, 200e9), keep_waiting=3)
    finally:
        ps.PoolSim.add = orig
    assert run_e71.requests(len(seen), run_e71.MAIN["max_model_len"]) == seen and len(seen) > 5
    assert all(p + m <= run_e71.REH["max_model_len"] for p, m in run_e71.requests(50, run_e71.REH["max_model_len"]))


def test_engine_settings():
    m = run_e71.MAIN
    d = run_e71.engine_kwargs("d", m)
    kids = d["kv_transfer_config"]["kv_connector_extra_config"]["connectors"]
    assert [k["kv_connector"] for k in kids] == ["NixlConnector", "OffloadingConnector"]
    assert d["prefix_cache_retention_interval"] is None and d["disable_log_stats"] is False
    assert d["kv_cache_memory_bytes"] == 15 * 2**30 and m["max_model_len"] == PRED["max_len"]
    # R-118 B1: the pin's start check. One max_model_len request needs its global blocks plus 5 sliding groups of
    # cdiv(1023 + 2 x budget, 16) + 1 blocks (async scheduling: 2 batches in flight), within the pool less the null block.
    # Tokens per pool block (global group): 31B stock 32, compact 48; 12B stock 64, compact 96 (E-70 records, 1 MiB).
    def need(mml, per_block, budget):
        return -(-mml // per_block) + 5 * (-(-(1_023 + 2 * budget) // 16) + 1)
    for plan, per, pool in ((m, (32, 48), m["pool_blocks"]), (run_e71.REH, (64, 96), 6 * 2**30 // 2**20)):
        for role in ("p_budget", "d_budget"):
            for tpb in per:
                assert need(plan["max_model_len"], tpb, plan[role]) <= pool - 1, (plan["max_model_len"], role, tpb)
    assert need(145_216, 32, m["d_budget"]) <= m["pool_blocks"] - 1 < need(145_248, 32, m["d_budget"])  # the edge
    assert m["pool_blocks"] - 1 - need(m["max_model_len"], 32, m["d_budget"]) >= 30      # R-119 F4: the margin
    p = run_e71.engine_kwargs("p", m)
    assert p["kv_transfer_config"]["kv_connector"] == "NixlConnector" and p["max_num_batched_tokens"] == 1_024
    assert m["warmup"] == PRED["warmup"] and m["window"] == PRED["horizon"] - PRED["warmup"]
    assert run_e71.WAIT == 3 and run_e71.MODEL_POOL == PRED["pool"]


def test_watchdogs_fit_the_arm_limits():
    build = {"main": 400, "reh": 120}                    # the engine build bound, heuristic
    for stage, rehearsal in (("main", False), ("reh", True)):
        for arm in stg.ARM_ORDER:
            cap = stg.SMOKE_LIMIT[stage] if arm == "C0" else stg.ARM_LIMIT[stage]
            assert run_e71.plan_for(arm, rehearsal)["watchdog"] + build[stage] <= cap, (stage, arm)
        # R-118 F6: at a slow step (85 ms 31B eager, 40 ms 12B; heuristic) the four arms, the four builds and the
        # snapshot (120 s) fit the session deadline less the reserve; a slower engine makes the deadline skip S2
        step = {"main": 0.085, "reh": 0.040}[stage]
        plans = [run_e71.plan_for(a, rehearsal) for a in stg.ARM_ORDER]
        total = sum(step * (p["warmup"] + p["window"]) + build[stage] * 0.6 for p in plans) + 120   # E-69 builds: 45-204 s
        assert total <= stg.TIMEOUT[stage] - 60 - stg.RESERVE_S, (stage, total)
        assert step * (plans[1]["warmup"] + plans[1]["window"]) + 60 < plans[1]["watchdog"], stage
        # A3: the speed gate constants equal the run_e71 plans
        assert stg.ARM_STEPS[stage] == plans[1]["warmup"] + plans[1]["window"], stage
        assert stg.ARM_WATCHDOG[stage] == plans[1]["watchdog"], stage


def test_speed_gate(tmp_path):
    # the real 12B rehearsal C0 window (68.8 ms per work step) passes; a 31B C0 at 100 ms does not fit the watchdog
    reh = HERE / "results" / "e71-reh-20261116T200322Z"
    ok, info = stg.speed_gate(reh, "reh", 1_000)
    assert ok and 68 < info["c0_step_ms"] < 70, info
    assert stg.speed_gate(reh, "main", 7_000)[0]                  # 68.8 ms x 22,000 = 1,514 s
    (tmp_path / "C0").mkdir()
    (tmp_path / "C0" / "d_run.json").write_text(json.dumps({"window": {"seconds": 30.0, "work_steps": 300}}))
    ok, info = stg.speed_gate(tmp_path, "main", 7_000)
    assert not ok and info["projected_arm_s"] == 2_200, info
    assert not stg.speed_gate(tmp_path / "missing", "main", 7_000)[0]


def arm(tps, sat=0.99, steps=20_000, compact=False, status="ok", early=0, pool=6_144, rate=None):
    d = {"status": status, "gpu": "NVIDIA A100-SXM4-80GB", "manifest_ok": True, "plan": {"pool_blocks": 6_144},
         "kv": {"layout": "LBHNC", "pool_blocks": pool}, "spec": {"applied": compact},
         "window": {"decode_tok_s": rate or 20 * tps, "tokens_per_step": tps, "saturated_share": sat,
                    "work_steps": steps, "early_finishes": early, "added": 50, "completed": 9}}
    return {"p": {"status": "ok", "manifest_ok": True, "gpu": "NVIDIA A100-SXM4-80GB", "kv": {"layout": "LBHNC"},
                  "spec": {"applied": compact}, "produced": 52}, "d": d}


def test_word_rule():
    lo, hi = PRED["r_step_interval"]
    mid = (lo + hi) / 2

    def mk(c, s1=5.0, s2=5.0, **kw):
        return {"C0": arm(5.0, compact=True, steps=300), "S1": arm(s1), "C": arm(c, compact=True, **kw), "S2": arm(s2)}
    assert an.word(mk(5 * mid), PRED)["word"] == "STEP-CONFIRMS"
    assert an.word(mk(5 * (lo - 0.02)), PRED)["word"] == "STEP-BELOW"
    assert an.word(mk(5 * (hi + 0.02)), PRED)["word"] == "STEP-ABOVE"
    # the A/A spread widens the band: 2% spread -> 1% on each side
    assert an.word(mk(5.05 * lo * 0.995, s2=5.1), PRED)["word"] == "STEP-CONFIRMS"
    # R-119 F6: the half-width has a floor of 1% when S1 and S2 agree
    assert an.word(mk(5 * lo * 0.992), PRED)["word"] == "STEP-CONFIRMS"
    assert an.word(mk(5 * lo * 0.985), PRED)["word"] == "STEP-BELOW"
    assert an.word(mk(5 * mid, s2=5.5), PRED)["word"] == "INVALID (V4)"
    assert an.word(mk(5 * mid, sat=0.5), PRED)["word"] == "INVALID (V3)"
    assert an.word(mk(5 * mid, steps=10_000), PRED)["word"] == "INVALID (V6)"
    assert an.word(mk(5 * mid, early=2), PRED)["word"] == "INVALID (V7)"
    assert an.word(mk(5 * mid, pool=6_000), PRED)["word"] == "INVALID (V8)"
    bad = mk(5 * mid)
    bad["C0"]["d"]["status"] = "failed"
    assert an.word(bad, PRED)["word"].startswith("INVALID")
    bad = mk(5 * mid)
    bad["C0"]["d"]["window"]["early_finishes"] = 1
    assert an.word(bad, PRED)["word"] == "INVALID (V7)"
    # the throughput ratio is a report only: an eager-like 1.24 keeps the word
    out = an.word(mk(5 * mid, rate=100 * 1.24), PRED)
    assert out["word"] == "STEP-CONFIRMS" and out["secondary"]["gate_1.25"] is False


def mounted_tree(root: Path) -> Path:
    for local, remote in stg.mounts():
        dst = root / remote.removeprefix(stg.CODE + "/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())
    return root


def test_manifest_and_caps():
    for local, _ in stg.mounts():
        assert local.exists(), local
    assert json.loads(stg.MANIFEST_LOCAL.read_text()) == stg.make_manifest()
    assert 14.5 < stg.cost_cap("main") < 15.2 and 2.8 < stg.cost_cap("reh") < 3.2


def test_stub_session(tmp_path, monkeypatch):
    for k in ("STUB_AA", "STUB_RATIO", "STUB_EARLY"):
        monkeypatch.delenv(k, raising=False)
    code = mounted_tree(tmp_path / "code")
    out = stg.session(str(code), str(tmp_path / "res"), "e71-stub", "reh", stub=True)
    rec = out["record"]
    assert rec["ok"] and all(rec["arms"][a]["status"]["d"] == "ok" for a in stg.ARM_ORDER), rec
    assert "REHEARSAL | STEP-CONFIRMS" in rec["decision"], rec["decision"]
    top = stg.unpack(out["tar"], tmp_path / "copy")
    assert (top / "summary.json").exists()
    monkeypatch.setenv("STUB_AA", "1.10")
    rec = stg.session(str(code), str(tmp_path / "res"), "e71-aa", "reh", stub=True)["record"]
    assert "INVALID (V4)" in rec["decision"], rec["decision"]
    # R-119 B1: an early finish in the smoke arm stops the session before the paid arms
    monkeypatch.delenv("STUB_AA")
    monkeypatch.setenv("STUB_EARLY", "1")
    rec = stg.session(str(code), str(tmp_path / "res"), "e71-early", "reh", stub=True)["record"]
    assert "C0 smoke gate failed" in rec.get("stop", "") and set(rec["arms"]) == {"C0"}, rec
