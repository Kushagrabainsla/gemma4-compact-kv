"""E-73 Mac tests: the D3Q sequence equals the model's, the routing of the E-71 driver, the word rules, the limits and
the cost cap, the manifest, and the stub session end to end. Run this file on its own (it routes `run_e71`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_capacity_model"), str(EXP / "gemma4_kv_decode_scaling")]
import analyze_e73 as an  # noqa: E402
import e73_stage as stg  # noqa: E402
import run_e71 as r71  # noqa: E402
import run_e73 as r73  # noqa: E402

PRED = json.loads((HERE / "prediction_interval_e73.json").read_text())
r73.install()


def test_sequence_equals_the_model_sequence():
    import e64
    import e71_predict as ep
    import poolsim as ps
    e64.draw = ep.draw
    g = e64.gen_factory("D3Q", 0)
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
    assert r73.requests(len(seen), r71.MAIN["max_model_len"]) == seen and len(seen) > 5
    prompts = [p for p, _ in r73.requests(400, 144_000)]
    assert 7_000 < sorted(prompts)[200] < 9_500                      # the D3Q prompt median, not D3H


def test_routing_and_limits():
    assert r71.requests is r73.requests and r71.ARMS == {"S": "stock", "C": "compact"}
    kw = r71.engine_kwargs("p", r71.MAIN)
    assert kw["kv_transfer_config"]["kv_connector_extra_config"]["kv_lease_duration"] == 600
    plan = r71.plan_for("S", False)
    assert plan is r71.MAIN and (plan["warmup"], plan["window"]) == (PRED["warmup"], PRED["horizon"] - PRED["warmup"])
    assert plan["max_model_len"] == PRED["max_len"] and r71.MODEL_POOL == PRED["pool"]
    for arm in stg.ARM_ORDER:
        assert plan["watchdog"] + 250 <= stg.ARM_LIMIT[arm]
    assert 0.075 * 22_000 + 60 < plan["watchdog"]                    # the predicted compact step 75 ms
    assert 2 * stg.ARM_LIMIT["S"] + 120 <= stg.TIMEOUT - 60 - stg.RESERVE_S      # both arms at their limits
    assert 9.6 < stg.cost_cap() < 10.0 and stg.cost_cap() < 10.45 - 0.5            # A2: below the credits left (D-127 waiver)
    assert plan["watchdog"] == r73.WATCHDOG_S == 2_050
    for local, _ in stg.mounts():
        assert local.exists(), local
    assert json.loads(stg.MANIFEST_LOCAL.read_text()) == stg.make_manifest()


def arms(r=1.24, sat=0.99, lease=0, lease_set=True, hook_problem=False, s_mode="OFF"):
    def role(compact, window=None):
        d = {"status": "ok", "gpu": "NVIDIA A100-SXM4-80GB", "manifest_ok": True,
             "hooks": {"hooked": [5], "problems": ["x"] if hook_problem else []},
             "mode": "VIEWS" if compact else s_mode,
             "kv": {"layout": "LBHNC", "pool_blocks": 6_144}, "spec": {"applied": True} if compact else None}
        if window:
            d["window"] = window
        return d
    w = lambda tps: {"tokens_per_step": tps, "decode_tok_s": tps * 14, "saturated_share": sat,  # noqa: E731
                     "work_steps": 20_000, "early_finishes": 0, "completed": 80}
    return {"S": {"p": role(False), "d": role(False, w(8.25)), "lease_lines": lease, "lease_set": lease_set},
            "C": {"p": role(True), "d": role(True, w(8.25 * r)), "lease_lines": 0, "lease_set": True}}


def test_word_rules():
    lo, hi = PRED["r_step_interval"]
    assert abs(hi * 1.01 - 1.24087) < 1e-5 and abs(lo * 0.99 - 1.15770) < 1e-5
    out = an.words(arms(1.24), PRED)
    assert (out["Q1"], out["Q2"]) == ("STEP-CONFIRMS", "SHIFT-MATCHES"), out
    assert abs(out["shift_model"] - 0.13527) < 1e-4
    assert an.words(arms(1.25), PRED)["Q1"] == "STEP-ABOVE"
    assert an.words(arms(1.15), PRED)["Q1"] == "STEP-BELOW"
    assert an.words(arms(1.30), PRED)["Q2"] == "SHIFT-SMALLER"         # shift 0.076 of 0.135: ratio 0.56
    out = an.words(arms(1.27), PRED)                                   # shift 0.106: ratio 0.79
    assert (out["Q1"], out["Q2"]) == ("STEP-ABOVE", "SHIFT-MATCHES")
    assert an.words(arms(1.19), PRED)["Q2"] == "SHIFT-LARGER"          # shift 0.186: ratio 1.38
    assert an.words(arms(1.24, lease=1), PRED)["Q1"] == "INVALID (W3)"
    assert an.words(arms(1.24, lease_set=False), PRED)["Q2"] == "INVALID (W3)"
    assert an.words(arms(1.24, sat=0.9), PRED)["Q1"] == "INVALID (W4)"
    assert an.words(arms(1.24, hook_problem=True), PRED)["Q1"] == "INVALID (W7)"
    assert an.words(arms(1.24, s_mode="VIEWS"), PRED)["Q1"] == "INVALID (W7)"     # A1 (R-123 F1): S must be OFF


def test_build_records_the_mode(monkeypatch):
    """A1 (R-123 F1): the wrapper records the mode that the E-71 build sets (a stand-in for compact_engine)."""
    import types

    class Splice:
        def set_mode(self, mode):
            self.mode = mode
    fake = types.SimpleNamespace(ViewsSplice=Splice)
    monkeypatch.setitem(sys.modules, "compact_engine", fake)

    def orig(layout, role, model, plan, rec):
        Splice().set_mode("VIEWS" if layout == "compact" else "OFF")
        return "llm", "ce"
    for layout, want in (("stock", "OFF"), ("compact", "VIEWS")):
        rec: dict = {}
        assert r73.recording_build(orig)(layout, "d", "m", {}, rec) == ("llm", "ce") and rec["mode"] == want
    assert Splice.set_mode.__name__ == "set_mode"                     # restored after the build


def mounted_tree(root: Path) -> Path:
    for local, remote in stg.mounts():
        dst = root / remote.removeprefix(stg.CODE + "/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())
    return root


def test_stub_session(tmp_path, monkeypatch):
    for k in ("STUB_RATIO", "STUB_LEASE", "STUB_AA"):
        monkeypatch.delenv(k, raising=False)
    code = mounted_tree(tmp_path / "code")
    out = stg.session(str(code), str(tmp_path / "res"), "e73-stub", stub=True)
    rec = out["record"]
    assert rec["ok"] and all(rec["arms"][a]["status"] == {"p": "ok", "d": "ok"} for a in stg.ARM_ORDER), rec
    assert "Q1 STEP-CONFIRMS" in rec["decision"] and "Q2 SHIFT-MATCHES" in rec["decision"], rec["decision"]
    assert (stg.s71.unpack(out["tar"], tmp_path / "copy") / "summary.json").exists()
    monkeypatch.setenv("STUB_RATIO", "1.30")
    rec = stg.session(str(code), str(tmp_path / "res"), "e73-high", stub=True)["record"]
    assert "Q1 STEP-ABOVE" in rec["decision"] and "Q2 SHIFT-SMALLER" in rec["decision"], rec["decision"]
    monkeypatch.delenv("STUB_RATIO")
    monkeypatch.setenv("STUB_LEASE", "1")
    rec = stg.session(str(code), str(tmp_path / "res"), "e73-lease", stub=True)["record"]
    assert "Q1 INVALID (W3)" in rec["decision"], rec["decision"]
