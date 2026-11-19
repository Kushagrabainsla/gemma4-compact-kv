"""E-72 Mac tests: the lease setting and the E-71 settings, the routing of the E-71 workload code, the three word rules,
the arm limits, the manifest and the cost cap, and the stub session end to end."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_capacity_model")]
import analyze_e72 as an  # noqa: E402
import e72_stage as stg  # noqa: E402
import run_e71 as r71  # noqa: E402
import run_e72 as r72  # noqa: E402

r72.install()


def test_settings_and_routing():
    p = r72.engine_kwargs("p", r72.PLAN["C2"])
    d = r72.engine_kwargs("d", r72.PLAN["C2"])
    assert p["kv_transfer_config"]["kv_connector_extra_config"]["kv_lease_duration"] == 600
    assert p["kv_transfer_config"]["kv_connector_extra_config"]["ucx_backend_params"] == r71.UCX["ucx_backend_params"]
    kids = d["kv_transfer_config"]["kv_connector_extra_config"]["connectors"]
    assert kids[0]["kv_connector_extra_config"]["kv_lease_duration"] == 600
    assert "kv_lease_duration" not in kids[1]["kv_connector_extra_config"]
    for kw in (p, d):                                   # the E-71 settings are unchanged
        assert kw["max_model_len"] == 144_000 and kw["kv_cache_memory_bytes"] == 15 * 2**30
    assert p["max_num_batched_tokens"] == 1_024 and d["max_num_batched_tokens"] == 2_048
    assert r71.build is r72.build and r71.engine_kwargs is r72.engine_kwargs and r71.ARMS["C2"] == "compact"
    c2 = r72.PLAN["C2"]
    assert (c2["warmup"], c2["window"]) == (2_000, 20_000) and r72.PLAN["EV"]["window"] == 0
    assert r72.load_probes(str(EXP / "gemma4_kv_tasks/data_e56.json"))[0]["gold"] == "ember-5809"


def test_limits_and_caps():
    build = 250
    assert r72.PLAN["C2"]["watchdog"] + build <= stg.ARM_LIMIT["C2"]
    assert r72.PLAN["EV"]["watchdog"] + build <= stg.ARM_LIMIT["EV"]
    assert r72.PROBE_WATCHDOG_S + build <= stg.ARM_LIMIT["EV"]                # EV: the build, then the probe
    assert 0.074 * 22_000 + 60 < r72.PLAN["C2"]["watchdog"]            # E-71 C step 74 ms, with margin to 95 ms
    assert 350 + 60 + stg.ARM_LIMIT["C2"] <= stg.TIMEOUT - 60 - stg.RESERVE_S + 100
    assert 6.0 < stg.cost_cap() < 6.4 and 2 * stg.cost_cap() <= 13.9       # D-118: credits about $13.9
    for local, _ in stg.mounts():
        assert local.exists(), local
    assert json.loads(stg.MANIFEST_LOCAL.read_text()) == stg.make_manifest()


def arms(tps=7.50915, lease=0, sat=0.999, diff=False, ev_status="ok", lease_set=True, hook_problem=False):
    probe = [{"tokens": [k, 1, 2], "logprobs": [-0.1, -0.2, -0.3], "finish_reason": "length", "gold_in_text": True}
             for k in range(8)]
    probe_ev = [dict(p, tokens=[p["tokens"][0], 9, 2], logprobs=[-0.1, -0.25, -0.3]) if diff and i == 3 else p
                for i, p in enumerate(probe)]

    def role(compact, extra=None):
        return {"status": "ok", "gpu": "NVIDIA A100-SXM4-80GB", "manifest_ok": True, "mode": "VIEWS",
                "hooks": {"hooked": [5], "problems": ["x"] if hook_problem else []},
                "kv": {"layout": "LBHNC", "pool_blocks": 6_144}, "spec": {"applied": compact}, **(extra or {})}
    win = {"tokens_per_step": tps, "saturated_share": sat, "work_steps": 20_000, "early_finishes": 0, "completed": 63}
    return {"EV": {"p": role(False), "d": role(False, {"probe": probe_ev, "status": ev_status}), "lease_lines": 0,
                   "lease_set": True},
            "C2": {"p": role(True), "d": role(True, {"probe": probe, "window": win, "status_window": "ok"}),
                   "lease_lines": lease, "lease_set": lease_set}}


def test_word_rules():
    out = an.words(arms())
    assert (out["Q1"], out["Q2"], out["Q3"]) == ("REPEATS", "STEP-CONFIRMS-CLEAN", "EXACT"), out
    assert an.words(arms(tps=7.50915 * 1.01))["Q1"] == "DRIFTS"
    assert an.words(arms(tps=7.50915 * 0.97))["Q1"] == "DIFFERS"
    assert an.words(arms(tps=5.4584 * 1.39))["Q2"] == "STEP-ABOVE-CLEAN"
    assert an.words(arms(tps=5.4584 * 1.28))["Q2"] == "STEP-BELOW-CLEAN"
    out = an.words(arms(diff=True))
    assert out["Q3"] == "NOT-EXACT" and out["probe_same"] == 7 and out["probe_first_diff"] == {3: 1}
    assert out["logprobs_equal"] == 7 and out["tokens_compared"] == 24
    out = an.words(arms(lease=2))                       # a lease expired: Q2 and Q3 are not clean, Q1 stays
    assert out["Q1"] == "REPEATS" and out["Q2"] == "INVALID (W3c)" and out["Q3"] == "INVALID (W3)"
    out = an.words(arms(lease_set=False))               # the positive control: no lease line, no clean word
    assert out["Q2"] == "INVALID (W3c)" and out["Q3"] == "INVALID (W3)"
    assert an.words(arms(sat=0.9))["Q1"] == "INVALID (W4)"
    out = an.words(arms(ev_status="failed"))            # R-121 B1: an EV failure costs Q3 only
    assert out["Q3"].startswith("INVALID") and out["Q1"] == "REPEATS" and out["Q2"] == "STEP-CONFIRMS-CLEAN"
    assert an.words(arms(hook_problem=True))["Q3"] == "INVALID (W8)"


def mounted_tree(root: Path) -> Path:
    for local, remote in stg.mounts():
        dst = root / remote.removeprefix(stg.CODE + "/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())
    return root


def test_stub_session(tmp_path, monkeypatch):
    for k in ("STUB_C2", "STUB_DIFF", "STUB_LEASE"):
        monkeypatch.delenv(k, raising=False)
    code = mounted_tree(tmp_path / "code")
    out = stg.session(str(code), str(tmp_path / "res"), "e72-stub", stub=True)
    rec = out["record"]
    assert rec["ok"] and all(rec["arms"][a]["status"] == {"p": "ok", "d": "ok"} for a in stg.ARM_ORDER), rec
    assert "Q1 REPEATS" in rec["decision"] and "STEP-CONFIRMS-CLEAN" in rec["decision"] and "Q3 EXACT" in rec["decision"]
    assert (stg.s71.unpack(out["tar"], tmp_path / "copy") / "summary.json").exists()
    monkeypatch.setenv("STUB_DIFF", "1")
    rec = stg.session(str(code), str(tmp_path / "res"), "e72-diff", stub=True)["record"]
    assert "Q3 NOT-EXACT" in rec["decision"], rec["decision"]
    monkeypatch.delenv("STUB_DIFF")
    monkeypatch.setenv("STUB_LEASE", "1")
    rec = stg.session(str(code), str(tmp_path / "res"), "e72-lease", stub=True)["record"]
    assert "Q2 INVALID (W3c)" in rec["decision"] and "Q3 INVALID (W3)" in rec["decision"], rec["decision"]
    monkeypatch.delenv("STUB_LEASE")
    monkeypatch.setenv("STUB_EVFAIL", "1")              # R-121 B1: C2 still runs after an EV probe failure
    rec = stg.session(str(code), str(tmp_path / "res"), "e72-evfail", stub=True)["record"]
    assert rec["arms"]["C2"]["status"] == {"p": "ok", "d": "ok"} and "Q1 REPEATS" in rec["decision"], rec
