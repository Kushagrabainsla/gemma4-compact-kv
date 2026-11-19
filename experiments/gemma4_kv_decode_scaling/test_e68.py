"""E-68 S3 tests (README_E68.md Section 12, as amended by A1): the launcher parts that do not need Modal, the stub
session on the mounted files only, part 2 on the stub table, and the word rule."""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import e68  # noqa: E402
import e68_stage as stg  # noqa: E402


def mounted_tree(root: Path) -> Path:
    """A1 (R-107 F3): only the files that the image mounts, at their container paths under root."""
    for local, remote in stg.mounts():
        dst = root / remote.removeprefix(stg.CODE + "/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(local.read_bytes())
    return root


def test_every_file_that_runs_exists_and_the_manifest_matches_the_e65_code():
    for local, remote in stg.mounts():
        assert local.is_file(), local
        assert remote.startswith(stg.CODE + "/")
    man = json.loads(stg.MANIFEST_LOCAL.read_text())
    e65 = json.loads((HERE / "code_manifest.json").read_text())
    assert set(man) == set(e65)
    # Only poolsim.py (E-66 and E-67 added fields; test_e66.py V1) and bench_e65.py (A3: the ratio tie behind an
    # environment flag) differ from the frozen E-65 manifest.
    assert sorted(k for k in man if man[k] != e65[k]) == ["bench_e65.py", "poolsim.py"]


def test_cost_caps():
    assert stg.cost_cap("reh") < 0.50 and stg.worst_case("reh") < 1.00
    assert stg.cost_cap("main") < 1.20 and stg.worst_case("main") < 2.40
    assert stg.TIMEOUT["main"] > stg.POD_LIMIT["main"] + 120 + 120     # the session ends before the Modal limit
    assert stg.TIMEOUT["reh"] > stg.POD_LIMIT["reh"] + 120 + 120


def test_stub_session_on_the_mounted_files_only(tmp_path):
    code = mounted_tree(tmp_path / "code")
    calls = []
    out = stg.session(str(code), str(tmp_path / "res"), "e68-stub", "reh", stub=True,
                      launcher_sha={"x": "y"}, on_start=lambda: calls.append(1))
    rec = out["record"]
    assert calls == [1]
    assert rec["manifest_ok"] and rec["session_exit"] == 0 and rec["ok"], rec
    assert rec["decision"].startswith("E-65 GPU: REHEARSAL"), rec["decision"]
    assert "failed none" in rec["decision"] and " bench: exit 0 " in rec["bench_line"]
    assert rec["launcher_sha"] == {"x": "y"}
    top = stg.unpack(out["tar"], tmp_path / "copy")
    assert (top / "e65" / "bench" / "fit.json").is_file()
    assert stg.sha_tree(top)["e65/bench/fit.json"] == rec["sha256"]["e65/bench/fit.json"]
    with pytest.raises(ValueError):                                      # results are never overwritten
        stg.unpack(out["tar"], tmp_path / "copy")
    # A1 (R-107 B2): a second attempt in the same run folder (a preemption restart) does nothing.
    again = stg.session(str(code), str(tmp_path / "res"), "e68-stub", "reh", stub=True)
    assert again["record"] == {"run_id": "e68-stub", "stage": "reh", "ok": False, "restart": True}


def test_wrong_code_stops_before_the_session(tmp_path):
    code = mounted_tree(tmp_path / "code")
    (code / "gemma4_kv_engine" / "compact_layout.py").write_text("# changed\n")
    out = stg.session(str(code), str(tmp_path / "res"), "e68-bad", "reh", stub=True)
    assert out["record"]["manifest_ok"] is False and out["record"]["ok"] is False
    assert "session_exit" not in out["record"]


def test_unpack_refuses_a_path_outside(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"x"
        info = tarfile.TarInfo("run/../../evil.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        stg.unpack(buf.getvalue(), tmp_path / "p")


def test_part2_on_the_stub_table(tmp_path):
    code = mounted_tree(tmp_path / "code")
    out = stg.session(str(code), str(tmp_path / "res"), "e68-stub2", "reh", stub=True)
    top = stg.unpack(out["tar"], tmp_path / "copy")
    gpu = json.loads((top / "e65" / "bench" / "fit.json").read_text())
    fits, pf, gw = e68.load(top / "e65" / "bench" / "fit.json")
    assert gw == "REHEARSAL" and set(fits) == {"r1", "r0625", "meas"}
    assert e68.read_ratio(gpu)["n"] >= 0
    res = e68.run_mode("all200", fits, pf, names=("M2",), seeds=(0,), pools=(8_724,), horizon=4_000, warmup=500)
    cell = res["cells"][0]
    for t in e68.TAGS:
        assert cell[f"r_{t}_M_L20"] > 0
    assert set(res["labels"]["meas"]["L20"]) == {"D-M2"}


def test_bracket_tables_reproduce_e67_cell():
    """The meas tag must not change the bracket columns: one E-67 cell recomputed with three tags."""
    import analyze_e65 as an
    import e66
    import e67
    fits, pf = e66.load()
    models2 = {t: an.Model(f) for t, f in fits.items()}
    a = e67.run_cell("all200", "M2", 8_724, 0, models2, fits, pf, horizon=4_000, warmup=500)
    fits3 = {**fits, "meas": fits["r0625"]}
    models3 = {t: an.Model(fits3[t]) for t in e68.TAGS}
    b = e67.run_cell("all200", "M2", 8_724, 0, models3, fits3, pf, horizon=4_000, warmup=500)
    for k in ("r_r1_M_L20", "r_r0625_M_L20", "r_r0625_M_pf_equal_L20"):
        assert a[k] == b[k]
    assert b["r_meas_M_L20"] == b["r_r0625_M_L20"]                     # a planted 0.625 table gives the 0.625 column


def lab(go: list[str], near: tuple[str, ...] = ()) -> dict:
    names = ("D-D1", "D-D2", "D-D3", "D-D4", "D-M1", "D-M2")

    def entry(k):
        n = 5 if k in go else (4 if k in near else 2)
        return {"label": "GO" if k in go else "MID", "summary": {"Q_M": {"median": 1.3 if k in go else 1.2}},
                "link_use_max": 0.05, "per_seed": [{"label": "GO" if i < n else "MID"} for i in range(5)]}
    return {"go_meas_L20": go, "labels": {t: {"L20": {k: entry(k) for k in names}} for t in e68.TAGS}}


def test_word_rule():
    none = {"labels": {}}
    valid = {"word": "VALID", "failed": [], "Phi": 0.2, "table": []}
    assert e68.word(lab([]), lab([]), none, valid)["word"] == "MEAS-NO"
    assert e68.word(lab([]), lab([], near=("D-M2",)), none, valid)["word"] == "MEAS-BORDERLINE"
    assert e68.word(lab([]), lab(["D-M2"]), none, valid)["word"] == "MEAS-CONFIG-GO"
    assert e68.word(lab(["D-D3"]), lab(["D-M2"]), none, valid)["word"] == "MEAS-DEFAULT-GO"
    assert e68.word(lab([]), lab([]), none, {"word": "INVALID", "failed": ["V4"]})["word"] == "INVALID"
    assert e68.word(lab([]), lab([]), none, {"word": "REHEARSAL", "failed": []})["word"] == "REHEARSAL | MEAS-NO"
    out = e68.word(lab([]), lab([], near=("D-M2",)), none, valid)
    assert out["table"]["D-M2"]["all200_meas"][2] == 4 and out["borderline_all200"] == ["D-M2"]


def test_read_ratio_report():
    gpu = {"table": [{"tokens": 2 ** 18, "read_ratio": 0.9}, {"tokens": 2 ** 19, "read_ratio": 0.8},
                     {"tokens": 2 ** 10, "read_ratio": 0.1}]}
    assert e68.read_ratio(gpu) == {"median": pytest.approx(0.85), "min": 0.8, "max": 0.9, "n": 2}


def test_v8_compares_the_bracket_columns_with_e67():
    ref = {m: json.loads((HERE / "results/e67" / f"e67_{m}.json").read_text()) for m in ("prompt200", "all200")}
    same = {m: {"mode": m, "labels": r["labels"]} for m, r in ref.items()}
    assert e68.v8(same["prompt200"], same["all200"], ref["prompt200"], ref["all200"]) == []
    changed = json.loads(json.dumps(same["all200"]))
    changed["labels"]["r0625"]["L20"]["D-M2"]["label"] = "MID"
    assert e68.v8(same["prompt200"], changed, ref["prompt200"], ref["all200"]) == ["all200 r0625 L20 D-M2"]
    out = e68.word(lab([]), lab(["D-M2"]), {"labels": {}}, {"word": "VALID", "failed": []}, ["all200 r0625 L20 D-M2"])
    assert out["word"] == "INVALID" and out["failed"] == ["V8"]


REH = HERE / "results" / "e68-reh-20261111T215042Z" / "e65" / "bench"


@pytest.mark.skipif(not REH.exists(), reason="the rehearsal results are not on this machine")
def test_a2_on_the_rehearsal_data():
    fit, bench = (json.loads((REH / f).read_text()) for f in ("fit.json", "bench.json"))
    assert fit["failed"] == ["V3"]
    out = e68.a2(fit, bench)
    assert out["a2"]["kept"] == [] and len(out["a2"]["waived"]) == 3
    assert all(w[3] < e68.A2_TOL for w in out["a2"]["waived"])
    assert out["failed"] == [] and out["word"] == "REHEARSAL"           # a rehearsal never becomes VALID
    assert fit["failed"] == ["V3"]                                     # the frozen analysis is not changed
    assert len(out["a2_drift_report"]) == sum(bool(p.get("sequential")) for p in bench["points"])


def pt(b, ln, sl, h, seq=False):
    return {"batch": b, "length": ln, "SL": sl, "H": h, "G": {"S32": [1.0, 1.0, 1.0], "C48": [1.0, 1.0, 1.0]},
            "sequential": seq}


def test_a2_bounds_keep_and_waive():
    bench = {"points": [pt(3, 32_768, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]), pt(4, 32_768, [1.0, 3.0, 2.0], [0.2, 0.2, 0.2]),
                        pt(8, 16_384, [1.0, 1.0, 1.0], [0.2, 0.5, 0.3])]}
    table = [{"B": 3, "L": 32_768, "t_stock_ms": 50.0, "t_compact_ms": 45.0},
             {"B": 4, "L": 32_768, "t_stock_ms": 50.0, "t_compact_ms": 40.0},
             {"B": 8, "L": 16_384, "t_stock_ms": 50.0, "t_compact_ms": 49.0}]
    fit = lambda pts: {"word": "INVALID", "failed": ["V3"], "table": table,  # noqa: E731
                       "checks": {"V3": False, "V3_failed_points": pts}}
    # SL: saving 10 / 50 x 10 x 2.0 ms / 50 = 0.08: kept.
    out = e68.a2(fit([[4, 32_768, "SL"]]), bench)
    assert out["word"] == "INVALID" and out["a2"]["kept"] == [[4, 32_768, "SL", 0.08]]
    # H: 2 x 0.3 ms = 0.6 ms = 1.2% of the step: kept.
    assert e68.a2(fit([[8, 16_384, "H"]]), bench)["word"] == "INVALID"
    # A G point is never waived.
    assert e68.a2(fit([[8, 16_384]]), bench)["word"] == "INVALID"
    # A small SL spread at a point with a small saving: waived, and the word becomes VALID.
    bench["points"][2]["SL"] = [1.0, 1.2, 1.1]
    out = e68.a2(fit([[8, 16_384, "SL"]]), bench)
    assert out["word"] == "VALID" and out["failed"] == [] and out["checks"]["V3"] is True


MAIN1 = HERE / "results" / "e68-main-20261111T215927Z" / "e65" / "bench"
TRY2 = {"G": {"S32": [2.789642333984375, 2.6866891479492185, 2.6882662963867188],
              "C48": [2.3770930480957033, 2.366177215576172, 2.3870054626464845]}}


def test_a3_tie_rule_behind_the_flag(monkeypatch):
    import bench_e65
    monkeypatch.delenv("E68_TIE_RULE", raising=False)
    assert bench_e65.tie_ok(TRY2) is False                             # the E-65 default: absolute, fails on PCIe
    monkeypatch.setenv("E68_TIE_RULE", "ratio")
    assert bench_e65.tie_ok(TRY2) is True                              # 0.884 against 0.903: inside 5%
    far = {"G": {"S32": [2.7, 2.7, 2.7], "C48": [2.7, 2.7, 2.7]}}       # ratio 1.0: 10.8% off
    assert bench_e65.tie_ok(far) is False
    off4 = {"G": {"S32": [2.538] * 3, "C48": [2.538 * 0.9027 * 1.04] * 3}}   # ratio 4% off, reads at E-58: refused
    assert bench_e65.tie_ok(off4) is False
    slow = {"G": {"S32": [2.538 * 1.12] * 3, "C48": [2.291 * 1.12] * 3}}     # same ratio, both reads 12% slow: refused
    assert bench_e65.tie_ok(slow) is False


def test_session_passes_the_ratio_tie_to_the_bench(tmp_path):
    out = stg.session(str(mounted_tree(tmp_path / "code")), str(tmp_path / "res"), "e68-env", "reh", stub=True)
    assert out["record"]["ok"]
    top = stg.unpack(out["tar"], tmp_path / "copy")
    assert json.loads((top / "e65" / "bench" / "bench.json").read_text())["meta"]["tie_rule"] == "ratio"


@pytest.mark.skipif(not MAIN1.exists(), reason="the attempt-1 results are not on this machine")
def test_a3_on_the_attempt_1_data():
    fit, bench = (json.loads((MAIN1 / f).read_text()) for f in ("fit.json", "bench.json"))
    assert sorted(fit["failed"]) == ["V3", "V4", "V5"]
    out = e68.a3(e68.a2(fit, bench), bench)               # attempt 1 ran before the bench recorded `tie_rule`
    assert out["a3"]["v4_ratio_ok"] and out["a3"]["tie_ratio"] == pytest.approx(0.8836, abs=1e-3)
    assert out["a3"]["g_waived"] == [[3, 32_768]] and out["a3"]["kept"] == []
    assert out["failed"] == ["V5"] and out["word"] == "INVALID"         # attempt 1 still has no core points


def test_a3_keeps_a_g_spread_that_is_not_a_first_round_effect():
    bench = {"meta": {"tie_rule": "ratio"}, "points": [{"batch": 3, "length": 32_768, "SL": [0.5] * 3, "H": [0.2] * 3,
                         "G": {"S32": [2.70, 2.80, 2.69], "C48": [2.38, 2.38, 2.38]}}]}
    fit = {"word": "INVALID", "failed": ["V3"], "table": [],
           "checks": {"V3": False, "V3_failed_points": [[3, 32_768]], "V4_values_ms": []}}
    out = e68.amend(fit, bench)
    assert out["word"] == "INVALID" and out["a3"]["kept"] == [[3, 32_768]]
    bench["points"][0]["G"]["S32"] = [2.80, 2.70, 2.69]
    assert e68.amend(fit, bench)["word"] == "VALID"


def test_a3_waiver_scope_and_cap():
    def p(b, ln, seq):
        return {"batch": b, "length": ln, "SL": [0.5] * 3, "H": [0.2] * 3, "sequential": seq,
                "G": {"S32": [2.80, 2.70, 2.69], "C48": [2.38, 2.38, 2.38]}}
    seq_pt = [16, 65_536]
    bench = {"meta": {"tie_rule": "ratio"}, "points": [p(16, 65_536, True)]}
    fit = {"word": "INVALID", "failed": ["V3"], "table": [],
           "checks": {"V3": False, "V3_failed_points": [seq_pt], "V4_values_ms": []}}
    assert e68.amend(fit, bench)["word"] == "INVALID"                  # a sequential point is never waived
    pts = [[1, 1_024], [2, 1_024], [4, 1_024], [8, 1_024]]
    bench = {"meta": {"tie_rule": "ratio"}, "points": [p(b, ln, False) for b, ln in pts]}
    fit = {**fit, "checks": {"V3": False, "V3_failed_points": pts, "V4_values_ms": []}}
    out = e68.amend(fit, bench)                                        # 4 waivable points: more than 3
    assert out["word"] == "INVALID" and out["a3"]["g_waived"] == [] and len(out["a3"]["kept"]) == 4


def test_a3_only_for_the_ratio_tie():
    bench = {"meta": {"tie_rule": "absolute"},
             "points": [{"batch": 3, "length": 32_768, "SL": [0.5] * 3, "H": [0.2] * 3,
                         "G": {"S32": [2.80, 2.70, 2.69], "C48": [2.38, 2.38, 2.38]}}]}
    fit = {"word": "INVALID", "failed": ["V3"], "table": [],
           "checks": {"V3": False, "V3_failed_points": [[3, 32_768]], "V4_values_ms": []}}
    out = e68.amend(fit, bench)
    assert "a3" not in out and out["word"] == "INVALID"                # the frozen rules stand on the SXM run
    bench["meta"]["tie_rule"] = "ratio"
    assert e68.amend(fit, bench)["word"] == "VALID"


def test_word_marks_a3_exploratory():
    out = e68.word(lab([]), lab([]), {"labels": {}}, {"word": "VALID", "failed": [], "a3": {"exploratory": "x"}})
    assert out["word"] == "MEAS-NO (EXPLORATORY, A3: cross-host A100 PCIe)"


def test_invalid_gpu_part_stops_part2(tmp_path):
    f = tmp_path / "fit.json"
    f.write_text(json.dumps({"word": "INVALID", "failed": ["V4"], "fit": {}}))
    with pytest.raises(SystemExit):
        e68.load(f)
