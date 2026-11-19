"""Pipeline tests of E-57 (stub engines, then the frozen analysis). Run with the other tests: uv run --no-project
--with numpy --with torch --with pytest python -B -m pytest -q -p no:cacheprovider experiments/gemma4_kv_engine/"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
DATA = HERE.parent / "gemma4_kv_tasks" / "data_e56.json"


def run(res: Path, plants: dict[str, str] | None = None, rehearsal: bool = False) -> dict:
    """Both stub engines (stock, then compact with the stock capacity file), then the analysis."""
    plants = plants or {}
    codes = {}
    for layout in ("stock", "compact"):
        cmd = [sys.executable, "-B", str(HERE / "run_e57.py"), "--layout", layout, "--data", str(DATA), "--out",
               str(res / layout), "--stub", "--stub-plant", plants.get(layout, "")]
        if layout == "compact":
            cmd += ["--stock-capacity", str(res / "stock" / "capacity.json")]
        if rehearsal:
            cmd.append("--rehearsal")
        codes[layout] = subprocess.run(cmd, capture_output=True, text=True, timeout=600).returncode
    out = subprocess.run([sys.executable, "-B", str(HERE / "analyze_e57.py"), "--res", str(res), "--data", str(DATA)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    s = json.loads((res / "decision.json").read_text())
    s["codes"], s["line"] = codes, out.stdout.strip().splitlines()[-1]
    return s


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return run(tmp_path_factory.mktemp("clean"))


def test_clean_stub_gives_build_go(clean):
    s = clean
    assert s["codes"] == {"stock": 0, "compact": 0}
    assert s["word"] == "BUILD-GO" and s["eq"] == "EXACT" and s["h1"] == "GO1", s["line"]
    assert s["cap_word"] == "GO-CAP" and s["tput_word"] == "TPUT-GAIN" and s["ovh_word"] == "OVH-OK", s["line"]
    assert not any(s["validity"].values()) and s["det"]["ok"]
    for L, ratio in (("32768", 1.338), ("65536", 1.403)):
        assert abs(s["cap"][L]["ratio"] - ratio) < 2e-3
    assert s["line"].startswith("E-57: BUILD-GO | EQ EXACT | GO1")
    assert s["eq_detail"]["schedule_match"] == {"score": True, "gsm8k": True, "needle": True}


def test_clean_stub_capacity_follows_the_gate_model(clean):
    """A1 (R-82 B1, B2): the steady footprints, not the prefill transient; the gate-model batches 3 -> 4 at 32K and
    2 -> 2 at 64K, each shown by a refused probe; the peak ratio is lower; C3 at 64K reuses the C2 row."""
    cap = clean["cap"]
    assert [cap["32768"]["batch"], cap["32768"]["batch_compact"]] == [3, 4]
    assert [cap["65536"]["batch"], cap["65536"]["batch_compact"]] == [2, 2]
    for L in ("32768", "65536"):
        assert cap[L]["limit"] == ["probe refused", "probe refused"]
        assert cap[L]["ratio_peak"] < cap[L]["ratio"] - 0.1
        assert abs(cap[L]["ratio_concurrency"] - cap[L]["ratio"]) < 0.01
        assert cap[L]["hook"] is not None and cap[L]["cache"] is not None and cap[L]["aa"] == 0.0
    assert tuple(cap["32768"]["footprint"]) == (1349.0, 1008.0)


def test_compact_that_differs_fails_the_build_check(tmp_path):
    s = run(tmp_path, {"compact": "compact_differs"})
    assert s["eq"] == "FAIL" and s["word"] == "BUILD-FAIL", s["line"]


def test_a_tiny_float_difference_is_near(tmp_path):
    """A1 (R-82 F1): a difference inside the E-56 margins (no answer change, NLL change 1e-4) is EQ NEAR, not a bug."""
    s = run(tmp_path, {"compact": "compact_near"})
    assert s["eq"] == "NEAR" and s["word"] == "BUILD-GO", s["line"]


def test_missing_compact_spec_is_invalid(tmp_path):
    s = run(tmp_path, {"compact": "no_compact"})
    assert s["word"] == "INVALID" and s["validity"]["V1"], s["line"]


def test_missing_view_fix_stops_the_compact_engine(tmp_path):
    s = run(tmp_path, {"compact": "no_fix"})
    assert s["codes"]["compact"] == 3 and s["word"] == "INVALID" and s["validity"]["V6"], s["line"]


def test_unhealthy_baseline_stops_the_stock_engine(tmp_path):
    s = run(tmp_path, {"stock": "unhealthy"})
    assert s["codes"]["stock"] == 3 and s["word"] == "INVALID" and s["validity"]["V5"], s["line"]


def test_preemption_steps_the_batch_down(tmp_path):
    """A preempting compact engine holds one request: TPUT-NONE, no clean C3 (NA-OVH), so the word is PIVOT (F9)."""
    s = run(tmp_path, {"compact": "preempt"})
    assert all(x["batch_compact"] == 1 for x in s["cap"].values())
    assert s["tput_word"] == "TPUT-NONE" and s["ovh_word"] == "NA-OVH" and s["word"] == "BUILD-PIVOT", s["line"]


def test_a_failed_reported_arm_does_not_stop_the_run(tmp_path):
    s = run(tmp_path, {"compact": "step_error"})
    rc = json.loads((tmp_path / "compact" / "run_main.json").read_text())
    assert s["codes"]["compact"] == 0 and [e["step"] for e in rc["step_errors"]] == ["NAIVE"]
    assert s["word"] == "BUILD-GO" and s["gross"] is not None, s["line"]


def test_slow_compact_path_shows_as_overhead(tmp_path):
    s = run(tmp_path, {"compact": "slow"})
    assert s["ovh_word"] == "OVH-HIGH", s["line"]


def test_rehearsal_runs_the_reduced_workload(tmp_path):
    s = run(tmp_path, rehearsal=True)
    rc = json.loads((tmp_path / "compact" / "run_main.json").read_text())
    assert s["word"] == "REHEARSAL" and rc["items"] == [48, 8] and sorted(s["cap"]) == ["16384", "8192"]


def test_a_probe_too_short_for_the_gap_check_stops_the_run(tmp_path):
    s = run(tmp_path, {"stock": "short_probe"})
    rs = json.loads((tmp_path / "stock" / "run_main.json").read_text())
    assert s["codes"]["stock"] == 3 and any("gaps for 0 of 10" in p for p in rs["preflight"]["problems"])
    assert s["word"] == "INVALID" and s["validity"]["V4"], s["line"]


def test_the_stub_ran_the_real_hooks(clean):
    """L-26: the pre-flight evidence comes from the real ViewsSplice on stub modules in the engine's call order."""
    assert clean["validity"]["V4"] == []


def steps_of(held, decoding, seconds=None, running=None):
    n = len(held)
    return [{"free": 6_143 - held[i], "decoding": decoding[i], "running": (running or [1] * n)[i],
             "seconds": (seconds or [0.05] * n)[i], "held": held[i]} for i in range(n)]


def test_footprint_skips_the_prefill_transient():
    """R-82 B1: the first decode steps still hold the last chunk's sliding blocks; C1 takes the steady median."""
    import run_e57 as r
    held = [400 * i for i in range(1, 7)] + [2_607, 1_900] + [1_349] * 60
    dec = [0] * 6 + [1] * 62
    fp = r.footprint_from_steps(steps_of(held, dec), 6_143)
    assert fp["blocks"] == 1_349 and fp["blocks_peak"] == 2_607 and fp["blocks_first_decode"] == 2_607


def test_window_skips_three_full_steps_and_drops_outliers():
    """R-82 B3: the window starts at the fourth full step; a step above 5 times the median is dropped and counted."""
    import run_e57 as r
    n = 200
    secs = [1.0] * 10 + [1.2, 0.9, 0.7] + [0.05] * (n - 13)     # 10 prefill steps, then 3 heavy full steps
    secs[60] = 2.0
    st = steps_of([3_000] * n, [1] * 10 + [2] * (n - 10), secs, [2] * n)
    w = r.window_from_steps(st, 2)
    assert w["full_steps"] == n - 10 and w["dropped"] == 1 and w["window_steps"] == r.CAP_WINDOW - 1
    assert w["step_ms"] == 50.0 and w["tokens_per_s"] == 40.0 and w["median_ms"] == 50.0


def test_batch_search_finds_and_shows_the_limit():
    """A1 (R-82 B2): down to the first clean row; up while the probe is admitted; each batch once."""
    import run_e57 as r

    def runner(true_max, probe_ok_below):
        def row(b):
            ok = b <= true_max
            return {"batch": b, "preemptions": 0, "window_steps": 128 if ok else 0, "step_ms": 50.0 if ok else None,
                    "probe_admitted": (b < probe_ok_below) if ok else None}
        return row

    always = lambda b: True  # noqa: E731
    rows, best, limit = r.search_batch(runner(3, 3), 5, 16, always)
    assert [x["batch"] for x in rows] == [5, 4, 3] and rows[best]["batch"] == 3 and limit == "next batch failed"
    rows, best, limit = r.search_batch(runner(3, 3), 2, 16, always)
    assert [x["batch"] for x in rows] == [2, 3] and rows[best]["batch"] == 3 and limit == "probe refused"
    rows, best, limit = r.search_batch(runner(9, 9), 3, 3, always)
    assert rows[best]["batch"] == 3 and limit == "max_num_seqs"
    rows, best, limit = r.search_batch(runner(20, 0), 16, 16, always)      # A2: the S5 case, refused at the cap
    assert rows[best]["batch"] == 16 and limit == "max_num_seqs"
    rows, best, limit = r.search_batch(runner(0, 0), 2, 16, always)
    assert best is None and limit == "none" and [x["batch"] for x in rows] == [2, 1]
    assert r.start_batch(6_143, 1_349, 2_607, 16) == 3 and r.start_batch(6_143, 1_008, 2_267, 16) == 4
    assert r.start_batch(6_143, 2_373, 3_631, 16) == 2 and r.start_batch(6_143, 1_690, 2_949, 16) == 2
    assert r.start_batch(6_143, None, 2_607, 16) == 1


def test_stub_engine_refuses_admission_like_the_gate():
    """The stub scheduler: a 64K stock request waits while one decodes (deferred admission), then runs; a third is
    refused while two decode."""
    import run_e57 as r
    w = r.StubWorld("stock", "", False)
    eng = r.StubEngine(w, 6_144)
    row = r.capacity_run(eng, 65_536, 3, True, lambda: None)
    assert row["end"] == "admission refused" and row["window_steps"] == 0 and row["alloc_fail"] >= r.REFUSE_STEPS
    assert not eng.busy() and row["free_after"] == row["free_before"]
    row = r.capacity_run(eng, 65_536, 2, True, lambda: None)
    assert row["end"] == "window" and r.clean_row(row) and row["probe_admitted"] is False and row["alloc_fail"] >= 1
