"""Tests of the EXPLORATORY part 2 path (explore_e68.py), on the real A100 SXM fit of S6 attempt 3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]
import e68  # noqa: E402
import explore_e68 as ex  # noqa: E402
from test_e68 import lab  # noqa: E402

FIT = HERE / "results/e68-main-sxm-20261111/fit_amended.json"
QUICK = dict(seeds=(0,), pools=(8_724,), horizon=4_000, warmup=500)


def gpu() -> dict:
    return ex.explore_fit(json.loads(FIT.read_text()))


def test_label_keeps_the_frozen_word():
    g = gpu()
    assert g["word"] == "EXPLORATORY" and g["frozen_word"] == "INVALID" and g["frozen_failed"] == ["V3"]
    assert g["fit"] == json.loads(FIT.read_text())["fit"]          # the fit itself is not changed


def test_frozen_gates_still_refuse(tmp_path):
    with pytest.raises(SystemExit):
        e68.load(FIT)
    ex.check(gpu())
    with pytest.raises(SystemExit):
        ex.check(json.loads(FIT.read_text()))


def test_none_and_mode_are_labelled(tmp_path):
    g = gpu()
    import e66
    _, pf = e66.load()
    n = ex.run_none(g, pf, tmp_path / "n.json", names=(("D", ("M2",)),), **QUICK)
    assert n["gpu_word"] == "EXPLORATORY" and n["word"].startswith("EXPLORATORY | ")
    m = ex.run_mode(g, "all200", tmp_path / "m.json", names=("M2",), **QUICK)
    assert m["gpu_word"] == "EXPLORATORY" and "exploratory" in json.loads((tmp_path / "m.json").read_text())
    assert m["cells"][0]["r_meas_M_L20"] > 0


def test_word_is_labelled_and_keeps_the_frozen_word(monkeypatch):
    monkeypatch.setattr(e68, "v8", lambda *a: [])
    out = ex.word(lab([]), lab([], near=("D-M2",)), {"labels": {}}, gpu())
    assert out["word"] == "EXPLORATORY MEAS-BORDERLINE" and out["frozen_word"] == "INVALID"
    monkeypatch.setattr(e68, "v8", lambda *a: ["x"])
    assert ex.word(lab([]), lab([]), {"labels": {}}, gpu())["word"] == "EXPLORATORY INVALID"
