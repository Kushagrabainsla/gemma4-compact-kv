"""CPU tests of E-74 on synthetic data (README Section 12, after A1). No capture file is read.

Run: uv run --no-project --with numpy --with torch --with pytest python -B -m pytest <this file> -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_e74 as e  # noqa: E402

an, ca = e.an, e.ca
W = 0.0615                                # a k_norm weight like the real ones (0.060 to 0.065); not a power of 2


def make(tokens: int = 192, q_rows: int = 24, seed: int = 0, w: float = W):
    """Synthetic K, V and q with the Gemma 4 structure: V rms-normed, K = rope(w V); BF16 values; a few strong
    channels and one large first row (a sink), so that the rows have different scales."""
    gen = torch.Generator().manual_seed(seed)
    wv = torch.full((ca.HEAD,), w, dtype=torch.float64)
    _, v = ca.make_kv(tokens, 4, wv, gen, torch.float64)
    boost = torch.ones(ca.HEAD, dtype=torch.float64)
    boost[[5, 70, 300, 410]] = 6.0
    v = v * boost
    v[0] = v[0] * 20.0
    pos = torch.arange(tokens, dtype=torch.float64)[:, None].expand(tokens, 4)
    k = ca.rope(wv * v, pos)
    q = torch.randn(q_rows, 32, ca.HEAD, generator=gen, dtype=torch.float64) * 4.0
    qpos = torch.arange(tokens - q_rows, tokens)
    return an.bf16(q.float()), an.bf16(k.float()), an.bf16(v.float()), qpos, an.bf16(wv.float())


def test_scale_and_codes_copy_the_vllm_rule():
    _, k, _, _, _ = make()
    s = e.row_scale(k)
    c = e.codes(k, s)
    for t, h in ((0, 0), (3, 1), (100, 3)):
        x = k[t, h]
        scale32 = torch.tensor(max(float(x.abs().max()) / 448.0, 1e-6), dtype=torch.float32)
        want = (x * (1.0 / scale32)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
        assert torch.equal(c[t, h], want)
        assert float(s[t, h, 0]) == float(scale32)
    assert float(e.row_scale(torch.zeros(2, 4, 8))[0, 0, 0]) == float(torch.tensor(1e-6, dtype=torch.float32))
    assert e.over_clamp(k, s) <= 1.0 + e.CLAMP_TOL
    assert e.over_clamp(k, s / 2) > 1.5                      # a scale that is too small is seen (V5)
    assert e.top_share(c) == 1.0                             # every row reaches the code 448 (V5, A1 F6)
    assert e.top_share(e.codes(k, e.row_scale(k, 2.0 ** 0.5))) == 0.0


def test_paired_scale_factors():
    q, k, v, qpos, w = make()
    assert e.FACTORS[0] == 1.0 and len(e.FACTORS) == 8 and all(g >= 1.0 for g in e.FACTORS)
    assert torch.equal(e.row_scale(k, 1.0), e.row_scale(k))
    for g in e.FACTORS:                                      # a factor of 1 or more cannot saturate
        assert e.over_clamp(k, e.row_scale(k, g)) <= 1.0 + e.CLAMP_TOL
    ref = an.attend(q, k, v, qpos)
    errs = [an.rel(e.row1_arm(q, k, v, qpos, w, g=g), ref) for g in e.FACTORS]
    assert len(set(errs)) > 1                                # the factors change the grid phase


def test_row1_without_the_fp8_cast():
    q, k, v, qpos, w = make()
    ref = an.attend(q, k, v, qpos)
    exact = e.row1_arm(q, k, v, qpos, w, cast=e.identity)    # only the BF16 rounding of the folded query stays
    assert an.rel(exact, ref) < 2e-2
    assert an.rel(e.row1_arm(q, k, v, qpos, w), ref) > 2 * an.rel(exact, ref)
    un = torch.tensor(ca.UNROT)
    k2 = k.clone()
    k2[..., un] = (w * v)[..., un]                           # K_unrot = w x V_unrot with no BF16 mismatch
    plain = ca.unpermute(an.attend(ca.views_query(q, w), *ca.views_of(ca.pack_views(k2, v)), qpos))
    assert an.rel(plain, an.attend(q, k2, v, qpos)) < 1e-4   # with no rounding at all, the fold is exact
    bf16c = ca.unpermute(an.attend(an.bf16(ca.views_query(q, w)), *ca.views_of(ca.pack_views(k, v)), qpos))
    assert an.rel(exact, bf16c) < 1e-5                       # ROW1_ID equals the E-54 BF16C arm (V8)


def test_row1_equals_pth_when_w_is_a_power_of_2():
    """(A1, R-126 B4) With w = 2^-4 and no rotated part, the PTH K codes and the ROW1 V_unrot codes are bit-identical,
    and the two arms give the same output."""
    gen = torch.Generator().manual_seed(3)
    w = 2.0 ** -4
    v = torch.randn(96, 4, ca.HEAD, generator=gen)
    v[..., torch.tensor(ca.ROT)] = 0.0
    v = an.bf16(v)
    k = an.bf16(w * v)
    q = an.bf16(torch.randn(16, 32, ca.HEAD, generator=gen))
    qpos = torch.arange(80, 96)
    wv = torch.full((ca.HEAD,), w)
    un = torch.tensor(ca.UNROT)
    kc = e.codes(k, e.row_scale(k))
    c1, s1 = e.row1_codes(k, v)
    assert torch.equal(kc[..., un], c1[..., e.NROT:ca.HEAD])
    assert torch.equal(s1, e.row_scale(v))                   # the ROW1 scale is the PTH V scale (R-126 B1)
    assert an.rel(e.row1_arm(q, k, v, qpos, wv), e.pth_arm(q, k, v, qpos)) < 1e-5


def test_row1_views_share_one_scale_and_cannot_saturate():
    _, k, v, _, _ = make()
    c, s = e.row1_codes(k, v)
    assert c.shape[-1] == 640 and s.shape == (k.shape[0], k.shape[1], 1)
    assert float(c.abs().max()) <= 448.0
    assert e.over_clamp(ca.pack_views(k, v), s) <= 1.0 + e.CLAMP_TOL
    assert float((ca.pack_views(k, v).abs().argmax(-1) < e.NROT).double().mean()) == 0.0


def test_row2_scales_prevent_saturation_and_match_the_standard_cache():
    q, k, v, qpos, w = make()
    w_un = float(w[ca.UNROT[0]])
    ks, vs = e.row2_scales(k, v, w_un)
    row = ca.pack_views(k, v)
    assert e.over_clamp(row[..., :e.NROT], ks.abs()) <= 1.0 + e.CLAMP_TOL
    assert e.over_clamp(row[..., e.NROT:], vs) <= 1.0 + e.CLAMP_TOL
    ref = an.attend(q, k, v, qpos)
    a, b = an.rel(e.row2_arm(q, k, v, qpos, w_un), ref), an.rel(e.row2std_arm(q, k, v, qpos, w_un), ref)
    assert e.TIE_LO <= a / b <= e.TIE_HI                     # the algebra control (V7) on synthetic data


def test_kernel_emulation_is_close_to_float32():
    q, k, v, qpos, w = make()
    ref = an.attend(q, k, v, qpos)
    for f32, kem in ((e.pth_arm(q, k, v, qpos), e.pth_arm(q, k, v, qpos, kem=True)),
                     (e.row1_arm(q, k, v, qpos, w), e.row1_arm(q, k, v, qpos, w, kem=True))):
        assert an.rel(kem, f32) < 1e-2                       # only the BF16 rounding of P x v_scale differs
        assert 0.5 < an.rel(kem, ref) / an.rel(f32, ref) < 2.0
    one = torch.ones_like(e.row_scale(k))
    assert an.rel(e.attend_kem(q, k, one, v, one, qpos), ref) < 5e-3


def test_negative_controls_fail():
    q, k, v, qpos, w = make()
    ref = an.attend(q, k, v, qpos)
    good = an.rel(e.row1_arm(q, k, v, qpos, w), ref)
    assert an.rel(e.row1_arm(q, k, v, qpos, w * 1.25), ref) > 2 * good        # a wrong w in the folded query
    c, s = e.row1_codes(k, v)
    qf = an.bf16(ca.views_query(q, w))
    shifted = ca.unpermute(an.attend(qf, c[..., :ca.HEAD] * s, c[..., 127:127 + ca.HEAD] * s, qpos))
    assert an.rel(shifted, ref) > 3 * good                                    # a V view with the wrong offset
    wrong = torch.roll(s, 1, 0)
    kc, vc = ca.views_of(c)
    assert an.rel(ca.unpermute(an.attend(qf, kc * wrong, vc * wrong, qpos)), ref) > 3 * good   # the wrong row's scale
    assert an.rel(e.row1_arm(q, k, v, qpos, w, cast=an.e5m2), ref) > 1.5 * good   # one mantissa bit less (V6)


def test_parts_and_exact0():
    q, k, v, qpos, w = make()
    ref = an.attend(q, k, v, qpos)
    full = an.rel(e.row1_arm(q, k, v, qpos, w), ref)
    parts = {p: an.rel(e.row1_arm(q, k, v, qpos, w, part=p), ref) for p in ("krot", "kun", "v")}
    assert all(0 < x < full * 1.05 for x in parts.values())  # each part alone is not worse than all of them
    assert an.rel(e.pth_arm(q, k, v, qpos, part="k"), ref) < an.rel(e.pth_arm(q, k, v, qpos), ref) * 1.05
    assert an.rel(e.row1_arm(q, k, v, qpos, w, exact0=True), ref) != full


def test_layer_metrics_on_a_synthetic_capture():
    q, k, v, qpos, w = make()
    cap = {"q": q.to(torch.bfloat16), "k": k.to(torch.bfloat16), "v": v.to(torch.bfloat16), "qpos": qpos}
    sc = {"calstd_joint": {"ks": 0.0078125, "vs": 0.125}, "tied": {"ks": float(w[1]) * 0.125, "vs": 0.125}}
    m = e.layer_metrics(cap, w, sc, "cpu")
    assert set(m["err"]) == set(e.ARMS) and set(m["errk"]) == set(e.ARMS_KEM)
    assert all(len(m["errj"][a]) == 8 and m["errj"][a][0] == m["err"][a] for a in e.JARMS)
    assert all(math.isfinite(x) and x > 0 for x in list(m["err"].values()) + list(m["errk"].values()))
    assert m["w_scalar"] and m["w_module"] and m["w_match"] and m["identity_err"] <= e.ID_TOL and m["w_value"] > 0
    assert m["nonfinite_kem"] == 0 and all(x <= 1.0 + e.CLAMP_TOL for x in m["over_clamp"].values())
    assert all(x >= e.TOP_SHARE for x in m["top_share"].values()) and m["same_v_scale"] == 1.0
    assert 0.0 <= m["sub"]["ROW1_K_rot"] <= 1.0 and m["row_max_in_K_rot"] == 0.0


# ---------------------------------------------------------------- the rule


def fake_rows(row1: float, row2: float, pth: float = 0.040, stock: float = 0.043) -> tuple[dict, list[str]]:
    """7 prompts x 10 layers with the same error per arm in every cell and every scale factor."""
    err = {a: 0.03 for a in e.ARMS} | {"STOCK": stock, "CALSTD_JOINT": 0.0256, "TIED": 0.0289, "PTH": pth,
                                       "ROW1": row1, "ROW2": row2, "ROW2STD": row2, "ROW1_E5M2": row1 * 3}
    cell = {"err": err, "errj": {"PTH": [pth] * 8, "ROW1": [row1] * 8, "ROW2": [row2] * 8},
            "errk": {"PTH": pth, "ROW1": row1, "ROW2": row2, "STOCK": stock},
            "sub": {"ROW1_K_rot": 0.01, "ROW1_V_unrot": 0.0, "ROW1_V_rot": 0.0, "ROW1_K_rot_zero": 0.0, "PTH_K": 0.0},
            "row_max_in_K_rot": 0.0, "same_v_scale": 1.0, "over_clamp": {"PTH": 1.0, "ROW1": 1.0, "ROW2": 1.0},
            "top_share": {"PTH_K": 1.0, "PTH_V": 1.0, "ROW1": 1.0}, "scale_floor_rows": 0.0, "nonfinite_kem": 0}
    pids = [str(i) for i in range(7)]
    return {str(layer): [copy.deepcopy(cell) for _ in pids] for layer in range(10)}, pids


def word(rows, pids, **checks) -> str:
    c = {f"V{i}": True for i in range(1, 9)} | checks
    return e.verdict({"checks": c, "stats": e.stats(rows, pids)}).split(" ")[0]


def test_rule_words():
    assert word(*fake_rows(0.042, 0.041)) == "GO"
    assert word(*fake_rows(0.056, 0.041)) == "PIVOT"         # ROW1 at 1.40 times PTH, ROW2 passes
    assert word(*fake_rows(0.080, 0.080)) == "KILL"          # both at 2.0 times PTH
    assert word(*fake_rows(0.056, 0.056)) == "INCONCLUSIVE"  # both between 1.25 and 1.5
    assert word(*fake_rows(0.080, 0.056)) == "INCONCLUSIVE"  # ROW1 fails, ROW2 neither passes nor fails
    assert word(*fake_rows(0.042, 0.041), V8=False) == "INVALID"
    assert word(*fake_rows(0.042, 0.080)) == "GO"            # ROW1 decides first
    assert word(*fake_rows(0.042, 0.041, stock=0.020)) == "GO"   # (A1, B1) a lucky STOCK does not decide


def test_rule_guards():
    rows, pids = fake_rows(0.042, 0.041)
    for r in rows["3"]:
        r["err"]["ROW1"] = 0.100                             # one layer at 2.5 times PTH: the layer guard
    assert word(rows, pids) == "PIVOT"
    rows, pids = fake_rows(0.042, 0.041)
    for rs in rows.values():
        for r in rs:
            r["errj"]["ROW1"][5] = 0.065                     # one scale factor at 1.6 times PTH: R_A_max
    assert word(rows, pids) == "PIVOT"
    rows, pids = fake_rows(0.042, 0.041)
    for rs in rows.values():
        for r in rs:
            r["errk"]["ROW1"] = 0.060                        # the kernel emulation above 1.25: no GO
    assert word(rows, pids) == "PIVOT"
    rows, pids = fake_rows(0.042, 0.041)
    for rs in rows.values():
        rs[6]["err"]["ROW1"] = 0.070                         # one prompt at 1.75 times PTH: the prompt guard
        rs[6]["err"]["ROW2"] = 0.070
    assert word(rows, pids) == "INCONCLUSIVE"


def test_v8_cell_limit_after_a2():
    assert e.v8_cell_ok(5.0e-3, 5.0e-3) and e.v8_cell_ok(5.04e-3, 5.0e-3)
    assert not e.v8_cell_ok(5.2e-3, 5.0e-3)                  # 4% off at a normal size: fails
    assert e.v8_cell_ok(5.4e-7, 3.1e-7)                      # the layer-29 case of run 1: float32 noise passes
    assert not e.v8_cell_ok(2.0e-6, 1.0e-7)                  # a difference above the floor still fails
    assert not e.v8_cell_ok(math.nan, 1.0e-3)


def test_pass_and_fail_cannot_hold_together():
    for row1 in (0.03, 0.045, 0.052, 0.058, 0.07, 0.2):
        st = e.stats(*fake_rows(row1, row1))
        passed = not e.passes(st["ROW1"])                    # an empty list of failed conditions is a PASS
        assert not (passed and e.fails(st["ROW1"]))
