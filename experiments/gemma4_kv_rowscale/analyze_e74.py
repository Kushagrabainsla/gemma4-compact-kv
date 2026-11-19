"""E-74 analysis (README Sections 4 to 7 as amended by A1 after R-126; I-182): one FP8 scale per row for the compact
K=V cache, on the stored Gemma 4 31B activations of E-54 (primary) and E-53 (reported). The Mac CPU, float32, no GPU.

Arms, per prompt and global layer (the reference is the float32 causal GQA attention on the BF16 values, as E-54):
  STOCK, CALSTD_JOINT, TIED   the E-54 arms with the E-54 code and the frozen E-54 scales (the replication control V2)
  PTH        the standard FP8 cache with the vLLM per-token-head rule: per (token, KV head) k_scale = max(max|K| / 448,
             1e-6), codes = fp8(clamp(K * (1 / k_scale))), and the same for V with its own scale
  ROW1       the compact row r = [K_rot | V_unrot | V_rot] with ONE scale per row, s = max(max|r| / 448, 1e-6); the K
             view (codes 0 to 511) and the V view (codes 128 to 639) both read with s; the folded BF16 query
             [q_rot | (q w)_unrot]
  ROW2       the tied row with a scale per row: codes of [K_rot / ks | V_unrot / vs | V_rot / vs], vs = max(max(max|V|,
             max|K_rot| / |w|) / 448, 1e-6), ks = float32(w x vs); the plain permuted query
  ROW2STD    the standard cache with the ROW2 scales of each row (the algebra control, V7)
  ROW1_E5M2  ROW1 with E5M2 codes (the sensitivity control, V6)
  ROW1_ID    ROW1 with no cast (the algebra control V8, against the E-54 BF16C record)
  reported:  ROW1_Q32 (float32 folded query), PTH_X0 and ROW1_X0 (the BOS row exact), ROW1_KROT, ROW1_KUN, ROW1_V,
             PTH_K, PTH_V (only one part from the codes)
PTH, ROW1 and ROW2 run with 8 paired scale factors 2^(j / 8), j = 0 to 7 (A1, R-126 B2); j = 0 is the vLLM rule. At
j = 0 they also run in the kernel emulation (KEM) of the pinned Triton kernel in the per-token-head mode:
S = dot(q, K codes) x k_scale in float32, then (P x v_scale) rounded to BF16, then the product with the V codes.

Two steps: (1) metrics_<set>.json from the capture files, one cell at a time (a stopped run continues); (2) the
validity checks V1 to V8 and the frozen rule from the metrics files.

Usage: analyze_e74.py --res <dir> [--metrics e54 e53] [--limit-cells N]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemma4_kv_scaletie"))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_accuracy"))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import analyze_e53 as an  # noqa: E402  (fp8, e5m2, bf16, rel, attend, w_choice, w_from_data, identity_err)
import analyze_e54 as a54  # noqa: E402  (std_arm, tied_arm, med, rms, ratio, SCALES, SCALES_SHA256)
import compact_attn as ca  # noqa: E402  (ROT, UNROT, pack_views, views_of, views_query, unpermute)

D = ca.HEAD
NROT = len(ca.ROT)
QMAX = an.FP8_MAX                       # 448: the E4M3 range that the vLLM per-token-head kernel uses
SCALE_FLOOR = 1e-6                      # the floor of a per-token-head scale in the vLLM write kernel
SUBNORMAL = 2.0 ** -6                   # E4M3: |code| below this is subnormal (step 2^-9)
FACTORS = tuple(2.0 ** (j / 8) for j in range(8))   # (A1, B2) the paired scale factors; j = 0 is the vLLM rule
SETS = {  # name: (capture folder, run record, the E-54 metrics record or None)
    "e54": (HERE.parent / "gemma4_kv_scaletie/results/e54-h100-20261102/capture",
            HERE.parent / "gemma4_kv_scaletie/results/e54-h100-20261102/e54/capture_run.json",
            HERE.parent / "gemma4_kv_scaletie/results/e54-h100-20261102/e54/metrics.json"),
    "e53": (HERE.parent / "gemma4_kv_accuracy/results/e53-h100-20261101/capture",
            HERE.parent / "gemma4_kv_accuracy/results/e53-h100-20261101/e53/capture_run.json", None),
}
FILES = {"e54": 70, "e53": 40}          # V1: the number of capture files of each set
REPL = {"STOCK": "STOCK", "CALSTD_JOINT": "CALSTD_JOINT", "TIED": "TIED", "ROW1_ID": "BF16C"}   # this run: E-54 record
REPL_POOL_TOL = 0.002                   # V2 and V8 (A1, F5): pooled P within 0.2% of the E-54 record
REPL_CELL_TOL = 0.01                    # V2 and V8: every cell error within 1% of its E-54 value
V8_ABS_FLOOR = 1e-6                     # (A2, after run 1, the user: "Yes, apply A2") V8 cell: 1% + 1e-6 absolute
ID_TOL = an.ID_TOL                      # V3: K_unrot against w x V_unrot
CLAMP_TOL = 1e-5                        # V5: no value above 448 x (1 + 1e-5) before the clamp
TOP_SHARE = 0.999                       # V5 (A1, F6): the largest |code| of a row is 448 in this share of the rows
E5M2_MIN = 1.5                          # V6
TIE_LO, TIE_HI = 0.90, 1.10             # V7
POOL_PASS, POOL_FAIL = 1.25, 1.5        # the rule (README Section 7; the E-54 thresholds)
LAYER_PASS, PROMPT_PASS = 2.0, 1.5
ROWS = ("ROW1", "ROW2")
JARMS = ("PTH", "ROW1", "ROW2")         # the arms with the 8 scale factors and the kernel emulation
ARMS = ("STOCK", "CALSTD_JOINT", "TIED", "PTH", "ROW1", "ROW2", "ROW2STD", "ROW1_E5M2", "ROW1_ID", "ROW1_Q32",
        "PTH_X0", "ROW1_X0", "ROW1_KROT", "ROW1_KUN", "ROW1_V", "PTH_K", "PTH_V")
ARMS_KEM = ("PTH", "ROW1", "ROW2", "STOCK")
NULL_BAND = {"median": 1.02, "p95": 1.19, "p99": 1.26}   # R-74: an equivalent storage at a random phase, against STOCK
BYTES = {"PTH": 2 * (D + 4), "ROW": NROT + D + 4}        # bytes per (token, KV head) with FP8 codes and float32 scales
CLASSES = {"plain8k": (0, 1, 2, 3), "chat": (4, 5), "long": (6,)}   # the E-54 prompt classes (reported)
ACTION = {
    "GO": "one scale per row gives the compact row the attention accuracy of an uncalibrated standard FP8 cache (the "
          "vLLM per-token-head mode); send the numbers to the reader; the next question is the engine design, then a "
          "model-level test (each needs a new plan)",
    "PIVOT": "the rotary part needs its own derived scale (ROW2); report it with the decomposition",
    "KILL": "a scale per row does not help the compact row; check the decomposition for a bug first; the FP8 route "
            "stays with the calibrated tied scales of E-54; send the numbers to the reader",
    "other": "no claim; report each failed condition; the user decides",
    "invalid": "no result; fix the cause; a rerun is free but needs a note of the change",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


# ---------------------------------------------------------------- the per-row storages


def row_scale(x: torch.Tensor, g: float = 1.0) -> torch.Tensor:
    """The vLLM per-token-head scale: max(max|x| / 448, 1e-6) over the last axis, float32, shape [..., 1]. g: the
    paired scale factor of A1 (1.0 is the vLLM rule)."""
    s = (x.abs().amax(-1, keepdim=True) / QMAX).clamp_min(SCALE_FLOOR).to(torch.float32)
    return s if g == 1.0 else (s * g).to(torch.float32)


def codes(x: torch.Tensor, s: torch.Tensor, cast=an.fp8) -> torch.Tensor:
    """The vLLM per-token-head write: x times (1 / s) in float32, clamp at +-448, round to nearest even."""
    return cast(x * (1.0 / s))


def over_clamp(x: torch.Tensor, s: torch.Tensor) -> float:
    """V5: the largest |x x (1 / s)| over 448 (1.0 means exactly on the limit)."""
    return float((x * (1.0 / s)).abs().max() / QMAX)


def top_share(c: torch.Tensor) -> float:
    """V5 (A1, F6): the share of rows whose largest |code| is 448."""
    return float((c.abs().amax(-1) == QMAX).double().mean())


def attend_kem(q, kc, ks, vc, vs, qpos) -> torch.Tensor:
    """The pinned Triton kernel in the per-token-head mode (README Section 2): S = dot(q, K codes) x k_scale[token]
    in float32; P = softmax(S); acc = dot(bf16(P x v_scale[token]), V codes). ks and vs: [L, Hk, 1]. A statistical
    match of the online softmax of the kernel, not a bit-exact one."""
    nq, hq, length, hk = q.shape[0], q.shape[1], kc.shape[0], kc.shape[1]
    g = hq // hk
    future = torch.arange(length, device=q.device)[None, :] > qpos[:, None]
    out = torch.empty(nq, hq, vc.shape[-1], dtype=torch.float32, device=q.device)
    for h in range(hk):
        s = torch.einsum("qgd,ld->qgl", q[:, h * g:(h + 1) * g], kc[:, h]) * ks[:, h, 0][None, None, :]
        p = torch.softmax(s.masked_fill(future[:, None, :], float("-inf")), dim=-1)
        pv = an.bf16(p * vs[:, h, 0][None, None, :])
        out[:, h * g:(h + 1) * g] = torch.einsum("qgl,ld->qgd", pv, vc[:, h])
    return out


def pth_arm(q, k, v, qpos, g: float = 1.0, kem: bool = False, exact0: bool = False, part: str = "all"):
    """The standard cache with the vLLM per-token-head scales (a K scale and a V scale per token and KV head).
    exact0: position 0 without the cast. part: "k" or "v" takes only that tensor from the codes (reported)."""
    ks, vs = row_scale(k, g), row_scale(v, g)
    kc, vc = codes(k, ks), codes(v, vs)
    if kem:
        return attend_kem(q, kc, ks, vc, vs, qpos)
    kq, vq = kc * ks, vc * vs
    if exact0:
        kq[0], vq[0] = k[0], v[0]
    return an.attend(q, k if part == "v" else kq, v if part == "k" else vq, qpos)


def row1_codes(k, v, cast=an.fp8, g: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """The ROW1 codes [.., 640] and the one scale per row [.., 1]."""
    row = ca.pack_views(k, v)
    s = row_scale(row, g)
    return codes(row, s, cast), s


def row1_arm(q, k, v, qpos, w: torch.Tensor, cast=an.fp8, g: float = 1.0, kem: bool = False, q32: bool = False,
             exact0: bool = False, part: str = "all") -> torch.Tensor:
    """ROW1 as an unchanged kernel reads it: both views with the scale s, and the folded query (BF16, or float32 with
    q32). part: "krot", "kun" or "v" takes only K_rot, only the K use of V_unrot, or only the V view from the codes."""
    row = ca.pack_views(k, v)
    c, s = row1_codes(k, v, cast, g)
    qf = ca.views_query(q, w)
    if not q32:
        qf = an.bf16(qf)
    if kem:
        kc, vc = ca.views_of(c)
        return ca.unpermute(attend_kem(qf, kc, s, vc, s, qpos))
    deq = c * s
    if exact0:
        deq[0] = row[0]
    kv, vv = ca.views_of(deq)
    if part == "krot":
        kv, vv = torch.cat([deq[..., :NROT], row[..., NROT:D]], -1), row[..., NROT:]
    elif part == "kun":
        kv, vv = torch.cat([row[..., :NROT], deq[..., NROT:D]], -1), row[..., NROT:]
    elif part == "v":
        kv = row[..., :D]
    return ca.unpermute(an.attend(qf, kv, vv, qpos))


def row2_scales(k, v, w_un: float, g: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """ROW2: vs = max(max(max|V|, max|K_rot| / |w|) / 448, 1e-6) per row and ks = float32(w x vs)."""
    rot = torch.tensor(ca.ROT, device=k.device)
    top = torch.maximum(v.abs().amax(-1, keepdim=True), k[..., rot].abs().amax(-1, keepdim=True) / abs(w_un))
    vs = (top / QMAX).clamp_min(SCALE_FLOOR).to(torch.float32)
    if g != 1.0:
        vs = (vs * g).to(torch.float32)
    return (w_un * vs).to(torch.float32), vs


def row2_arm(q, k, v, qpos, w_un: float, g: float = 1.0, kem: bool = False) -> torch.Tensor:
    """ROW2: codes of [K_rot / ks | V_unrot / vs | V_rot / vs]; the K view with ks, the V view with vs; the plain
    permuted query."""
    ks, vs = row2_scales(k, v, w_un, g)
    row = ca.pack_views(k, v)
    c = torch.cat([codes(row[..., :NROT], ks), codes(row[..., NROT:], vs)], -1)
    kc, vc = ca.views_of(c)
    qp = ca.views_query(q, torch.ones(D, device=q.device))
    o = attend_kem(qp, kc, ks, vc, vs, qpos) if kem else an.attend(qp, kc * ks, vc * vs, qpos)
    return ca.unpermute(o)


def row2std_arm(q, k, v, qpos, w_un: float) -> torch.Tensor:
    """The standard cache with the ROW2 scales of each row (V7)."""
    ks, vs = row2_scales(k, v, w_un)
    return an.attend(q, codes(k, ks) * ks, codes(v, vs) * vs, qpos)


def share(mask: torch.Tensor) -> float:
    return float(mask.double().mean())


def layer_metrics(cap: dict, w_mod: torch.Tensor | None, sc: dict, device: str) -> dict:
    """Every metric of one prompt and global layer (README Section 5). sc: the frozen E-54 scales of the layer."""
    q, k, v = (cap[n].to(device).float() for n in ("q", "k", "v"))
    qpos = cap["qpos"].to(device).long()
    form, w, dist = an.w_choice(an.w_from_data(k, v), w_mod)
    wu = w[torch.tensor(ca.UNROT, device=device)]
    w_un = float(wu[0])
    out = {"tokens": int(k.shape[0]), "w_form": form, "w_module": w_mod is not None,
           "w_match": bool(dist) and min(dist.values()) <= an.W_MATCH,
           "w_scalar": int(torch.unique(wu).numel()) == 1, "w_value": w_un,
           "identity_err": an.identity_err(k, v, w)}
    ref = an.attend(q, k, v, qpos)
    out["ref_finite"] = bool(torch.isfinite(ref).all())
    err = lambda x: an.rel(x, ref)   # noqa: E731
    e = {"STOCK": err(a54.std_arm(q, k, v, qpos, 1.0, 1.0)),
         "CALSTD_JOINT": err(a54.std_arm(q, k, v, qpos, sc["calstd_joint"]["ks"], sc["calstd_joint"]["vs"])),
         "TIED": err(a54.tied_arm(q, k, v, qpos, sc["tied"]["ks"], sc["tied"]["vs"])),
         "ROW2STD": err(row2std_arm(q, k, v, qpos, w_un)),
         "ROW1_E5M2": err(row1_arm(q, k, v, qpos, w, cast=an.e5m2)),
         "ROW1_ID": err(row1_arm(q, k, v, qpos, w, cast=identity)),
         "ROW1_Q32": err(row1_arm(q, k, v, qpos, w, q32=True)),
         "PTH_X0": err(pth_arm(q, k, v, qpos, exact0=True)), "ROW1_X0": err(row1_arm(q, k, v, qpos, w, exact0=True)),
         "ROW1_KROT": err(row1_arm(q, k, v, qpos, w, part="krot")),
         "ROW1_KUN": err(row1_arm(q, k, v, qpos, w, part="kun")),
         "ROW1_V": err(row1_arm(q, k, v, qpos, w, part="v")),
         "PTH_K": err(pth_arm(q, k, v, qpos, part="k")), "PTH_V": err(pth_arm(q, k, v, qpos, part="v"))}
    out["errj"] = {"PTH": [err(pth_arm(q, k, v, qpos, g)) for g in FACTORS],
                   "ROW1": [err(row1_arm(q, k, v, qpos, w, g=g)) for g in FACTORS],
                   "ROW2": [err(row2_arm(q, k, v, qpos, w_un, g)) for g in FACTORS]}
    e |= {a: out["errj"][a][0] for a in JARMS}
    out["err"] = e
    ok = {"PTH": pth_arm(q, k, v, qpos, kem=True), "ROW1": row1_arm(q, k, v, qpos, w, kem=True),
          "ROW2": row2_arm(q, k, v, qpos, w_un, kem=True),
          "STOCK": an.attend(q, an.fp8(k), an.fp8(v), qpos, torch.bfloat16)}   # (A1, B3) P rounded to BF16
    out["errk"] = {a: an.rel(x, ref) for a, x in ok.items()}
    out["nonfinite_kem"] = sum(int((~torch.isfinite(x)).sum()) for x in ok.values())
    del ok
    # reported: where the codes are subnormal and where the largest value of a row is; the scale rule (V5)
    row = ca.pack_views(k, v)
    c1, s1 = row1_codes(k, v)
    sub = (c1.abs() < SUBNORMAL) & (c1 != 0)
    ks2, vs2 = row2_scales(k, v, w_un)
    ksp, vsp = row_scale(k), row_scale(v)
    kc, vc = codes(k, ksp), codes(v, vsp)
    out["sub"] = {"ROW1_K_rot": share(sub[..., :NROT]), "ROW1_V_unrot": share(sub[..., NROT:D]),
                  "ROW1_V_rot": share(sub[..., D:]), "ROW1_K_rot_zero": share(c1[..., :NROT] == 0),
                  "PTH_K": share((kc.abs() < SUBNORMAL) & (kc != 0))}
    out["row_max_in_K_rot"] = share(row.abs().argmax(-1) < NROT)
    out["same_v_scale"] = share(s1 == vsp)                   # the ROW1 scale equals the PTH V scale (R-126 B1)
    out["over_clamp"] = {"PTH": max(over_clamp(k, ksp), over_clamp(v, vsp)), "ROW1": over_clamp(row, s1),
                         "ROW2": max(over_clamp(row[..., :NROT], ks2.abs()), over_clamp(row[..., NROT:], vs2))}
    out["top_share"] = {"PTH_K": top_share(kc), "PTH_V": top_share(vc), "ROW1": top_share(c1)}
    out["scale_floor_rows"] = share(s1 <= SCALE_FLOOR)
    return out


# ---------------------------------------------------------------- step 1: metrics


def first_cell_problem(cell: dict, old: dict | None) -> str | None:
    """(A1, F5) The stop after the first E-54 cell: the three E-54 arms and ROW1_ID against the E-54 record."""
    if old is None:
        return "the E-54 record has no such cell"
    bad = {a: a54.ratio(cell["err"][a], old["err"][b]) for a, b in REPL.items()
           if not abs(a54.ratio(cell["err"][a], old["err"][b]) - 1.0) <= REPL_CELL_TOL}
    return f"cell errors outside {REPL_CELL_TOL} of the E-54 record: {bad}" if bad else None


def compute_metrics(name: str, res: Path, device: str = "cpu", limit: int | None = None) -> dict:
    """metrics_<name>.json from the capture files of one set. The file is written after each cell, and a cell that is
    in the file is not computed again. The SHA-256 of each capture file is checked against the run record (V1)."""
    cap_dir, run_path, ref_path = SETS[name]
    run = load(run_path, {})
    want = {(str(e["id"]), str(layer)): f.get("sha256") for e in run.get("prompts", [])
            for layer, f in (e.get("files") or {}).items()}
    record = load(ref_path, {}).get("prompts") or {} if ref_path else {}
    scales = json.loads(a54.SCALES.read_text())["layers"]
    path = res / f"metrics_{name}.json"
    out = load(path, {"set": name, "started_at": now(), "device": device, "torch": torch.__version__, "cells": {},
                      "sha_checked": 0, "sha_mismatch": [], "errors": [], "seconds": 0.0})
    out["scales_sha256"] = sha_file(a54.SCALES)
    kn_path = cap_dir / "k_norm.pt"
    knorm = torch.load(kn_path, map_location="cpu", weights_only=True) if kn_path.exists() else {}
    files = sorted(cap_dir.glob("p*/L*.pt"), key=lambda f: (int(f.parent.name[1:]), int(f.stem[1:])))
    out["files_found"] = len(files)
    done = 0
    for f in files:
        pid, layer = f.parent.name[1:], str(int(f.stem[1:]))
        key = f"{pid}/{layer}"
        if key in out["cells"]:
            continue
        if limit is not None and done >= limit:
            break
        t0 = time.time()
        try:
            b = f.read_bytes()
            got = hashlib.sha256(b).hexdigest()
            if want.get((pid, layer)) != got:
                out["sha_mismatch"].append(key)
            else:
                out["sha_checked"] += 1
            cap = torch.load(io.BytesIO(b), map_location=device, weights_only=True)
            del b
            cell = layer_metrics(cap, knorm.get(int(layer)), scales[layer], device)
            cell["seconds"] = round(time.time() - t0, 1)
            out["cells"][key] = cell
        except Exception as exc:  # recorded; V1 then fails for this cell
            out["errors"].append(f"{key}: {type(exc).__name__}: {exc}"[:300])
        out["seconds"] = round(out["seconds"] + time.time() - t0, 1)
        if ref_path and len(out["cells"]) == 1 and key in out["cells"]:
            problem = first_cell_problem(out["cells"][key], (record.get(pid) or {}).get(layer))
            if problem:
                out["status"], out["stopped"] = "stopped_first_cell", problem
                path.write_text(json.dumps(out, indent=1))
                raise SystemExit(f"STOP after the first cell ({key}): {problem}")
        path.write_text(json.dumps(out, indent=1))
        done += 1
        print(f"{name} {key}: {time.time() - t0:.1f} s", flush=True)
    out["status"] = "ok" if len(out["cells"]) == len(files) and not out["errors"] else "partial"
    out["finished_at"] = now()
    path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- step 2: checks and rule

med, rms, ratio = a54.med, a54.rms, a54.ratio


def table(met: dict) -> tuple[dict[str, list[dict]], list[str]]:
    """rows[layer] = the cells of the layer in prompt order; the prompt ids."""
    cells = met.get("cells") or {}
    pids = sorted({key.split("/")[0] for key in cells}, key=int)
    layers = sorted({key.split("/")[1] for key in cells}, key=int)
    complete = [p for p in pids if all(f"{p}/{layer}" in cells for layer in layers)]
    return {layer: [cells[f"{p}/{layer}"] for p in complete] for layer in layers}, complete


def pooled(rows: dict[str, list[dict]], key: str, arm: str, j: int | None = None) -> float:
    """P_X: the RMS over the layers of the median over the prompts of the error of arm X (factor j of `errj`)."""
    pick = (lambda r: r[key][arm]) if j is None else (lambda r: r[key][arm][j])
    return rms([med([pick(r) for r in rs]) for rs in rows.values()])


def prompt_ratio(rows, pids, arm, base) -> dict[str, float]:
    return {p: ratio(rms([rs[i]["err"][arm] for rs in rows.values()]), rms([rs[i]["err"][base] for rs in rows.values()]))
            for i, p in enumerate(pids)}


def layer_ratio(rows, arm, base) -> dict[str, float]:
    return {layer: med([ratio(r["err"][arm], r["err"][base]) for r in rs]) for layer, rs in rows.items()}


def stats(rows: dict[str, list[dict]], pids: list[str]) -> dict:
    """The decision statistics and the reported numbers (README Sections 5 and 7)."""
    P = {a: pooled(rows, "err", a) for a in ARMS}
    Pk = {a: pooled(rows, "errk", a) for a in ARMS_KEM}
    Pj = {a: [pooled(rows, "errj", a, j) for j in range(len(FACTORS))] for a in JARMS}
    out = {"P": P, "P_kem": Pk, "P_factors": Pj, "null_band_vs_STOCK": NULL_BAND, "bytes_per_row": BYTES}
    for x in ROWS:
        r_a = [ratio(a, b) for a, b in zip(Pj[x], Pj["PTH"])]
        finite = all(math.isfinite(v) for v in r_a)
        out[x] = {"R_A": r_a, "R_A_med": statistics.median(r_a) if finite else math.nan,
                  "R_A_max": max(r_a) if finite else math.nan, "R_A_min": min(r_a) if finite else math.nan,
                  "R_A_kem": ratio(Pk[x], Pk["PTH"]),
                  "g_A": layer_ratio(rows, x, "PTH"), "R_A_prompt": prompt_ratio(rows, pids, x, "PTH"),
                  # reported only (A1, B1): against STOCK, TIED and CALSTD_JOINT
                  "vs_STOCK": ratio(P[x], P["STOCK"]), "vs_STOCK_factors": [ratio(a, P["STOCK"]) for a in Pj[x]],
                  "vs_STOCK_kem": ratio(Pk[x], Pk["STOCK"]), "g_STOCK": layer_ratio(rows, x, "STOCK"),
                  "vs_STOCK_prompt": prompt_ratio(rows, pids, x, "STOCK"),
                  "vs_TIED": ratio(P[x], P["TIED"]), "vs_CALSTD_JOINT": ratio(P[x], P["CALSTD_JOINT"])}
    flat = [r for rs in rows.values() for r in rs]
    out |= {"e5m2": ratio(P["ROW1_E5M2"], P["ROW1"]), "tie": ratio(P["ROW2"], P["ROW2STD"]),
            "row1_vs_row2": ratio(P["ROW1"], P["ROW2"]), "row1_q32_vs_row1": ratio(P["ROW1_Q32"], P["ROW1"]),
            "x0": {"PTH": P["PTH_X0"], "ROW1": P["ROW1_X0"], "R_A": ratio(P["ROW1_X0"], P["PTH_X0"])},
            "parts": {a: P[a] for a in ("ROW1_KROT", "ROW1_KUN", "ROW1_V", "PTH_K", "PTH_V")},
            "pth": {"vs_STOCK": ratio(P["PTH"], P["STOCK"]), "vs_STOCK_factors": [ratio(a, P["STOCK"]) for a in Pj["PTH"]],
                    "g_STOCK": layer_ratio(rows, "PTH", "STOCK"), "vs_STOCK_prompt": prompt_ratio(rows, pids, "PTH", "STOCK"),
                    "vs_TIED": ratio(P["PTH"], P["TIED"]), "vs_CALSTD_JOINT": ratio(P["PTH"], P["CALSTD_JOINT"])},
            "per_layer_median": {a: {layer: med([r["err"][a] for r in rs]) for layer, rs in rows.items()}
                                 for a in ARMS},
            "sub_mean": {n: sum(r["sub"][n] for r in flat) / len(flat) for n in flat[0]["sub"]},
            "sub_max": {n: max(r["sub"][n] for r in flat) for n in flat[0]["sub"]},
            "row_max_in_K_rot_max": max(r["row_max_in_K_rot"] for r in flat),
            "same_v_scale_min": min(r["same_v_scale"] for r in flat),
            "over_clamp_max": {a: max(r["over_clamp"][a] for r in flat) for a in JARMS},
            "top_share_min": {a: min(r["top_share"][a] for r in flat) for a in flat[0]["top_share"]},
            "scale_floor_rows_max": max(r["scale_floor_rows"] for r in flat),
            "nonfinite_kem": sum(r["nonfinite_kem"] for r in flat)}
    cls = {}
    for name, ids in CLASSES.items():
        idx = [i for i, p in enumerate(pids) if int(p) in ids]
        if idx:
            sub = {layer: [rs[i] for i in idx] for layer, rs in rows.items()}
            cls[name] = {x: {"R_A": ratio(pooled(sub, "err", x), pooled(sub, "err", "PTH")),
                             "vs_STOCK": ratio(pooled(sub, "err", x), pooled(sub, "err", "STOCK"))} for x in ROWS}
    out["classes"] = cls
    return out


def replication(met: dict, rows: dict, pids: list[str], ref_path: Path | None) -> dict:
    """V2 and V8: the arms of REPL against the E-54 record, pooled (0.2%) and per cell (1%)."""
    if ref_path is None or not rows or not any(rows.values()):
        return {"V2": False, "V8": False, "why": "no data"}
    ref = load(ref_path, {}).get("prompts") or {}
    old_rows = {layer: [(ref.get(p) or {}).get(layer) for p in pids] for layer in rows}
    if any(c is None for rs in old_rows.values() for c in rs):
        return {"V2": False, "V8": False, "why": "the E-54 record does not have every cell"}
    pool = {a: ratio(pooled(rows, "err", a), pooled(old_rows, "err", b)) for a, b in REPL.items()}
    worst = {a: 0.0 for a in REPL}
    v8_bad = []
    for layer, rs in rows.items():
        for i, (cell, old) in enumerate(zip(rs, old_rows[layer])):
            for a, b in REPL.items():
                worst[a] = max(worst[a], abs(ratio(cell["err"][a], old["err"][b]) - 1.0))
            if not v8_cell_ok(cell["err"]["ROW1_ID"], old["err"]["BF16C"]):
                v8_bad.append(f"{pids[i]}/{layer}")
    pool_ok = lambda a: math.isfinite(pool[a]) and abs(pool[a] - 1.0) <= REPL_POOL_TOL   # noqa: E731
    return {"V2": all(pool_ok(a) and worst[a] <= REPL_CELL_TOL for a in ("STOCK", "CALSTD_JOINT", "TIED")),
            "V8": pool_ok("ROW1_ID") and not v8_bad, "pool_ratio": pool, "cell_worst": worst,
            "v8_cells_outside": v8_bad}


def v8_cell_ok(a: float, b: float) -> bool:
    """(A2) The V8 cell limit: |ROW1_ID - BF16C| <= 1% of BF16C + 1e-6. Run 1 showed that a purely relative limit
    cannot hold in layer 29, where w = 2^-4 makes the BF16C error float32 noise (about 2e-7)."""
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= REPL_CELL_TOL * b + V8_ABS_FLOOR


def passes(s: dict) -> list[str]:
    """The failed PASS conditions of one row design (README Section 7 after A1); an empty list means PASS."""
    bad = []
    if not s["R_A_med"] <= POOL_PASS:
        bad.append(f"R_A_med {s['R_A_med']:.3f} above {POOL_PASS}")
    if not s["R_A_max"] <= POOL_FAIL:
        bad.append(f"R_A_max {s['R_A_max']:.3f} above {POOL_FAIL}")
    if not s["R_A_kem"] <= POOL_PASS:
        bad.append(f"R_A_kem {s['R_A_kem']:.3f} above {POOL_PASS}")
    bad += [f"layer {layer}: g_A {x:.3f} above {LAYER_PASS}" for layer, x in s["g_A"].items() if not x <= LAYER_PASS]
    bad += [f"prompt {p}: R_A {x:.3f} above {PROMPT_PASS}" for p, x in s["R_A_prompt"].items() if not x <= PROMPT_PASS]
    return bad


def fails(s: dict) -> bool:
    return not s["R_A_med"] <= POOL_FAIL


def summarize(res: Path) -> dict:
    met = load(res / "metrics_e54.json", {})
    met53 = load(res / "metrics_e53.json", {})
    rows, pids = table(met)
    flat = [r for rs in rows.values() for r in rs]
    full = bool(flat) and len(met.get("cells") or {}) == FILES["e54"] and len(pids) == 7
    st = stats(rows, pids) if full else {}
    rep = (replication(met, rows, pids, SETS["e54"][2]) if full
           else {"V2": False, "V8": False, "why": "the E-54 set is not complete"})
    sha_ok = all(m.get("sha_checked") == FILES[n] and not m.get("sha_mismatch") and not m.get("errors")
                 and m.get("scales_sha256") == a54.SCALES_SHA256 for n, m in (("e54", met), ("e53", met53)))
    cells_all = flat + list((met53.get("cells") or {}).values())
    finite = bool(st) and all(math.isfinite(st[x][n]) for x in ROWS for n in ("R_A_med", "R_A_max", "R_A_kem"))
    checks = {"V1": sha_ok,
              "V2": bool(rep.get("V2")),
              "V3": bool(cells_all) and all(c["identity_err"] <= ID_TOL and c["w_module"] and c["w_match"]
                                            and c["w_scalar"] and c["w_value"] > 0 for c in cells_all),
              "V4": (bool(cells_all) and all(c["ref_finite"] for c in cells_all) and finite
                     and st.get("nonfinite_kem") == 0),
              "V5": (bool(st) and all(x <= 1.0 + CLAMP_TOL for x in st["over_clamp_max"].values())
                     and all(x >= TOP_SHARE for x in st["top_share_min"].values())),
              "V6": bool(st) and st["e5m2"] >= E5M2_MIN,
              "V7": bool(st) and TIE_LO <= st["tie"] <= TIE_HI,
              "V8": bool(rep.get("V8")),
              "_replication": rep, "_e5m2": st.get("e5m2"), "_tie": st.get("tie"),
              "_over_clamp": st.get("over_clamp_max"), "_top_share": st.get("top_share_min"),
              "_sha": {n: {"checked": m.get("sha_checked"), "mismatch": m.get("sha_mismatch"), "errors": m.get("errors")}
                       for n, m in (("e54", met), ("e53", met53))}}
    rows53, pids53 = table(met53)
    full53 = bool(rows53) and any(rows53.values()) and len(met53.get("cells") or {}) == FILES["e53"]
    return {"checks": checks, "stats": st, "stats_e53": stats(rows53, pids53) if full53 else {}, "prompts": pids,
            "prompts_e53": pids53, "seconds": {"e54": met.get("seconds"), "e53": met53.get("seconds")},
            "torch": met.get("torch"), "platform": platform.platform(), "factors": list(FACTORS)}


def verdict(s: dict) -> str:
    """The frozen rule (README Sections 6 and 7 after A1)."""
    c = s["checks"]
    names = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
    bad = [k for k in names if not c.get(k)]
    if bad:
        why = {"V1": f"capture files or scales file: {c.get('_sha')}", "V2": f"replication {c.get('_replication')}",
               "V3": "the identity K_unrot = w x V_unrot, the scalar module w or its sign", "V4": "a value is not finite",
               "V5": f"the scale rule: clamp {c.get('_over_clamp')}, top share {c.get('_top_share')}",
               "V6": f"E5M2 sensitivity {c.get('_e5m2')}", "V7": f"tie control {c.get('_tie')}",
               "V8": f"ROW1 with no cast against BF16C: {c.get('_replication')}"}
        return (f"INVALID ({', '.join(bad)}: " + "; ".join(f"{k} {why[k]}" for k in bad)[:900]
                + f"; action: {ACTION['invalid']})")
    st = s["stats"]
    tag = "; ".join(f"{x}: R_A median {st[x]['R_A_med']:.3f} ({st[x]['R_A_min']:.3f} to {st[x]['R_A_max']:.3f} over "
                    f"the 8 factors; KEM {st[x]['R_A_kem']:.3f}); against STOCK {st[x]['vs_STOCK']:.3f}, against TIED "
                    f"{st[x]['vs_TIED']:.3f}" for x in ROWS)
    tag += (f"; PTH against STOCK {st['pth']['vs_STOCK']:.3f} (random-phase null: median {NULL_BAND['median']}, 95th "
            f"percentile {NULL_BAND['p95']}); P PTH {st['P']['PTH']:.4f}, STOCK {st['P']['STOCK']:.4f}, TIED "
            f"{st['P']['TIED']:.4f}, ROW1 {st['P']['ROW1']:.4f}, ROW2 {st['P']['ROW2']:.4f}; ROW1/ROW2 "
            f"{st['row1_vs_row2']:.3f}; subnormal K_rot codes of ROW1 {st['sub_mean']['ROW1_K_rot']:.4f}; E5M2 "
            f"{st['e5m2']:.2f}; tie {st['tie']:.3f}")
    bad1, bad2 = passes(st["ROW1"]), passes(st["ROW2"])
    if not bad1:
        return f"GO ({tag}; action: {ACTION['GO']})"
    if not bad2:
        return f"PIVOT (ROW1: {'; '.join(bad1)[:400]}; {tag}; action: {ACTION['PIVOT']})"
    if fails(st["ROW1"]) and fails(st["ROW2"]):
        return f"KILL ({tag}; action: {ACTION['KILL']})"
    return (f"INCONCLUSIVE (ROW1: {'; '.join(bad1)[:400]}; ROW2: {'; '.join(bad2)[:400]}; {tag}; action: "
            f"{ACTION['other']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, type=Path, help="the results folder of this run")
    ap.add_argument("--metrics", nargs="*", default=[], choices=sorted(SETS), help="compute the metrics of these sets")
    ap.add_argument("--limit-cells", type=int, help="compute at most this number of new cells per set")
    a = ap.parse_args()
    a.res.mkdir(parents=True, exist_ok=True)
    for name in a.metrics:
        compute_metrics(name, a.res, "cpu", a.limit_cells)
    s = summarize(a.res)
    s["decision"] = verdict(s)
    s["code_sha256"] = {p.name: sha_file(p) for p in (Path(__file__), Path(a54.__file__), Path(an.__file__),
                                                      Path(ca.__file__))}
    (a.res / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    (a.res / "decision.txt").write_text(s["decision"] + "\n")
    print(s["decision"])


if __name__ == "__main__":
    main()
