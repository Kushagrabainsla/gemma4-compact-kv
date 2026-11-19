"""E-65 analysis, part 1 (README Sections 5 to 7, as amended by A1): the GPU checks, the measured decode-step model W-M,
and the share of the byte saving that becomes step time (phi). Writes `fit.json` for part 2 (`e65_model.py`).

The W-M decode step of a batch with contexts c_1..c_B (B requests, mean context Lbar), in seconds:
  stock:   T_OTHER + 2 G_S(B, Lbar) + 10 SL(B, Lbar)
  compact: T_OTHER + 2 G_C(B, Lbar) + 10 SL(B, Lbar) + 2 H(B)
G: the measured read of 5 global layers; SL: of 5 sliding layers with the engine window (A1, R-101 B2); H: the hook
transform of 5 layers (graph times, medians of 3 rounds). T_OTHER = 50.733 ms (the E-59 stock graph step at batch 3
and 32K) - 2 G_S(3, 32K) - 10 SL(3, 32K): everything that is not attention, fixed over B (an assumption).
G and SL between grid points: piecewise linear in B within each L row (linear extrapolation past the last B of a row
from its last two points), then piecewise linear in L between the rows (the 1,024 row below 1,024 tokens; linear
extrapolation past the longest row). H: piecewise linear in B over the 1,024 row. A1 (R-101 F6): the point (1, 1,024)
is left out of the G rows (its caches fit in the A100 L2).

Usage: analyze_e65.py --res <dir with bench.json> [--out fit.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_kv_capacity_model")]

E59_STOCK_MS = 50.7333            # E-59 S_g: the stock graph decode step, batch 3, 32K (one A100 SXM)
E59_ATTN_MS = 10.67               # E-59 graph profile: attention kernels of the stock step (report-only T_OTHER)
E58_READ_MS = {"S32": 5.076, "C48": 4.582}      # E-58 R at D32, 10 layers (CUDA events, raw calls)
E58_HOOK_GRAPH_MS = 0.396         # E-58 H VIEWS graph, 10 layers, batch 3 (V6, report only)
TIE = (3, 32_768)
L2_POINT = (1, 1_024)             # A1 (R-101 F6): left out of the G rows
SPREAD_G, SPREAD_G_MIN_TOKENS = 0.03, 16_384
SPREAD_SH, SPREAD_SH_MIN_MS = 0.05, 0.2         # A1 (R-101 F3): SL and H rounds
V1_REL = 0.02                     # A1 (R-101 F4): the largest difference <= 2% of the output scale, and a scale > 0
TIE_TOL = 0.05
PHI_MIN_TOKENS = 2 ** 18
VLLM = "0.33.0rc1.dev229+g3966924d5"


def interp1(xs: list[float], ys: list[float], x: float, below: str = "extrapolate") -> float:
    """Piecewise linear; linear extrapolation past both ends (or the first value below the range if below='clamp')."""
    if len(xs) == 1:
        return ys[0]
    if x <= xs[0] and below == "clamp":
        return ys[0]
    i = bisect.bisect_left(xs, x)
    i = min(max(i, 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def interp2(rows: dict[int, tuple[list, list]], b: float, lbar: float) -> float:
    ls = sorted(rows)
    vals = [interp1(*rows[ln], b) for ln in ls]
    return max(interp1(ls, vals, lbar, below="clamp"), 0.0)


class Model:
    """The W-M decode-step model from `fit.json` (all times in seconds)."""

    def __init__(self, fit: dict):
        self.t_other = fit["t_other_s"]
        self.rows = {k: {int(ln): (row["B"], row["s"]) for ln, row in fit[k].items()}
                     for k in ("G_S32", "G_C48", "SL")}
        self.h = (fit["H"]["B"], fit["H"]["s"])
        self.cache: dict = {}

    def g(self, lo: str, b: float, lbar: float) -> float:
        return interp2(self.rows[f"G_{lo}"], b, lbar)

    def sl(self, b: float, lbar: float) -> float:
        return interp2(self.rows["SL"], b, lbar)

    def step(self, layout: str, b: int, lbar: float, t_other: float | None = None) -> float:
        key = (layout, b, int(lbar) // 64, t_other)
        if key not in self.cache:
            lo = "C48" if layout == "compact" else "S32"
            t = (self.t_other if t_other is None else t_other) + 2 * self.g(lo, b, lbar) + 10 * self.sl(b, lbar)
            if layout == "compact":
                t += 2 * interp1(*self.h, b)
            self.cache[key] = t
        return self.cache[key]


def spread(xs: list[float]) -> float:
    return max(xs) / min(xs) - 1


def rows_of(pts: list[dict], value, skip=()) -> dict:
    out: dict = {}
    for ln in sorted({p["length"] for p in pts}):
        row = sorted((p["batch"], value(p)) for p in pts
                     if p["length"] == ln and (p["batch"], p["length"]) not in skip)
        if row:
            out[str(ln)] = {"B": [b for b, _ in row], "s": [s for _, s in row]}
    return out


def analyze(bench: dict) -> dict:
    meta, pts = bench["meta"], bench["points"]
    stub, reh = meta.get("stub", False), meta.get("rehearsal", False)
    plan = [tuple(p) for p in meta.get("plan", [])]
    core = [tuple(p) for p in meta.get("core", plan)]
    have = {(p["batch"], p["length"]): p for p in pts}
    checks: dict = {}
    checks["V1"] = bool(pts) and all(p["v1"]["out_scale"] > 0 and (p["v1"]["identical"] or
                                     p["v1"]["max_abs"] <= V1_REL * p["v1"]["out_scale"]) for p in pts)
    checks["V2"] = stub or reh or ("A100" in meta.get("gpu", "") and meta.get("vllm") == VLLM)
    g_bad = [(p["batch"], p["length"]) for p in pts if p["tokens"] >= SPREAD_G_MIN_TOKENS
             and any(spread(p["G"][lo]) > SPREAD_G for lo in ("S32", "C48"))]
    sh_bad = [(p["batch"], p["length"], k) for p in pts for k in ("SL", "H")
              if st.median(p[k]) >= SPREAD_SH_MIN_MS and spread(p[k]) > SPREAD_SH]
    checks["V3"] = stub or (not g_bad and not sh_bad)          # the stub times are CPU noise
    checks["V3_failed_points"] = g_bad + sh_bad
    missing_core = [p for p in core if p not in have]
    checks["V5"] = bool(plan) and not missing_core
    checks["V5_missing_core"] = missing_core
    checks["extrapolated"] = [p for p in plan if p not in have and p not in core]
    tie = have.get(TIE)
    if tie is not None:
        gs, gc = 2 * st.median(tie["G"]["S32"]), 2 * st.median(tie["G"]["C48"])
        checks["V4_values_ms"] = [round(gs, 3), round(gc, 3), E58_READ_MS["S32"], E58_READ_MS["C48"]]
        checks["V4"] = stub or reh or (abs(gs / E58_READ_MS["S32"] - 1) <= TIE_TOL
                                        and abs(gc / E58_READ_MS["C48"] - 1) <= TIE_TOL)
        checks["V6_report_ms"] = [round(2 * st.median(tie["H"]), 3), E58_HOOK_GRAPH_MS]
    else:
        checks["V4"] = stub
    checks["tie_status"] = bench.get("tie_status")
    failed = [k for k in ("V1", "V2", "V3", "V4", "V5") if not checks[k]]

    fit: dict = {"G_S32": rows_of(pts, lambda p: st.median(p["G"]["S32"]) / 1e3, skip=(L2_POINT,)),
                 "G_C48": rows_of(pts, lambda p: st.median(p["G"]["C48"]) / 1e3, skip=(L2_POINT,)),
                 "SL": rows_of(pts, lambda p: st.median(p["SL"]) / 1e3)}
    l0 = min((p["length"] for p in pts), default=0)
    base = sorted((p["batch"], p) for p in pts if p["length"] == l0)
    fit["H"] = {"B": [b for b, _ in base], "s": [st.median(p["H"]) / 1e3 for _, p in base], "length": l0}
    if tie is not None:
        attn = 2 * st.median(tie["G"]["S32"]) / 1e3 + 10 * st.median(tie["SL"]) / 1e3
        fit["t_other_s"] = E59_STOCK_MS / 1e3 - attn
        fit["t_other_profile_s"] = (E59_STOCK_MS - E59_ATTN_MS) / 1e3
        fit["attention_model_ms_at_tie"] = round(1e3 * attn, 3)
        fit["sliding_50_ms_at_tie"] = round(10 * st.median(tie["SL"]), 3)
    else:                                          # stub: a fixed part of the size of E-59
        fit["t_other_s"] = fit["t_other_profile_s"] = 0.040
    import e64
    table = []
    for p in sorted(pts, key=lambda q: (q["length"], q["batch"])):
        b, ln = p["batch"], p["length"]
        gs, gc = st.median(p["G"]["S32"]) / 1e3, st.median(p["G"]["C48"]) / 1e3
        ts = fit["t_other_s"] + 2 * gs + 10 * st.median(p["SL"]) / 1e3
        tc = ts - 2 * gs + 2 * gc + 2 * st.median(p["H"]) / 1e3
        d = {"decode_ctx": [ln] * b, "prefill_tokens": 0}
        ra = e64.step_time(d, e64.G_COMPACT, 0.5) / e64.step_time(d, e64.G_STOCK, 0.5)
        rho = tc / ts
        table.append({"B": b, "L": ln, "tokens": b * ln, "read_ratio": gc / gs, "rho": rho, "rho_A": ra,
                      "phi": (1 - rho) / (1 - ra), "global_share": 2 * gs / ts, "t_stock_ms": 1e3 * ts,
                      "t_compact_ms": 1e3 * tc, "hook_ms": 2 * st.median(p["H"]),
                      "sliding_ms": 10 * st.median(p["SL"]), "sequential": p.get("sequential", False),
                      "spread": max(spread(p["G"]["S32"]), spread(p["G"]["C48"]))})
    big = [r["phi"] for r in table if r["tokens"] >= PHI_MIN_TOKENS]
    phi = st.median(big) if big else None
    word = "REHEARSAL" if (stub or reh) else ("INVALID" if failed else "VALID")
    return {"word": word, "failed": failed, "checks": checks, "fit": fit, "table": table, "Phi": phi,
            "Phi_points": len(big)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    bench = json.loads((args.res / "bench.json").read_text())
    res = analyze(bench)
    (args.out or args.res / "fit.json").write_text(json.dumps(res, indent=1) + "\n")
    ph = "NA" if res["Phi"] is None else f"{res['Phi']:.3f}"
    print(f"E-65 GPU: {res['word']} | Phi {ph} over {res['Phi_points']} points | T_OTHER "
          f"{1e3 * res['fit']['t_other_s']:.2f} ms | checks {res['checks']} | failed {res['failed'] or 'none'}")


if __name__ == "__main__":
    main()
