"""E-71 sizing (exploratory, before the design is frozen): the E-67 decode-node model with CPU offload (all200, L20),
timed with the E-68 A100 SXM read table (tag `meas`), for D3 and two shorter-prompt variants, at the decode pool that
a Gemma 4 31B decode engine gets on one 80 GB A100 (15 GiB = 6,144 blocks). It reports, per workload, the predicted
compact-over-stock decode-throughput ratio, preemptions, re-admission hits, and the prefill rate that one prefill
engine needs to keep the decode engine saturated. Not a decision: shorter horizon than E-64 (sizing only).

Usage: e71_predict.py <out json>   (nice 10, one core)
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "gemma4_kv_decode_scaling"), str(EXP / "gemma4_kv_capacity_model")]
import analyze_e65 as an  # noqa: E402
import e64  # noqa: E402
import e66  # noqa: E402
import e67  # noqa: E402
import sweep  # noqa: E402

FIT = EXP / "gemma4_kv_decode_scaling/results/e68-main-sxm-20261111/fit_amended.json"
POOLS = [6_144]
SEEDS = [0, 1, 2, 3, 4]
HORIZON, WARMUP = 60_000, 12_000
VARIANTS = {"D3": (32_768, 2_048), "D3H": (16_384, 2_048), "D3Q": (8_192, 2_048)}
_orig_draw = e64.draw


def draw(rng, name):
    if name in ("D3H", "D3Q"):
        p, o = VARIANTS[name]
        return sweep.lognormal(rng, p, 0.6, 256, 131_072), sweep.lognormal(rng, o, 0.6, 16, 16_384)
    return _orig_draw(rng, name)


def window_prediction(models, fits, pf) -> dict:
    """README Section 4: the frozen prediction for the engine window (D3H, seed 0, pool 6,144, steps [2,000, 22,000))."""
    row = e67.run_cell("all200", "D3H", 6_144, 0, models, fits, pf, horizon=22_000, warmup=2_000)
    q = row["r_meas_M_L20"]
    return {"workload": "D3H", "seed": 0, "pool": 6_144, "warmup": 2_000, "horizon": 22_000, "Q_pred": q,
            "band": [q - 0.05, q + 0.05], "r_step": row["r_step"], "included": row["included"],
            "Q_pred_gemm": row["r_meas_M_gemm_L20"], "Q_pred_max": row["r_meas_M_max_L20"],
            "preemptions": {lay: row[f"preemptions_{lay}"] for lay in ("stock", "compact")},
            "decode_per_step": {lay: row[f"decode_per_step_{lay}"] for lay in ("stock", "compact")},
            "decode_tok_s": {lay: row[f"dps_meas_M_{lay}"] for lay in ("stock", "compact")}}


def main() -> None:
    out = Path(sys.argv[1])
    e64.draw = draw
    fits, pf = e66.load()
    fits["meas"] = json.loads(FIT.read_text())["fit"]
    models = {t: an.Model(f) for t, f in fits.items()}
    if "--window" in sys.argv[2:]:
        res = window_prediction(models, fits, pf)
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res))
        return
    res: dict = {"note": "exploratory sizing for E-71; horizon 60,000, warmup 12,000; pool 6,144", "workloads": {}}
    t0 = time.time()
    for name in VARIANTS:
        rows = [e67.run_cell("all200", name, pool, seed, models, fits, pf, horizon=HORIZON, warmup=WARMUP)
                for seed in SEEDS for pool in POOLS]
        p_med, o_med = VARIANTS[name]
        mean_ratio = (p_med * 1.197) / (o_med * 1.197)          # lognormal sigma 0.6: mean = median x e^0.18
        # one pool only, so the E-64 label rule (3 pools) does not apply: report the per-seed ratios directly
        q = sorted(r["r_meas_M_L20"] for r in rows)
        w = {"Q_M_per_seed": q, "Q_M_by_seed": {str(r["seed"]): r["r_meas_M_L20"] for r in rows},
             "Q_M": {"median": st.median(q), "min": q[0], "max": q[-1]},
             "Q_M_gemm_median": st.median(r["r_meas_M_gemm_L20"] for r in rows),
             "Q_M_pf_equal_median": st.median(r["r_meas_M_pf_equal_L20"] for r in rows),
             "r_step": st.median(r["r_step"] for r in rows), "included": sum(r["included"] for r in rows)}
        for lay in ("stock", "compact"):
            w[f"preempt_per_kstep_{lay}"] = st.median(r[f"preemptions_{lay}"] / (HORIZON - WARMUP) * 1_000 for r in rows)
            w[f"readmits_{lay}"] = st.median(r[f"readmits_{lay}"] for r in rows)
            w[f"hit_share_{lay}"] = st.median(r[f"hits_{lay}"] / max(1, r[f"readmits_{lay}"]) for r in rows)
            w[f"decode_tok_s_{lay}"] = st.median(r[f"dps_meas_M_{lay}"] for r in rows)
            w[f"pool_bound_{lay}"] = st.median(r[f"pool_bound_share_{lay}"] for r in rows)
        # one prefill engine must prefill (prompt / output) tokens for every decoded token
        w["prefill_tok_s_needed_stock"] = w["decode_tok_s_stock"] * mean_ratio
        w["prefill_tok_s_needed_compact"] = w["decode_tok_s_compact"] * mean_ratio
        res["workloads"][name] = w
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"{name}: Q_M median {w['Q_M']['median']:.3f} [{w['Q_M']['min']:.3f}, {w['Q_M']['max']:.3f}] "
              f"gemm {w['Q_M_gemm_median']:.3f} pf_equal {w['Q_M_pf_equal_median']:.3f} | decode tok/s "
              f"stock {w['decode_tok_s_stock']:.0f} compact {w['decode_tok_s_compact']:.0f} | prefill needed "
              f"{w['prefill_tok_s_needed_compact']:.0f} tok/s | preempt/kstep {w['preempt_per_kstep_stock']:.2f} "
              f"{w['preempt_per_kstep_compact']:.2f} | {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
