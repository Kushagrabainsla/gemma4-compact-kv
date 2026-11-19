"""E-73 (README_E73.md Section 7): the frozen words from `<res>/{S,C}/{p_run,d_run}.json`, the P stdout files and
`prediction_interval_e73.json`.

Usage: analyze_e73.py <results dir> <prediction_interval_e73.json>   -> <results dir>/summary.json, one line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("S", "C")
FLOOR, SAT_MIN, WINDOW = 0.01, 0.95, 20_000
R_D3H_ENGINE = (1.37571 + 1.37657) / 2                 # E-71 R_step and E-72 R_clean (README_E73 Section 5)
R_D3H_MODEL = 1.3637955535182213                       # prediction_interval.json, seed 0 depth 3
EXCESS_D3H = 1.37657 / R_D3H_MODEL - 1                 # E-72 (clean)
SHIFT_LO, SHIFT_HI = 0.75, 1.25
LEASE_LINES = ("Releasing expired KV blocks", "Potentially invalid KV blocks")
LEASE_SET = "'kv_lease_duration': 600"


def load(res: Path) -> dict:
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for role in ("p", "d"):
            f = res / arm / f"{role}_run.json"
            out[arm][role] = json.loads(f.read_text()) if f.exists() else None
        f = res / arm / "p_stdout.txt"
        text = f.read_text(errors="replace") if f.exists() else ""
        out[arm]["lease_lines"] = sum(text.count(s) for s in LEASE_LINES)
        out[arm]["lease_set"] = LEASE_SET in text
    return out


def ok(r) -> bool:
    return bool(r) and r.get("status") == "ok"


def words(arms: dict, pred: dict) -> dict:
    lo, hi = pred["r_step_interval"]
    band = [lo * (1 - FLOOR), hi * (1 + FLOOR)]
    r0 = pred["r_step_seed0_depth3"]
    out: dict = {"band": band, "model_seed0": r0, "lease_lines": {a: arms[a]["lease_lines"] for a in ARMS}}
    roles = [arms[a][r] for a in ARMS for r in ("p", "d") if arms[a][r]]
    win = {a: ((arms[a]["d"] or {}).get("window") or {}) for a in ARMS}
    done = all(ok(arms[a][r]) for a in ARMS for r in ("p", "d"))
    hooks_ok = all(bool((r.get("hooks") or {}).get("hooked")) and (r.get("hooks") or {}).get("problems") == []
                   for r in roles)
    v = {"W1": len(roles) == 4 and all("A100" in (r.get("gpu") or "") and "80GB" in (r.get("gpu") or "") for r in roles),
         "W2": len(roles) == 4 and all(r.get("manifest_ok") for r in roles),
         "W3": all(arms[a]["lease_lines"] == 0 and arms[a]["lease_set"] for a in ARMS),
         "W4": done and all((win[a].get("saturated_share") or 0) >= SAT_MIN for a in ARMS),
         "W5": done and all(win[a].get("work_steps") == WINDOW and win[a].get("early_finishes") == 0
                            and (win[a].get("completed") or 0) >= 1 for a in ARMS),
         "W6": done and all(((arms[a]["d"] or {}).get("kv") or {}).get("pool_blocks") == 6_144 for a in ARMS),
         "W7": done and hooks_ok
               and all((arms["C"][r].get("spec") or {}).get("applied") is True and arms["C"][r].get("mode") == "VIEWS"
                       and (arms["C"][r].get("kv") or {}).get("layout") == "LBHNC" for r in ("p", "d"))
               and all(not (arms["S"][r].get("spec") or {}).get("applied") and arms["S"][r].get("mode") == "OFF"
                       for r in ("p", "d"))}            # A1 (R-123 F1): the hook mode is recorded and checked
    out["checks"] = v
    out["windows"] = win
    if done and win["S"].get("tokens_per_step") and win["C"].get("tokens_per_step"):
        r = win["C"]["tokens_per_step"] / win["S"]["tokens_per_step"]
        out["R_step"] = r
        out["shift_engine"] = R_D3H_ENGINE - r
        out["shift_model"] = R_D3H_MODEL - r0
        out["shift_ratio"] = out["shift_engine"] / out["shift_model"]
        out["excess"] = r / r0 - 1
        out["excess_D3H"] = EXCESS_D3H
        if win["S"].get("decode_tok_s") and win["C"].get("decode_tok_s"):
            out["R_tput"] = win["C"]["decode_tok_s"] / win["S"]["decode_tok_s"]
            out["R_tput_vs"] = {"Q_graph": pred["Q_graph_interval"], "Q_eager": pred["Q_eager_interval"]}
    if not all(v.values()) or "R_step" not in out:
        bad = ", ".join(k for k, x in v.items() if not x) or "no window"
        out["Q1"] = out["Q2"] = f"INVALID ({bad})"
    else:
        r = out["R_step"]
        out["Q1"] = "STEP-BELOW" if r < band[0] else "STEP-ABOVE" if r > band[1] else "STEP-CONFIRMS"
        s = out["shift_ratio"]
        out["Q2"] = "SHIFT-SMALLER" if s < SHIFT_LO else "SHIFT-LARGER" if s > SHIFT_HI else "SHIFT-MATCHES"
    return out


def main() -> None:
    res, pred = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    out = words(load(res), pred)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"E-73: Q1 {out['Q1']} (R_step {out.get('R_step')}, band {out['band']}) | Q2 {out['Q2']} "
          f"(shift ratio {out.get('shift_ratio')}) | excess {out.get('excess')} (D3H {EXCESS_D3H:.4f}) | "
          f"R_tput {out.get('R_tput')} | lease lines {out['lease_lines']} | checks {out['checks']}")


if __name__ == "__main__":
    main()
