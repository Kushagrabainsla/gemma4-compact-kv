"""E-72 (README_E72.md Section 7): the three frozen words from `<res>/{EV,C2}/{p_run,d_run}.json` and the P stdout
files.

Usage: analyze_e72.py <results dir>   -> <results dir>/summary.json, one line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

C_E71, S_E71 = 7.50915, (5.45755 + 5.45925) / 2              # E-71 tokens per work step (README Section "Result")
BAND = (1.29524, 1.37748)                                     # the frozen E-71 band (interval with the 1% floor)
REPEAT, DRIFT, SAT_MIN, PROBES = 0.005, 0.02, 0.95, 8
LEASE_LINES = ("Releasing expired KV blocks", "Potentially invalid KV blocks")
LEASE_SET = "'kv_lease_duration': 600"                       # in P's vLLM start line (non-default args)


def load(res: Path) -> dict:
    out = {}
    for arm in ("EV", "C2"):
        out[arm] = {}
        for role in ("p", "d"):
            f = res / arm / f"{role}_run.json"
            out[arm][role] = json.loads(f.read_text()) if f.exists() else None
        f = res / arm / "p_stdout.txt"
        text = f.read_text(errors="replace") if f.exists() else ""
        out[arm]["lease_lines"] = sum(text.count(s) for s in LEASE_LINES)
        out[arm]["lease_set"] = LEASE_SET in text              # A1 (R-121 F5): the positive control
    return out


def ok(r) -> bool:
    return bool(r) and r.get("status") == "ok"


def words(arms: dict) -> dict:
    ev, c2 = arms["EV"], arms["C2"]
    roles = [r for a in (ev, c2) for r in (a["p"], a["d"]) if r]
    w = (c2["d"] or {}).get("window") or {}
    tps = w.get("tokens_per_step")
    pev, pc2 = (ev["d"] or {}).get("probe") or [], (c2["d"] or {}).get("probe") or []
    v = {"W1": len(roles) == 4 and all("A100" in (r.get("gpu") or "") and "80GB" in (r.get("gpu") or "") for r in roles),
         "W2": len(roles) == 4 and all(r.get("manifest_ok") for r in roles),
         # A1 (R-121 B1, F5): W3c (C2 only) gates Q2; W3 (both arms) gates Q3; each needs the lease line in P's log
         "W3c": c2["lease_lines"] == 0 and c2["lease_set"] and ok(c2["p"]),
         "W3": all(a["lease_lines"] == 0 and a["lease_set"] for a in (ev, c2)) and ok(ev["p"]) and ok(c2["p"]),
         "W4": (w.get("saturated_share") or 0) >= SAT_MIN,
         "W5": bool(c2["d"]) and (c2["d"].get("status_window") == "ok" or ok(c2["d"]))
               and w.get("work_steps") == 20_000 and w.get("early_finishes") == 0 and (w.get("completed") or 0) >= 1,
         "W6": all(((a["d"] or {}).get("kv") or {}).get("pool_blocks") == 6_144 for a in (ev, c2)),
         "W7": ok(ev["d"]) and ok(c2["d"]) and len(pev) == PROBES and len(pc2) == PROBES
               and all(p.get("finish_reason") in ("stop", "length") for p in pev + pc2),
         "W8": all(((r.get("spec") or {}).get("applied") is True) and r.get("mode") == "VIEWS"
                   and (r.get("kv") or {}).get("layout") == "LBHNC" for r in (c2["p"], c2["d"]) if r)
               and all(not (r.get("spec") or {}).get("applied") and r.get("mode") == "VIEWS"
                       for r in (ev["p"], ev["d"]) if r)
               and all(r for r in (c2["p"], c2["d"], ev["p"], ev["d"]))
               # A1 (R-121 F4): the hooks are in place, with no problem, in every role
               and all(bool((r.get("hooks") or {}).get("hooked")) and (r.get("hooks") or {}).get("problems") == []
                       for r in (c2["p"], c2["d"], ev["p"], ev["d"]) if r)}
    out: dict = {"checks": v, "c2_window": w, "lease_lines": {a: arms[a]["lease_lines"] for a in ("EV", "C2")}}

    def need(*keys):
        bad = [k for k in keys if not v[k]]
        return "INVALID (" + ", ".join(bad) + ")" if bad else None

    if (bad := need("W1", "W2", "W4", "W5", "W6")) or tps is None:
        out["Q1"] = bad or "INVALID (no window)"
    else:
        out["d"] = tps / C_E71 - 1
        out["Q1"] = "REPEATS" if abs(out["d"]) <= REPEAT else "DRIFTS" if abs(out["d"]) <= DRIFT else "DIFFERS"
    if (bad := need("W1", "W2", "W3c", "W4", "W5", "W6", "W8")) or tps is None:
        out["Q2"] = bad or "INVALID (no window)"
    else:
        out["R_clean"] = tps / S_E71
        out["Q2"] = ("STEP-BELOW-CLEAN" if out["R_clean"] < BAND[0] else
                     "STEP-ABOVE-CLEAN" if out["R_clean"] > BAND[1] else "STEP-CONFIRMS-CLEAN")
    same = [a["tokens"] == b["tokens"] for a, b in zip(pev, pc2)]
    out["probe_same"] = sum(same)
    out["probe_first_diff"] = {i: next((j for j, (x, y) in enumerate(zip(pev[i]["tokens"], pc2[i]["tokens"])) if x != y),
                                       min(len(pev[i]["tokens"]), len(pc2[i]["tokens"])))
                               for i, s in enumerate(same) if not s}
    out["gold"] = {"EV": sum(p.get("gold_in_text", False) for p in pev), "C2": sum(p.get("gold_in_text", False) for p in pc2)}
    # A1 (R-121 B2): reported, not a word: probes whose sampled-token logprobs are bit-equal, and the tokens compared
    out["logprobs_equal"] = sum(a.get("logprobs") is not None and a.get("logprobs") == b.get("logprobs")
                                for a, b in zip(pev, pc2))
    out["tokens_compared"] = sum(min(len(a["tokens"]), len(b["tokens"])) for a, b in zip(pev, pc2))
    out["num_cached_tokens"] = {"EV": [p.get("num_cached_tokens") for p in pev],
                                "C2": [p.get("num_cached_tokens") for p in pc2]}
    if bad := need("W1", "W2", "W3", "W6", "W7", "W8"):
        out["Q3"] = bad
    else:
        out["Q3"] = "EXACT" if out["probe_same"] == PROBES else "NOT-EXACT"
    return out


def main() -> None:
    res = Path(sys.argv[1])
    out = words(load(res))
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"E-72: Q1 {out['Q1']} (d {out.get('d')}) | Q2 {out['Q2']} (R_clean {out.get('R_clean')}) | Q3 {out['Q3']} "
          f"({out['probe_same']} of {PROBES} same; gold {out['gold']}) | lease lines {out['lease_lines']} | "
          f"checks {out['checks']}")


if __name__ == "__main__":
    main()
