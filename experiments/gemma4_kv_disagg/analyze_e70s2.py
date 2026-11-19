"""E-70 stage 2 (README_stage2.md Section 7): the frozen word from `<res>/{STOCK2,COMPACT2}/{p_run,d_run}.json`.

Usage: analyze_e70s2.py <results dir>   -> <results dir>/summary.json and one line on stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PAIRS = ("STOCK2", "COMPACT2")
WORKS_C, PROMPT_ONLY_C, MIN_EVENTS, MIN_RELOADS, MIN_PREEMPT = 0.9, 0.1, 10, 10, 10
CODE_WORDS = ("nixl", "offloading", "kv_offload", "kv_connector", "multi_connector", "compact_engine",
              "nixl_compact", "kv_cache_layout", "triton_attn")


def load(res: Path) -> dict:
    out = {}
    for pair in PAIRS:
        out[pair] = {}
        for role in ("p", "d"):
            f = res / pair / f"{role}_run.json"
            out[pair][role] = json.loads(f.read_text()) if f.exists() else None
    return out


def ok(r) -> bool:
    return bool(r) and r.get("status") == "ok"


def facts(d: dict) -> dict:
    cs = [c for c in d.get("checks") or [] if "checked" in c]
    # A1 (R-116 F1): a NIXL check counts only with blocks compared, a full-prompt hit and no failed load
    good = [c for c in cs if c["checked"] > 0 and c.get("hit", 0) >= c.get("prompt", 1) - 1 and not c.get("failed_load")]
    cov = d.get("coverage") or {}
    rl = d.get("reload") or {}
    return {"nixl_checked": sum(c["checked"] for c in cs), "nixl_mismatch": sum(c["mismatch"] for c in cs),
            "nixl_compared": len(good), "median_c": cov.get("median_c"), "min_c": cov.get("min_c"),
            "zero_hit_events": cov.get("zero_hit_events", 0), "decode_events": cov.get("decode_events", 0),
            "checked_reloads": rl.get("checked_reloads", 0), "mismatched_blocks": rl.get("mismatched_blocks", 0),
            "a1_strict_mismatch": rl.get("a1_strict_mismatch"), "recomputed_first_version":
            rl.get("recomputed_first_version"), "preemptions": (d.get("counters") or {}).get("preemptions", 0),
            "load_gb_per_s": (d.get("stats") or {}).get("load_gb_per_s"), "complete": d.get("n_complete"),
            "gold": sum(bool(a.get("gold_in_text")) for a in d.get("answers") or [])}


def word(pairs: dict, n: int = 8) -> dict:
    out: dict = {"pairs": {}}
    for pair, roles in pairs.items():
        p, d = roles["p"], roles["d"]
        e = {"p": (p or {}).get("status", "not run"), "d": (d or {}).get("status", "not run"),
             "error": ((d or {}).get("error") or (p or {}).get("error") or "")[:300]}
        if ok(p) and ok(d):
            e.update(facts(d))
        out["pairs"][pair] = e
    s, c = out["pairs"]["STOCK2"], out["pairs"]["COMPACT2"]
    roles = [r for pr in pairs.values() for r in pr.values() if r]
    v = {"V1": bool(roles) and all("L40S" in (r.get("gpu") or "") for r in roles if r.get("gpu")),
         "V2": bool(roles) and all(r.get("manifest_ok") for r in roles),
         "V3": s["p"] == "ok" and s["d"] == "ok" and s.get("preemptions", 0) >= MIN_PREEMPT
         and s.get("decode_events", 0) >= MIN_EVENTS and s.get("nixl_mismatch", 1) == 0
         and s.get("nixl_compared", 0) == n and s.get("complete") == n,               # A1 (R-116 F1)
         "V4": all(((r.get("spec") or {}).get("applied")
                    and ((r.get("kv") or {}).get("layout") or r.get("layout_env")) == "LBHNC")
                   for r in pairs["COMPACT2"].values() if r and r.get("gpu"))}
    out["checks"] = v
    cd = pairs["COMPACT2"]["d"] or {}
    tb = (cd.get("error") or "") + (cd.get("traceback") or "")
    if not all(v.values()):
        w = "INVALID (" + ", ".join(k for k, x in v.items() if not x) + ")"
    elif c["d"] != "ok" or c["p"] != "ok":
        w = "RELOAD-FAILS" if any(x in tb for x in CODE_WORDS) else "INVALID (COMPACT2 did not complete)"
    elif c.get("mismatched_blocks", 1) > 0 or c.get("nixl_mismatch", 1) > 0:
        w = "RELOAD-FAILS"
    elif c.get("decode_events", 0) < MIN_EVENTS or c.get("checked_reloads", 0) < MIN_RELOADS \
            or c.get("nixl_compared", 0) < n or c.get("complete") != n:
        w = "INVALID (too few COMPACT2 events, checks or complete requests)"
    elif (c["median_c"] is not None and c["median_c"] >= WORKS_C and c.get("zero_hit_events", 1) == 0
          and (c.get("min_c") or 0) >= 0.5):                       # A1 (R-116 F1): no hidden zero-hit reloads
        w = "RELOAD-WORKS"
    elif c["median_c"] is not None and c["median_c"] <= PROMPT_ONLY_C and c.get("zero_hit_events", 0) == 0:
        w = "RELOAD-PROMPT-ONLY"                                   # hits stop at the prompt end (not zero hits)
    else:
        w = "RELOAD-PARTIAL"
    out["word"] = w
    return out


def main() -> None:
    res = Path(sys.argv[1])
    out = word(load(res))
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"E-70 stage 2: {out['word']} | checks {out['checks']} | {out['pairs']}")


if __name__ == "__main__":
    main()
