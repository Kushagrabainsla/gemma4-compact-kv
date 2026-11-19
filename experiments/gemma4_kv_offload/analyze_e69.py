"""E-69 (README Section 7, as amended by A1): the frozen word from the arm records `<res>/<arm>/run_e69.json`, and
the rehearsal VERIFY items.

Usage: analyze_e69.py <results dir> [--rehearsal]   -> <results dir>/summary.json and one line on stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("C-C", "S-C", "S-R", "C-R", "C-C0", "C-CX", "C-C64")
PAIRS = (("S-C", "C-C"), ("S-R", "C-R"))
WORKS_C, PROMPT_ONLY_C = 0.9, 0.1
MIN_PREEMPT, MIN_RELOADS, MIN_DECODE_EVENTS = 10, 10, 10
START_LIMIT_S = 300
ENV_WORDS = ("OutOfMemory", "out of memory", "/dev/shm", "shm", "cudaHostRegister", "MemoryError", "Killed",
             "No space left")
CODE_WORDS = ("offloading", "kv_offload", "kv_connector", "compact_engine", "kv_cache_layout", "triton_attn",
              "worker_utils", "kv_cache_interface")


def load(res: Path) -> dict:
    out = {}
    for arm in ARMS:
        f = res / arm / "run_e69.json"
        out[arm] = json.loads(f.read_text()) if f.exists() else None
    return out


def ok(r: dict | None) -> bool:
    return bool(r) and r.get("status") == "ok"


def crash_kind(r: dict | None) -> str:
    """A1 F9: ENV (memory, /dev/shm, host registration) or CODE (the connector, the compact patch, the layout)."""
    txt = ((r or {}).get("error", "") + "\n" + (r or {}).get("traceback", ""))
    if any(w in txt for w in ENV_WORDS):
        return "ENV"
    if any(w in txt for w in CODE_WORDS):
        return "CODE"
    return "OTHER"


def checks(arms: dict, gpu_want: tuple[str, ...]) -> dict:
    c: dict = {}
    done = {a: r for a, r in arms.items() if ok(r) and not r.get("start_only")}
    c["V1"] = bool(done) and all(all(w in r["gpu"] for w in gpu_want) for r in done.values())
    c["V2"] = bool(done) and all(r.get("manifest_ok") for r in done.values())
    c["V3"] = bool(done) and all(r["stats"]["none_steps"] == 0 and r["stats"]["steps"] > 0
                                 and r["stats_preemptions"] == r["counters"]["preemptions"] for r in done.values())
    cc, sc = arms["C-C"], arms["S-C"]
    c["V4"] = (ok(cc) and ok(sc) and cc["counters"]["preemptions"] >= MIN_PREEMPT
               and sc["counters"]["preemptions"] >= MIN_PREEMPT
               and (cc.get("coverage") or {}).get("decode_events", 0) >= MIN_DECODE_EVENTS
               and cc["reload"]["checked_reloads"] >= MIN_RELOADS)
    v5 = {}
    for s, k in PAIRS:
        if ok(arms[s]) and ok(arms[k]):
            v5[f"{s}/{k}"] = arms[s]["eq_tokens"] == arms[k]["eq_tokens"]
    c["V5_pairs"] = v5
    c["V5"] = (v5.get("S-C/C-C") is True and all(v5.values())
               and all(r["counters"]["fix_reads"] > 0 for a, r in done.items() if a.startswith("C"))
               and all(r.get("eq_load_bytes", 0) == 0 for r in done.values()))      # A1 F5: no load in the EQ set
    c["V6"] = bool(done) and all(r.get("layout") == "LBHNC" for r in done.values())
    return c


def verify(arms: dict) -> dict:
    """The rehearsal VERIFY items that decide whether the main run may start (A1 F10)."""
    v: dict = {}
    c0, cx, c64, cc = arms["C-C0"], arms["C-CX"], arms["C-C64"], arms["C-C"]
    v["C-C0_prompt_only"] = ok(c0) and (c0.get("coverage") or {}).get("median_c") is not None \
        and c0["coverage"]["median_c"] <= PROMPT_ONLY_C
    v["C-CX_mismatch_found"] = ok(cx) and cx["reload"]["mismatched_blocks"] > 0
    v["C-C64_start_s"] = c64.get("engine_seconds") if c64 else None
    v["C-C64_ok"] = ok(c64) and (c64.get("engine_seconds") or 1e9) <= START_LIMIT_S
    v["load_bytes"] = ok(cc) and cc["stats"]["kv"].get("vllm:kv_offload_load_bytes", 0) > 0
    v["layout_lines"] = (cc or {}).get("layout_lines", [])[:12]
    v["kernel_block"] = (cc or {}).get("runner")
    v["all_pass"] = all(v[k] for k in ("C-C0_prompt_only", "C-CX_mismatch_found", "C-C64_ok", "load_bytes"))
    return v


def word(arms: dict, rehearsal: bool = False) -> dict:
    gpu_want = ("L40S",) if rehearsal else ("A100", "80GB")
    c = checks(arms, gpu_want)
    out: dict = {"checks": c, "arms": {a: (r or {}).get("status", "not run") for a, r in arms.items()}}
    cc, sc = arms["C-C"], arms["S-C"]
    for a in ("C-C", "S-C", "C-C0", "C-CX"):
        if ok(arms[a]):
            rl, kv = arms[a]["reload"], arms[a]["stats"].get("kv", {})
            cpu = (((arms[a].get("engine_kwargs") or {}).get("kv_transfer_config") or {})
                   .get("kv_connector_extra_config") or {}).get("cpu_bytes_to_use")
            out[a] = {"coverage": arms[a].get("coverage"), "mismatched_blocks": rl["mismatched_blocks"],
                      "a1_strict_mismatch": rl.get("a1_strict_mismatch"),            # R-112 F3
                      "recomputed_first_version": rl.get("recomputed_first_version"),
                      "recomputed_third": rl.get("recomputed_third"),
                      "checked_reloads": rl["checked_reloads"], "same_id_share": rl["same_id_share"],
                      "load_gb_per_s": arms[a]["stats"]["load_gb_per_s"],
                      "tier_fill": (kv.get("vllm:kv_offload_store_bytes", 0) / cpu) if cpu else None}   # R-112 N
    if not ok(sc):
        w = f"INVALID (S-C did not complete, {crash_kind(sc)}; CONNECTOR-FAILS-STOCK, exploratory)"
    elif not ok(cc):
        kind = crash_kind(cc)
        if kind == "ENV":
            w = "INVALID (ENV: C-C failed on memory, /dev/shm or host registration)"
        elif kind == "CODE" and (cc or {}).get("tries", 1) >= 2:
            w = "CONNECTOR-FAILS"
        else:
            w = f"INVALID (C-C not complete: {kind}, try {(cc or {}).get('tries', 0)})"
    elif cc["reload"]["mismatched_blocks"] > 0:
        w = "CONNECTOR-FAILS"                          # a mismatch decides even if a V check also fails
    elif cc["reload"]["checked_reloads"] == 0:
        w = "INVALID (zero checked reloads)"
    elif not all(c[k] for k in ("V1", "V2", "V3", "V4", "V5", "V6")):
        w = "INVALID (" + ", ".join(k for k in ("V1", "V2", "V3", "V4", "V5", "V6") if not c[k]) + ")"
    else:
        med = cc["coverage"]["median_c"]
        if med <= PROMPT_ONLY_C:                       # A1 B2: PROMPT-ONLY is tested first
            w = "CONNECTOR-PROMPT-ONLY"
        elif med >= WORKS_C:
            w = "CONNECTOR-WORKS"
        else:
            w = "CONNECTOR-PARTIAL"
    if rehearsal:
        out["verify"] = verify(arms)
    if not w.startswith("INVALID"):
        w = f"{w} (A3)"                                # R-112 F3: the word is made under the A3 V-RELOAD rule
    out["word"] = f"REHEARSAL | {w}" if rehearsal else w
    return out


def main() -> None:
    res = Path(sys.argv[1])
    rehearsal = "--rehearsal" in sys.argv[2:]
    out = word(load(res), rehearsal)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    short = {k: v for k, v in out["checks"].items() if k != "V5_pairs"}
    print(f"E-69: {out['word']} | checks {short} | C-C {out.get('C-C')} | S-C {out.get('S-C')}"
          + (f" | verify {{all_pass: {out['verify']['all_pass']}}}" if rehearsal else ""))


if __name__ == "__main__":
    main()
