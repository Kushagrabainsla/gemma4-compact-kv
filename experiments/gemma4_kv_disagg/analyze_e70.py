"""E-70 stage 1 (README Section 7): the frozen word from `<res>/<PAIR>/{p_run,d_run}.json`.

Usage: analyze_e70.py <results dir> [--n 8]   -> <results dir>/summary.json and one line on stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PAIRS = ("STOCK", "COMPACT-NOPATCH", "COMPACT")
GAP_TEXT = "cannot be partitioned into NIXL regions"
REF_TOKENS = 32


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


def byte_check(d: dict, n: int) -> dict:
    cs = d.get("checks") or []
    full = [c for c in cs if "checked" in c]
    return {"requests": len(cs), "compared": len(full), "missing": len(cs) - len(full),
            "checked": sum(c["checked"] for c in full), "mismatch": sum(c["mismatch"] for c in full),
            "global": sum(c["global"] for c in full), "sliding": sum(c["sliding"] for c in full),
            "per_request_ok": all(c["global"] > 0 and c["sliding"] > 0 for c in full) and len(full) == n,
            "failed_loads": sum(bool(c.get("failed_load")) for c in full),
            "complete": d.get("n_complete")}


def gold(p: dict, d: dict) -> dict:
    ref = p.get("ref") or []
    ans = d.get("answers") or []
    first_div = []
    for a, b in zip(ref, ans):
        k = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
        first_div.append(k)
    return {"p_gold": sum(a["gold_in_text"] for a in ref), "d_gold": sum(a["gold_in_text"] for a in ans),
            "first_divergence": first_div}


def word(pairs: dict, n: int = 8) -> dict:
    out: dict = {"pairs": {}}
    for pair, roles in pairs.items():
        p, d = roles["p"], roles["d"]
        e = {"p": (p or {}).get("status", "not run"), "d": (d or {}).get("status", "not run"),
             "p_error": (p or {}).get("error", "")[:300], "d_error": (d or {}).get("error", "")[:300]}
        if ok(p) and ok(d):
            e.update(byte_check(d, n), **gold(p, d))
        out["pairs"][pair] = e
    st, nop, cc = pairs["STOCK"], pairs["COMPACT-NOPATCH"], pairs["COMPACT"]
    roles = [r for pr in pairs.values() for r in pr.values() if r]
    v = {"V1": bool(roles) and all("L40S" in (r.get("gpu") or "") for r in roles if r.get("gpu")),
         "V2": bool(roles) and all(r.get("manifest_ok") for r in roles),
         "V3": ok(st["p"]) and ok(st["d"]) and out["pairs"]["STOCK"].get("per_request_ok", False)
         and out["pairs"]["STOCK"].get("mismatch", 1) == 0
         and out["pairs"]["STOCK"].get("failed_loads", 1) == 0                       # A1 (R-114 N9)
         and out["pairs"]["STOCK"].get("complete") == n,
         # A1 (R-114 F1): the roles that fail at start also record the spec and the layout setting
         "V4": all(((r.get("spec") or {}).get("applied")
                    and ((r.get("kv") or {}).get("layout") or r.get("layout_env")) == "LBHNC")
                   for pr in (nop, cc) for r in pr.values() if r and r.get("gpu"))}
    out["checks"] = v
    gap_txt = " ".join(((r or {}).get("error") or "") + ((r or {}).get("traceback") or "")    # A1 (R-114 N6)
                       for r in nop.values())
    out["gap"] = "GAP-CONFIRMED" if GAP_TEXT in gap_txt else (
        "GAP-NOT-REPRODUCED" if ok(nop["p"]) or ok(nop["d"]) else "GAP-UNCLEAR")
    c = out["pairs"]["COMPACT"]
    if not v["V3"]:
        w = "INVALID (ENV: the stock pair did not transfer byte for byte)" if v["V1"] and v["V2"] else \
            "INVALID (" + ", ".join(k for k in ("V1", "V2", "V3") if not v[k]) + ")"
    elif not (v["V1"] and v["V2"] and v["V4"]):
        w = "INVALID (" + ", ".join(k for k in ("V1", "V2", "V4") if not v[k]) + ")"
    elif not (ok(cc["p"]) and ok(cc["d"])):
        w = "PATCH-FAILS"
    elif c.get("mismatch", 1) > 0 or c.get("failed_loads", 0) > 0 or c.get("compared", 0) < n:
        w = "PATCH-FAILS"
    elif (c.get("global", 0) >= n and c.get("sliding", 0) >= n and c.get("complete") == n
          and c.get("per_request_ok")):                       # A1 (R-114 F2): every request has both kinds
        w = "PATCH-WORKS"
    else:
        w = "PATCH-FAILS"
    out["word"] = w
    return out


def main() -> None:
    res = Path(sys.argv[1])
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
    out = word(load(res), n)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    brief = {k: {kk: vv for kk, vv in v.items() if kk in ("p", "d", "checked", "mismatch", "global", "sliding",
                                                          "p_gold", "d_gold", "complete")}
             for k, v in out["pairs"].items()}
    print(f"E-70: {out['word']} | {out['gap']} | checks {out['checks']} | {brief}")


if __name__ == "__main__":
    main()
