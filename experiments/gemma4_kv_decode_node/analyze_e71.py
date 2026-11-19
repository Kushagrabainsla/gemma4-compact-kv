"""E-71 (README Section 7 as amended by A1 after R-118): the frozen word from `<res>/{C0,S1,C,S2}/{p_run,d_run}.json`
and `prediction_interval.json`.

Primary (time-free): R_step = (decode tokens per work step of C) / (the mean of S1 and S2), against the frozen r_step
interval, widened on each side by half the A/A spread of the step statistic. Secondary (a report, no word): the
throughput ratio R against the Q_graph and Q_eager intervals, and the 1.25 line.

Usage: analyze_e71.py <results dir> <prediction_interval.json> [--rehearsal]   -> <results dir>/summary.json, one line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("S1", "C", "S2")
SMOKE = "C0"
GATE, AA_MAX, SAT_MIN, HALF_MIN = 1.25, 0.03, 0.95, 0.01


def load(res: Path) -> dict:
    out = {}
    for arm in (SMOKE, *ARMS):
        out[arm] = {}
        for role in ("p", "d"):
            f = res / arm / f"{role}_run.json"
            out[arm][role] = json.loads(f.read_text()) if f.exists() else None
    return out


def ok(r) -> bool:
    return bool(r) and r.get("status") == "ok"


def win(arms: dict, a: str) -> dict:
    return ((arms.get(a) or {}).get("d") or {}).get("window") or {}


def place(x: float, iv) -> str:
    lo, hi = iv
    return "below" if x < lo else "above" if x > hi else "inside"


def word(arms: dict, pred: dict, rehearsal: bool = False, window: int = 20_000) -> dict:
    out: dict = {"arms": {}, "prediction": {k: pred[k] for k in ("r_step_interval", "Q_graph_interval",
                                                                  "Q_eager_interval")}}
    for a, roles in arms.items():
        d, p = roles.get("d") or {}, roles.get("p") or {}
        out["arms"][a] = {"p": p.get("status", "not run"), "d": d.get("status", "not run"), "window": d.get("window"),
                          "produced": p.get("produced"), "error": (d.get("error") or p.get("error") or "")[:300]}
    roles = [r for pr in arms.values() for r in pr.values() if r]
    want = ("A100", "80GB")                    # the rehearsal also runs on A100 80GB (D-125, 2026-11-16)
    done = all(ok((arms.get(a) or {}).get(r)) for a in (SMOKE, *ARMS) for r in ("p", "d"))
    tps = {a: win(arms, a).get("decode_tok_s") for a in ARMS}
    tst = {a: win(arms, a).get("tokens_per_step") for a in ARMS}
    pool_want = None if rehearsal else ((arms["S1"].get("d") or {}).get("plan") or {}).get("pool_blocks")
    v = {"V1": bool(roles) and all(all(w in (r.get("gpu") or "") for w in want) for r in roles if r.get("gpu")),
         "V2": bool(roles) and all(r.get("manifest_ok") for r in roles),
         "V3": done and all((win(arms, a).get("saturated_share") or 0) >= SAT_MIN for a in ARMS),
         "V5": done and all(((r.get("spec") or {}).get("applied") and (r.get("kv") or {}).get("layout") == "LBHNC")
                            for a in (SMOKE, "C") for r in arms[a].values()),
         "V6": done and all(win(arms, a).get("work_steps") == window for a in ARMS),      # a sanity check (R-119 F3)
         "V7": done and all(win(arms, a).get("early_finishes") == 0 for a in (SMOKE, *ARMS))     # A3: C0 cannot
               and all((win(arms, a).get("completed") or 0) >= 1      # complete a 2K-token request in 350 steps
                       for a in ARMS),
         "V8": done and (pool_want is None or all(((arms[a]["d"].get("kv") or {}).get("pool_blocks")) == pool_want
                                                  for a in ARMS))}
    if done and all(tst.values()):
        s = (tst["S1"] + tst["S2"]) / 2
        out["R_step"] = tst["C"] / s
        out["aa_step"] = abs(tst["S1"] - tst["S2"]) / s
        v["V4"] = out["aa_step"] <= AA_MAX
    else:
        v["V4"] = False
    if done and all(tps.values()):
        s = (tps["S1"] + tps["S2"]) / 2
        out["R"], out["aa_tput"] = tps["C"] / s, abs(tps["S1"] - tps["S2"]) / s
        out["secondary"] = {"graph": place(out["R"], pred["Q_graph_interval"]),
                            "eager": place(out["R"], pred["Q_eager_interval"]),
                            "gate_1.25": out["R"] >= GATE, "aa_tput_ok": out["aa_tput"] <= AA_MAX}
    v = dict(sorted(v.items()))
    out["checks"] = v
    if not all(v.values()):
        w = "INVALID (" + ", ".join(k for k, x in v.items() if not x) + ")"
    else:
        lo, hi = pred["r_step_interval"]
        half = max(out["aa_step"] / 2, HALF_MIN)                         # R-119 F6: two runs can agree by chance
        out["r_step_band"] = [lo * (1 - half), hi * (1 + half)]
        w = {"below": "STEP-BELOW", "above": "STEP-ABOVE", "inside": "STEP-CONFIRMS"}[place(out["R_step"],
                                                                                           out["r_step_band"])]
    out["word"] = f"REHEARSAL | {w}" if rehearsal else w
    return out


def main() -> None:
    res, pred_path = Path(sys.argv[1]), Path(sys.argv[2])
    rehearsal = "--rehearsal" in sys.argv[3:]
    window = 1_500 if rehearsal else 20_000
    out = word(load(res), json.loads(pred_path.read_text()), rehearsal, window)
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"E-71: {out['word']} | R_step {out.get('R_step')} | A/A step {out.get('aa_step')} | R {out.get('R')} | "
          f"secondary {out.get('secondary')} | checks {out['checks']}")


if __name__ == "__main__":
    main()
