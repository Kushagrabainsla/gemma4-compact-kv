"""E-63 sweep (as amended by A1): the FIX and MIX grids, the checks and the rule of README.md Sections 4 to 7 and 13.

Usage: sweep.py <out json>   (run with `nice -n 10`; one core; pure Python; writes the JSON after each part)
"""
import itertools
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import poolsim as ps

FIX_POOLS = [round(2_000 * (40_000 / 2_000) ** (i / 24)) for i in range(25)]
FIX_LENS = [8_192, 16_384, 32_768, 65_536, 131_072]
MIX_POOLS = [round(4_000 * (32_000 / 4_000) ** (i / 8)) for i in range(9)]
DECIDE_POOLS = (6_000, 24_000)
DECIDE_LENS = {16_384, 32_768, 65_536, 131_072}
DECIDE_MIX = ("M1", "M2", "M3")
SEED, HORIZON, WARMUP, MIN_BOUND, MIN_CELLS = 63, 12_000, 2_000, 0.5, 5


def lognormal(rng: random.Random, median: float, sigma: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, round(math.exp(math.log(median) + sigma * rng.gauss(0, 1))))))


def draw(rng: random.Random, name: str) -> tuple[int, int]:
    if name == "M1":
        return lognormal(rng, 16_384, 1.0, 256, 131_072), lognormal(rng, 300, 0.8, 16, 4_096)
    if name == "M2":
        return lognormal(rng, 49_152, 0.6, 256, 131_072), lognormal(rng, 500, 0.8, 16, 4_096)
    if name == "M3":
        p = rng.randint(500, 4_096) if rng.random() < 0.7 else rng.randint(32_768, 131_072)
        return p, lognormal(rng, 300, 0.8, 16, 4_096)
    return lognormal(rng, 2_048, 0.8, 256, 131_072), lognormal(rng, 300, 0.8, 16, 4_096)   # M4 control


def gen_factory(name: str):
    def gen():
        rng = random.Random(f"{SEED}-{name}")
        return (draw(rng, name) for _ in itertools.count())
    return gen


def label(q):
    if q is None:
        return "NA"
    return "GO" if q >= 1.25 else ("KILL" if q < 1.10 else "MID")


def word_of(q_fix, q_mix) -> str:
    a, b = label(q_fix), label(q_mix)
    if a == b == "GO":
        return "GO-V2"
    if a == b == "KILL":
        return "KILL-V2"
    return f"PIVOT-V2({a},{b})"


def longest_prompt(blocks: int, layout: str, cap: int = 1_048_576) -> int:
    sim, lo, hi = ps.PoolSim(blocks, layout), 0, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sim.reservation(mid) <= blocks:
            lo = mid
        else:
            hi = mid - 1
    return lo


def fix_grid(**kw) -> list[dict]:
    rows = []
    for pool in FIX_POOLS:
        for L in FIX_LENS:
            row = {"pool": pool, "gib": pool * ps.BLOCK_BYTES / 2**30, "L": L}
            bmax = max(ps.closed_form_batch(pool, lay, L, **kw) for lay in ("stock", "compact"))
            out_len = max(256, 2 * (bmax + 1) * math.ceil(L / kw.get("budget", 2_048)))   # A1 F2
            row["out_len"] = out_len
            for lay in ("stock", "compact"):
                b = ps.closed_form_batch(pool, lay, L, **kw)
                # A2: a lone request must fit the empty pool with its full prompt and output (else it can only
                # preempt itself; this matters for the first-chunk reservation grid).
                full = ps.PoolSim(pool, lay, **{k: v for k, v in kw.items() if k != "full_isl"})
                if full.reservation(L + out_len) > pool - full.wm:
                    b = 0
                row[f"closed_{lay}"] = b
                row[f"steady_{lay}"] = ps.PoolSim(pool, lay).steady(L)
                if b >= 1:
                    out = ps.run_saturated(pool, lay, [(L, out_len)] * (bmax + 3), max_steps=500_000, **kw)
                    assert out["max_held"] <= pool, "V2 held"                               # A1 F4
                    row[f"b_{lay}"], row[f"pre_{lay}"] = out["max_decoding"], out["preemptions"]
                else:
                    row[f"b_{lay}"], row[f"pre_{lay}"] = 0, 0
            row["foot_ratio"] = row["steady_stock"] / row["steady_compact"]
            row["r_fix"] = row["b_compact"] / row["b_stock"] if row["b_stock"] >= 1 else None
            rows.append(row)
    return rows


def workload_foot(name: str, pool: int) -> float:
    rng, st_, co, n = random.Random(f"{SEED}-{name}"), 0, 0, 0
    probe, a, b = ps.PoolSim(pool, "stock"), ps.PoolSim(pool, "stock"), ps.PoolSim(pool, "compact")
    while n < 400:
        p, m = draw(rng, name)
        if probe.reservation(p + m) > pool:
            continue
        st_ += a.steady(p + m // 2)
        co += b.steady(p + m // 2)
        n += 1
    return st_ / co


def mix_grid(names=("M1", "M2", "M3", "M4"), **kw) -> list[dict]:
    rows = []
    for name in names:
        for pool in MIX_POOLS:
            row = {"workload": name, "pool": pool, "gib": pool * ps.BLOCK_BYTES / 2**30}
            for lay in ("stock", "compact"):
                out = ps.run_stream(pool, lay, gen_factory(name), HORIZON, WARMUP, **kw)
                for k in ("decode_per_step", "decode_per_second", "pool_bound_share", "preemptions",
                          "recomputed_tokens", "hol_stops", "skipped", "finished"):
                    row[f"{k}_{lay}"] = out[k]
            row["foot_ratio"] = workload_foot(name, pool)
            if row["decode_per_step_stock"] <= 0:
                raise RuntimeError(f"no stock decoding in the window: {name} pool {pool} {kw}")
            row["r_mix"] = row["decode_per_step_compact"] / row["decode_per_step_stock"]
            row["r_mix_time"] = row["decode_per_second_compact"] / row["decode_per_second_stock"]
            row["realized_share"] = (row["r_mix"] - 1) / (row["foot_ratio"] - 1)
            row["included"] = row["pool_bound_share_stock"] >= MIN_BOUND
            rows.append(row)
    return rows


def q_values(fix: list[dict], mix: list[dict]) -> dict:
    lo, hi = DECIDE_POOLS
    f = [r["r_fix"] for r in fix if lo <= r["pool"] <= hi and r["L"] in DECIDE_LENS and r["r_fix"]]
    cells = [r for r in mix if lo <= r["pool"] <= hi and r["workload"] in DECIDE_MIX]
    inc = [r for r in cells if r["included"]]
    q_mix = st.median(r["r_mix"] for r in inc) if len(inc) >= MIN_CELLS else None
    q_mix_t = st.median(r["r_mix_time"] for r in inc) if len(inc) >= MIN_CELLS else None
    return {"Q_fix": st.median(f), "Q_mix": q_mix, "Q_mix_time": q_mix_t, "mix_cells": len(cells),
            "mix_included": len(inc),
            "realized_share_median": st.median(r["realized_share"] for r in inc) if inc else None}


def checks() -> dict:
    v = {}
    v["V1_batches"] = (ps.closed_form_batch(6_143, "stock", 32_768), ps.closed_form_batch(6_143, "compact", 32_768),
                       ps.closed_form_batch(6_143, "stock", 65_536),
                       ps.closed_form_batch(6_143, "compact", 65_536)) == (3, 4, 2, 2)
    s, c = ps.PoolSim(6_143, "stock"), ps.PoolSim(6_143, "compact")
    v["V1_footprints"] = (s.steady(32_768), c.steady(32_768), s.steady(65_536)) == (1_349, 1_008, 2_373) \
        and abs(c.steady(65_536) - 1_690) <= 1
    peaks = {("stock", 32_768): 2_602, ("compact", 32_768): 2_262, ("stock", 65_536): 3_626, ("compact", 65_536): 2_944}
    v["V1_peaks"] = all(abs(ps.run_saturated(6_143, lay, [(L - 64, 64)])["peak_single"] - pk) <= 6
                        for (lay, L), pk in peaks.items())
    a = ps.run_stream(8_000, "compact", gen_factory("M1"), 3_000, 500)
    b = ps.run_stream(8_000, "compact", gen_factory("M1"), 3_000, 500)
    v["V3"] = a == b
    v["V4"] = all(abs((ps.closed_form_batch(10**6, "compact", L) / ps.closed_form_batch(10**6, "stock", L))
                      / (ps.PoolSim(10**6, "stock").steady(L) / ps.PoolSim(10**6, "compact").steady(L)) - 1) < 0.02
                  for L in FIX_LENS)
    return v


def main() -> None:
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res: dict = {"rule": "README.md Section 7 as amended by A1 (Section 13)"}

    def save():
        res["seconds"] = time.time() - t0
        out.write_text(json.dumps(res, indent=1))

    res["checks"] = checks()
    save()
    res["fix"] = fix_grid()
    res["V5_sim_vs_closed"] = sum(1 for r in res["fix"] for lay in ("stock", "compact")
                                  if r[f"closed_{lay}"] >= 1 and r[f"b_{lay}"] < r[f"closed_{lay}"] - 1)
    save()
    res["mix"] = mix_grid()
    save()
    q = q_values(res["fix"], res["mix"])
    res.update(q)
    lo, hi = DECIDE_POOLS
    m4 = [r for r in res["mix"] if r["workload"] == "M4" and lo <= r["pool"] <= hi]
    res["checks"]["V6_M4_control"] = all(0.97 <= r["r_mix"] <= r["foot_ratio"] + 0.03 for r in m4)
    base_word = word_of(q["Q_fix"], q["Q_mix"])
    # Robustness (A1 F9): the label must not flip under sync scheduling (cap 193) or watermark 0.01.
    rob = {}
    for tag, kw in (("cap193", {"cap_batches": 1}), ("watermark", {"watermark": 0.01})):
        qq = q_values(fix_grid(**kw), mix_grid(names=DECIDE_MIX, **kw))
        rob[tag] = dict(qq, word=word_of(qq["Q_fix"], qq["Q_mix"]))
        save()
    res["robustness"] = rob
    # Sensitivity (reported, not deciding; same decide sets and the same statistics).
    sens = {}
    for tag, kw in (("budget8192", {"budget": 8_192}), ("first_chunk_reservation", {"full_isl": False})):
        qq = q_values(fix_grid(**kw), mix_grid(names=DECIDE_MIX, **kw))
        sens[tag] = dict(qq, word=word_of(qq["Q_fix"], qq["Q_mix"]))
        save()
    res["sensitivity"] = sens
    res["longest_prompt"] = [{"pool": p, "stock": longest_prompt(p, "stock"), "compact": longest_prompt(p, "compact")}
                             for p in MIX_POOLS]
    if not all(res["checks"].values()):
        word = "INVALID"
    elif q["Q_mix"] is None:
        word = "PIVOT-V2(too few pool-bound cells)"
    else:
        word = base_word
        if word_of(q["Q_fix"], q["Q_mix_time"]) != base_word:
            word = f"PIVOT-V2(step and time labels differ: {base_word} / {word_of(q['Q_fix'], q['Q_mix_time'])})"
        elif any(r["word"] != base_word for r in rob.values()):
            word = f"PIVOT-V2(label flips under robustness: {base_word})"
    res["word"] = word
    save()
    print(f"E-63: {word} | Q_fix {q['Q_fix']:.3f} | Q_mix {q['Q_mix']} | Q_mix_time {q['Q_mix_time']} | "
          f"included {q['mix_included']}/{q['mix_cells']} | checks {res['checks']} | V5 {res['V5_sim_vs_closed']} | "
          f"{res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
