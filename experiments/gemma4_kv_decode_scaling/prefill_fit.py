"""E-65: the measured prefill weight of the W-M time model, from the E-59 step traces (old data; README Section 4.4).

E-59 (one A100 SXM, vLLM 3966924d5, Gemma 4 31B BF16, eager prefill, 2,048 batched tokens) timed every engine step of
each row. In a row, the steps before any request decodes are pure prefill chunks of the first request: 2,048 tokens
at the start contexts 0, 2,048, 4,096, ... With async scheduling, step 0 computes no full chunk (79 ms) and step k
computes chunk k - 1. This fits, per layout, t(chunk) = a + b x (mid-chunk context / 1,024) seconds for a full
chunk, by least squares over the pure steps 1 to 14 (chunks 0 to 13, all full).
Stock: engines G1 and G4 (STOCK rows). Compact: engine G3 (COMPACT rows of batch 3).

Usage: prefill_fit.py <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

E59 = (Path(__file__).resolve().parent.parent / "gemma4_kv_graphs" / "results" / "e59-a100-20261105" / "e59")
BUDGET = 2_048
SOURCES = {"stock": (("G1", "STOCK"), ("G4", "STOCK")), "compact": (("G3", "COMPACT"),)}


def pure_chunks(row: dict) -> list[tuple[float, float]]:
    """(mid-chunk context in units of 1,024 tokens, seconds) for each full pure-prefill step of a row."""
    tr = row["trace"]
    out = []
    for k, (sec, dec) in enumerate(zip(tr["seconds"], tr["decoding"])):
        if dec != 0:
            break
        if k == 0:                        # the async scheduling lag: the first step computes no full chunk (79 ms)
            continue
        out.append((((k - 1) * BUDGET + BUDGET / 2) / 1_024, sec))   # step k computes chunk k - 1
    return out


def fit(points: list[tuple[float, float]]) -> dict:
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    a = my - b * mx
    res = [y - (a + b * x) for x, y in points]
    return {"a_s": a, "b_s_per_1024": b, "n": n, "max_abs_residual_s": max(abs(r) for r in res),
            "max_rel_residual": max(abs(r) / y for r, (_, y) in zip(res, points))}


def main() -> None:
    out: dict = {"source": str(E59.relative_to(E59.parents[4])), "budget": BUDGET}
    for layout, srcs in SOURCES.items():
        pts = []
        for eng, arm in srcs:
            rec = json.loads((E59 / eng / "run_e59.json").read_text())
            for row in rec["rows"]:
                if row["tag"].startswith(eng) and row["batch"] == 3:
                    pts.extend(pure_chunks(row))
        out[layout] = fit(pts)
    out["compact_over_stock_at_16k"] = ((out["compact"]["a_s"] + 16 * out["compact"]["b_s_per_1024"])
                                        / (out["stock"]["a_s"] + 16 * out["stock"]["b_s_per_1024"]))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
