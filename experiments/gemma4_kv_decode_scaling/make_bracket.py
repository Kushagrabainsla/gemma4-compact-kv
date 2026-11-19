"""E-65 A1 (R-101 B1): the free Mac bracket. Planted W-M tables with the compact / stock global read ratio fixed at a
value r, for part 2 (`e65_model.py`) before any GPU result. r = 1.0 is "no read saving"; r = 0.625 is the full byte
saving. The planted numbers are tied to measurements:
- G_S (5 global layers) = 0.02 ms + 2.56e-8 s x B x L: 2.538 ms at 3 x 32K (half the E-58 10-layer read, 5.076 ms).
- SL (5 sliding layers, engine window) = 0.05 ms + 0.17 ms x B: 0.56 ms at batch 3, so 50 layers give 5.6 ms (the
  E-59 profile attention 10.67 ms minus the global read, R-101 B2).
- H (5 layers) = 0.2 ms (E-58: 0.396 ms for 10 layers in a graph).
- T_OTHER = 50.733 - 2 x 2.538 - 10 x 0.56 = 40.06 ms (the E-59 profile value).

Usage: make_bracket.py <ratio> <out fit json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

B = (1, 2, 4, 8, 16, 32, 64)
L = (1_024, 4_096, 16_384, 32_768, 65_536, 131_072)


def gs(b: int, ln: int) -> float:
    return 2e-5 + 2.56e-8 * b * ln


def main() -> None:
    r = float(sys.argv[1])
    rows = lambda f: {str(ln): {"B": list(B), "s": [f(b, ln) for b in B]} for ln in L}  # noqa: E731
    fit = {"G_S32": rows(gs), "G_C48": rows(lambda b, ln: r * gs(b, ln)),
           "SL": rows(lambda b, ln: 5e-5 + 1.7e-4 * b), "H": {"B": list(B), "s": [2e-4] * len(B), "length": 1_024},
           "t_other_s": 0.050733 - 2 * gs(3, 32_768) - 10 * (5e-5 + 1.7e-4 * 3), "t_other_profile_s": 0.04006}
    Path(sys.argv[2]).write_text(json.dumps({"word": "BRACKET", "ratio": r, "failed": [], "fit": fit}, indent=1)
                                 + "\n")
    print(f"bracket r={r}: T_OTHER {1e3 * fit['t_other_s']:.2f} ms")


if __name__ == "__main__":
    main()
