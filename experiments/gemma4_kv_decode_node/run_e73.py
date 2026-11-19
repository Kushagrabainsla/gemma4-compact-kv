"""E-73 (README_E73.md): one engine role (P or D) of one arm, S (stock, hooks OFF) or C (compact, VIEWS). It runs the
E-71 driver (`run_e71.main`) unchanged, with three routes: the D3Q request sequence (`requests`), the NIXL lease of
E-72 (`engine_kwargs`), and the arm names S and C. `--stub` uses `stub_e73` (the E-71 stub plus the P lease line).

Usage: run_e73.py --role p|d --arm S|C --model <dir> --dir <arm dir> [--stub]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "gemma4_kv_disagg"), str(EXP / "gemma4_kv_offload"), str(EXP / "gemma4_kv_engine"),
                str(EXP / "gemma4_kv_model"), str(EXP / "gemma4_kv_capacity_model")]
import run_e71 as r71  # noqa: E402
import run_e72 as r72  # noqa: E402

ARMS = {"S": "stock", "C": "compact"}
WORKLOAD, PROMPT_MEDIAN, OUTPUT_MEDIAN = "D3Q", 8_192, 2_048
WATCHDOG_S = 2_050                               # A2 (README_E73): 93 ms per step for 22,000 steps


def requests(n: int, max_len: int) -> list[tuple[int, int]]:
    """D3Q, seed 0, as the model sees it (`e71_predict.draw` for D3Q; the poolsim skip rule on the usable pool; the
    engine length limit). The same code as `run_e71.requests` with the D3Q seed string and prompt median."""
    import poolsim as ps
    import sweep
    rng = random.Random(f"64-{WORKLOAD}-0")
    probe = ps.PoolSim(r71.MODEL_POOL, "stock", offload=(False, 200e9))
    out: list[tuple[int, int]] = []
    while len(out) < n:
        p = sweep.lognormal(rng, PROMPT_MEDIAN, 0.6, 256, 131_072)
        m = sweep.lognormal(rng, OUTPUT_MEDIAN, 0.6, 16, 16_384)
        if p + m > max_len or probe.reservation(p + m) > r71.MODEL_POOL - probe.wm:
            continue
        out.append((p, m))
    return out


def recording_build(orig):
    """A1 (R-123 F1): the E-71 build, with the hook mode that it sets recorded in `rec["mode"]` (OFF for S, VIEWS for
    C), so W7 can check it."""
    def build(layout, role, model, plan, rec):
        import compact_engine as ce
        orig_set = ce.ViewsSplice.set_mode

        def set_mode(self, mode):
            rec["mode"] = mode
            return orig_set(self, mode)
        ce.ViewsSplice.set_mode = set_mode
        try:
            return orig(layout, role, model, plan, rec)
        finally:
            ce.ViewsSplice.set_mode = orig_set
    return build


def install() -> None:
    if not hasattr(r71, "engine_kwargs_e71"):
        r71.engine_kwargs_e71 = r71.engine_kwargs
        r71.build = recording_build(r71.build)
    r71.engine_kwargs = r72.engine_kwargs        # the E-72 lease on P and on D's NIXL child; the E-71 build is kept
    r71.requests = requests
    r71.MAIN = dict(r71.MAIN, watchdog=WATCHDOG_S)  # A2: the D watchdog fits the lower arm limit
    r71.ARMS.clear()
    r71.ARMS.update(ARMS)
    if "--stub" in sys.argv:
        import stub_e73
        sys.modules["stub_e71"] = stub_e73


if __name__ == "__main__":
    install()
    r71.main()
