"""E-71 sizing (exploratory): how the predicted D3H ratio depends on the window length and the seed, at pool 6,144
(warm-up 2,000 steps). Usage: e71_window_scan.py <out json>"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e71_predict as ep  # noqa: E402

WINDOWS = (10_000, 20_000, 30_000, 40_000)


def main() -> None:
    out = Path(sys.argv[1])
    ep.e64.draw = ep.draw
    fits, pf = ep.e66.load()
    fits["meas"] = json.loads(ep.FIT.read_text())["fit"]
    models = {t: ep.an.Model(f) for t, f in fits.items()}
    res: dict = {}
    t0 = time.time()
    for w in WINDOWS:
        res[str(w)] = {}
        for seed in range(5):
            row = ep.e67.run_cell("all200", "D3H", 6_144, seed, models, fits, pf, horizon=2_000 + w, warmup=2_000)
            res[str(w)][str(seed)] = round(row["r_meas_M_L20"], 4)
        out.write_text(json.dumps(res, indent=1) + "\n")
        vals = sorted(res[str(w)].values())
        print(f"window {w}: by seed {res[str(w)]} | median {vals[2]} range [{vals[0]}, {vals[-1]}] | "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
