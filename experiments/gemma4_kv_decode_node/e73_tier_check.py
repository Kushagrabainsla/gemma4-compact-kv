"""Check that the tier size reaches the pool model: CPU evictions and the tier capacity for D3Q and D3H seed 0 to 4 at
pool 6,143, window [2,000, 22,000), depth 3, with the 200 GB, the 64 GiB and a 10 GB tier (a positive control)."""
import json
import sys
from pathlib import Path

HERE = Path("experiments/gemma4_kv_decode_node")
sys.path.insert(0, str(HERE))
import e71_predict as ep  # noqa: E402

ep.e64.draw = ep.draw
ep.e67.MODES["all69"] = (False, float(64 * 2**30))
ep.e67.MODES["all10"] = (False, 10e9)
orig_run = ep.e67.ps.run_stream
ep.e67.ps.run_stream = lambda *a, **kw: orig_run(*a, keep_waiting=3, **kw)
fits, pf = ep.e66.load()
fits["meas"] = json.loads(ep.FIT.read_text())["fit"]
models = {t: ep.an.Model(f) for t, f in fits.items()}
print("block bytes", ep.e67.ps.BLOCK_BYTES, "cap blocks 64GiB", int(64 * 2**30 // ep.e67.ps.BLOCK_BYTES))
for name in ("D3Q", "D3H"):
    for mode in ("all200", "all69", "all10"):
        rs, ev = [], []
        for seed in range(5):
            row = ep.e67.run_cell(mode, name, 6_143, seed, models, fits, pf, horizon=22_000, warmup=2_000)
            rs.append(round(row["r_step"], 4))
            ev.append((row["cpu_evictions_stock"], row["cpu_evictions_compact"]))
        print(name, mode, "r_step", rs, "evictions (stock, compact)", ev, flush=True)
