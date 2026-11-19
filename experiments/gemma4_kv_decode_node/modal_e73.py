"""E-73 Modal app (README_E73.md Sections 4 and 8): one call, one container with two A100 80GB.

Usage (Modal client 1.6.3, Apache-2.0, in an isolated uv environment):
  uv run --no-project --with modal==1.6.3 modal run experiments/gemma4_kv_decode_node/modal_e73.py
Results go to experiments/gemma4_kv_decode_node/results/e73-<UTC stamp>/; a copy stays in the volume
msthesis-e71-results. The weights come from msthesis-e69-weights.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e73_stage as stg  # noqa: E402

image = stg.build_image(modal).add_local_python_source("e73_stage", "e71_stage")
app = modal.App(stg.APP_NAME, image=image)
res_vol = modal.Volume.from_name(stg.RESULTS_VOLUME)
w_vol = modal.Volume.from_name(stg.WEIGHTS_VOLUME)


def launcher_sha() -> dict:
    return {"modal_e73.py": stg.s71.file_sha(__file__), "e73_stage.py": stg.s71.file_sha(stg.__file__)}


@app.function(gpu=f"{stg.GPU}:{stg.GPU_COUNT}", memory=stg.MEMORY_MIB, timeout=stg.TIMEOUT, cpu=stg.CPU_CORES,
              startup_timeout=stg.STARTUP_TIMEOUT, scaledown_window=stg.SCALEDOWN_S, retries=0,
              volumes={stg.RESULTS: res_vol, stg.WEIGHTS: w_vol})
def run_main(run_id: str) -> dict:
    out = stg.session(stg.CODE, stg.RESULTS, run_id, launcher_sha=launcher_sha(), on_start=res_vol.commit,
                      on_progress=res_vol.commit)
    res_vol.commit()
    return out


@app.local_entrypoint()
def main():
    run_id = "e73-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    t0 = time.time()
    print(f"E-73: {stg.GPU_COUNT} x {stg.GPU}, limit startup {stg.STARTUP_TIMEOUT} s + run {stg.TIMEOUT} s, "
          f"cap ${stg.cost_cap():.2f}", flush=True)
    res = run_main.remote(run_id)
    rec = res["record"]
    if rec.get("restart"):
        print(f"E-73: STOPPED, a restart of {run_id} was refused; read the volume copy")
        return
    dest = stg.s71.unpack(res["tar"], HERE / "results")
    rec["launcher_sha_match"] = rec.get("launcher_sha") == launcher_sha()
    (dest / "local_record.json").write_text(json.dumps({**rec, "wall_s_local": time.time() - t0}, indent=1) + "\n")
    print(f"E-73: ok {rec.get('ok')} | {rec.get('decision', '')[:900]} | {time.time() - t0:.0f} s | {dest}")
