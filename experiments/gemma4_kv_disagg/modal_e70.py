"""E-70 stage 1 Modal app (README Sections 4 and 9): one call, one container with two L40S GPUs.

Usage (Modal client 1.6.3, Apache-2.0, in an isolated uv environment):
  uv run --no-project --with modal==1.6.3 modal run experiments/gemma4_kv_disagg/modal_e70.py
Results go to experiments/gemma4_kv_disagg/results/e70-<UTC stamp>/ on the Mac; a copy stays in the Modal volume
msthesis-e70-results. The weights come from the E-69 volume msthesis-e69-weights (Gemma 4 12B).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e70_stage as stg  # noqa: E402

image = stg.build_image(modal).add_local_python_source("e70_stage")
app = modal.App(stg.APP_NAME, image=image)
res_vol = modal.Volume.from_name(stg.RESULTS_VOLUME, create_if_missing=True)
w_vol = modal.Volume.from_name(stg.WEIGHTS_VOLUME)


def launcher_sha() -> dict:
    return {"modal_e70.py": stg.file_sha(__file__), "e70_stage.py": stg.file_sha(stg.__file__)}


@app.function(gpu=f"{stg.GPU}:{stg.GPU_COUNT}", cpu=stg.CPU_CORES, memory=stg.MEMORY_MIB, timeout=stg.TIMEOUT,
              startup_timeout=stg.STARTUP_TIMEOUT, scaledown_window=stg.SCALEDOWN_S, retries=0,
              volumes={stg.RESULTS: res_vol, stg.WEIGHTS: w_vol})
def run(run_id: str, pairs: tuple | None = None) -> dict:
    out = stg.session(stg.CODE, stg.RESULTS, run_id, launcher_sha=launcher_sha(), on_start=res_vol.commit,
                      on_progress=res_vol.commit, pairs=pairs)
    res_vol.commit()
    return out


@app.local_entrypoint()
def main(stage: int = 1):
    run_id = ("e70-" if stage == 1 else "e70s2-") + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    t0 = time.time()
    print(f"E-70 stage {stage}: {stg.GPU_COUNT} x {stg.GPU}, limit startup {stg.STARTUP_TIMEOUT} s + run "
          f"{stg.TIMEOUT} s, cap per try ${stg.cost_cap():.2f}", flush=True)
    res = run.remote(run_id, stg.STAGE2_PAIRS if stage == 2 else None)
    rec = res["record"]
    if rec.get("restart"):
        print(f"E-70: STOPPED, a restart of {run_id} was refused; read the volume copy")
        return
    dest = stg.unpack(res["tar"], HERE / "results")
    rec["launcher_sha_match"] = rec.get("launcher_sha") == launcher_sha()
    (dest / "local_record.json").write_text(json.dumps({**rec, "wall_s_local": time.time() - t0}, indent=1) + "\n")
    print(f"E-70: ok {rec.get('ok')} | {rec.get('decision', '')[:900]} | {time.time() - t0:.0f} s | {dest}")
