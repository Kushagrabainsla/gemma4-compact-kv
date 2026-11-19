"""E-71 Modal app (README Sections 4 and 8): one call per stage, one container with two GPUs.

Usage (Modal client 1.6.3, Apache-2.0, in an isolated uv environment):
  uv run --no-project --with modal==1.6.3 modal run experiments/gemma4_kv_decode_node/modal_e71.py --stage reh
  uv run --no-project --with modal==1.6.3 modal run experiments/gemma4_kv_decode_node/modal_e71.py --stage main
Results go to experiments/gemma4_kv_decode_node/results/e71-<stage>-<UTC stamp>/; a copy stays in the volume
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
import e71_stage as stg  # noqa: E402

image = stg.build_image(modal).add_local_python_source("e71_stage")
app = modal.App(stg.APP_NAME, image=image)
res_vol = modal.Volume.from_name(stg.RESULTS_VOLUME, create_if_missing=True)
w_vol = modal.Volume.from_name(stg.WEIGHTS_VOLUME)
COMMON = dict(cpu=stg.CPU_CORES, startup_timeout=stg.STARTUP_TIMEOUT, scaledown_window=stg.SCALEDOWN_S, retries=0,
              volumes={stg.RESULTS: res_vol, stg.WEIGHTS: w_vol})


def launcher_sha() -> dict:
    return {"modal_e71.py": stg.file_sha(__file__), "e71_stage.py": stg.file_sha(stg.__file__)}


def _run(run_id: str, stage: str) -> dict:
    out = stg.session(stg.CODE, stg.RESULTS, run_id, stage, launcher_sha=launcher_sha(), on_start=res_vol.commit,
                      on_progress=res_vol.commit)
    res_vol.commit()
    return out


@app.function(gpu=f"{stg.GPU['reh']}:{stg.GPU_COUNT}", memory=stg.MEMORY_MIB["reh"], timeout=stg.TIMEOUT["reh"],
              **COMMON)
def run_reh(run_id: str) -> dict:
    return _run(run_id, "reh")


@app.function(gpu=f"{stg.GPU['main']}:{stg.GPU_COUNT}", memory=stg.MEMORY_MIB["main"], timeout=stg.TIMEOUT["main"],
              **COMMON)
def run_main(run_id: str) -> dict:
    return _run(run_id, "main")


@app.local_entrypoint()
def main(stage: str):
    if stage not in ("reh", "main"):
        raise SystemExit(f"unknown stage {stage}")
    run_id = f"e71-{stage}-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    t0 = time.time()
    print(f"E-71 {stage}: {stg.GPU_COUNT} x {stg.GPU[stage]}, limit startup {stg.STARTUP_TIMEOUT} s + run "
          f"{stg.TIMEOUT[stage]} s, cap per try ${stg.cost_cap(stage):.2f}", flush=True)
    res = (run_reh if stage == "reh" else run_main).remote(run_id)
    rec = res["record"]
    if rec.get("restart"):
        print(f"E-71 {stage}: STOPPED, a restart of {run_id} was refused; read the volume copy")
        return
    dest = stg.unpack(res["tar"], HERE / "results")
    rec["launcher_sha_match"] = rec.get("launcher_sha") == launcher_sha()
    (dest / "local_record.json").write_text(json.dumps({**rec, "wall_s_local": time.time() - t0}, indent=1) + "\n")
    print(f"E-71 {stage}: ok {rec.get('ok')} | {rec.get('decision', '')[:900]} | {time.time() - t0:.0f} s | {dest}")
