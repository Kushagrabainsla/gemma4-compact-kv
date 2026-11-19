"""E-69 Modal app (README Sections 4 and 9).

Stages, one per call, each after the user's approval:
  probe       CPU only: versions, /dev/shm, memory limits, the hook functions at the pin, the code hashes.
  fetch-reh   CPU only: the Gemma 4 12B download into the weights volume (A1 F7).
  reh         S5 rehearsal on one L40S (Gemma 4 12B, the REH plan and the controls).
  fetch-main  CPU only: the Gemma 4 31B download into the weights volume.
  main        S6 run on one A100 80GB (Gemma 4 31B, the MAIN plan).

Usage (Modal client 1.6.3, Apache-2.0, in an isolated uv environment):
  uv run --no-project --with modal==1.6.3 modal run experiments/gemma4_kv_offload/modal_e69.py --stage probe
Results go to experiments/gemma4_kv_offload/results/e69-<stage>-<UTC stamp>/ on the Mac; a copy stays in the Modal
volume msthesis-e69-results (read it with `modal volume get` if the call fails after the GPU ran).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e69_stage as stg  # noqa: E402

image = stg.build_image(modal).add_local_python_source("e69_stage")
app = modal.App(stg.APP_NAME, image=image)
res_vol = modal.Volume.from_name(stg.RESULTS_VOLUME, create_if_missing=True)
w_vol = modal.Volume.from_name(stg.WEIGHTS_VOLUME, create_if_missing=True)
LIMITS = dict(startup_timeout=stg.STARTUP_TIMEOUT, scaledown_window=stg.SCALEDOWN_S, retries=0)


def launcher_sha() -> dict:
    return {"modal_e69.py": stg.file_sha(__file__), "e69_stage.py": stg.file_sha(stg.__file__)}


@app.function(cpu=2.0, memory=8_192, timeout=600, volumes={stg.RESULTS: res_vol}, **LIMITS)
def probe(run_id: str) -> dict:
    ok_man, lines = stg.check_manifest(stg.CODE)
    p = subprocess.run([sys.executable, f"{stg.CODE}/gemma4_kv_offload/probe_e69.py"], capture_output=True, text=True)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        out = {"error": (p.stdout + p.stderr)[-1_000:]}
    out.update(manifest_ok=ok_man, manifest=lines, launcher_sha=launcher_sha(),
               versions_ok=out.get("versions", {}).get("vllm") == stg.VLLM)
    root = Path(stg.RESULTS) / run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe.json").write_text(json.dumps(out, indent=1) + "\n")
    res_vol.commit()
    return out


@app.function(cpu=4.0, memory=16_384, timeout=2_400, volumes={stg.WEIGHTS: w_vol}, **LIMITS)
def fetch(stage: str) -> dict:
    """A1 F7: the model download on a CPU container into the weights volume (not on the billed GPU)."""
    model, rev = stg.MODEL[stage]
    t0 = time.time()
    snap, log = stg.download(model, rev, 2_200)
    w_vol.commit()
    return {"model": model, "revision": rev, "snapshot": snap, "seconds": round(time.time() - t0, 1),
            "log": log[-1_500:]}


def _run(run_id: str, stage: str, arms: tuple | None = None) -> dict:
    out = stg.session(stg.CODE, stg.RESULTS, run_id, stage, launcher_sha=launcher_sha(), on_start=res_vol.commit,
                      on_progress=res_vol.commit, arms=arms)
    res_vol.commit()
    w_vol.commit()
    return out


@app.function(gpu=stg.GPU["reh"], cpu=stg.CPU_CORES, memory=stg.MEMORY_MIB["reh"], timeout=stg.TIMEOUT["reh"],
              volumes={stg.RESULTS: res_vol, stg.WEIGHTS: w_vol}, **LIMITS)
def run_reh(run_id: str, arms: tuple | None = None) -> dict:
    return _run(run_id, "reh", arms)


@app.function(gpu=stg.GPU["main"], cpu=stg.CPU_CORES, memory=stg.MEMORY_MIB["main"], timeout=stg.TIMEOUT["main"],
              volumes={stg.RESULTS: res_vol, stg.WEIGHTS: w_vol}, **LIMITS)
def run_main(run_id: str) -> dict:
    return _run(run_id, "main")


@app.local_entrypoint()
def main(stage: str):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"e69-{stage}-{stamp}"
    t0 = time.time()
    local_sha = launcher_sha()
    if stage == "probe":
        res = probe.remote(run_id)
        res["launcher_sha_match"] = res["launcher_sha"] == local_sha
        dest = HERE / "results" / run_id
        dest.mkdir(parents=True)
        (dest / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"E-69 probe: versions_ok {res['versions_ok']} manifest_ok {res['manifest_ok']} hooks_ok "
              f"{res.get('hooks_ok')} dev_shm {res.get('dev_shm')} cgroup {res.get('cgroup_memory')} "
              f"launcher_sha_match {res['launcher_sha_match']} | {time.time() - t0:.0f} s")
        return
    if stage in ("fetch-reh", "fetch-main"):
        res = fetch.remote(stage.split("-")[1])
        dest = HERE / "results" / run_id
        dest.mkdir(parents=True)
        (dest / "fetch.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"E-69 {stage}: snapshot {res['snapshot']} | {res['seconds']} s")
        return
    if stage not in ("reh", "reh-cc", "main"):
        raise SystemExit(f"unknown stage {stage}")
    base = "main" if stage == "main" else "reh"
    print(f"E-69 {stage}: {stg.GPU[base]}, limit startup {stg.STARTUP_TIMEOUT} s + run {stg.TIMEOUT[base]} s, "
          f"cap per attempt ${stg.cost_cap(base):.2f}", flush=True)
    if stage == "main":
        res = run_main.remote(run_id)
    else:
        res = run_reh.remote(run_id, ("C-C",) if stage == "reh-cc" else None)   # R-112 F2: reh-cc runs C-C only
    rec = res["record"]
    if rec.get("restart"):
        print(f"E-69 {stage}: STOPPED, a restart of {run_id} was refused; read the volume copy")
        return
    dest = stg.unpack(res["tar"], HERE / "results")
    rec["launcher_sha_match"] = rec.get("launcher_sha") == local_sha
    (dest / "local_record.json").write_text(json.dumps({**rec, "wall_s_local": time.time() - t0}, indent=1) + "\n")
    print(f"E-69 {stage}: ok {rec.get('ok')} | {rec.get('decision', '')[:800]} | {time.time() - t0:.0f} s | {dest}")
