"""E-68 Modal app (README_E68.md Sections 4 and 9, as amended by A1): the unchanged E-65 GPU session in a Modal function.

Stages, one per call, each after the user's approval:
  probe     CPU only: the image builds; python and pip are on PATH; the pinned vLLM, torch and Triton import; the code
            hashes match the manifest.
  timeout   CPU only: Modal stops a function at its timeout (the hard cost limit of every GPU stage).
  reh       S5 rehearsal on one A10 (REH=1: 8 points of the E-65 grid).
  main      S6 run on one A100 80GB (the full E-65 grid).

Usage (Modal client 1.6.3, Apache-2.0, in an isolated uv environment):
  uv run --no-project --with modal==1.6.3 modal run experiments/gemma4_kv_decode_scaling/modal_e68.py --stage probe
The results are written to experiments/gemma4_kv_decode_scaling/results/e68-<stage>-<UTC stamp>/ on the Mac, and a
copy stays in the Modal volume msthesis-e68-results. If the call fails after the GPU ran, read the copy with
`modal volume get msthesis-e68-results <run id>` (A1, R-107 F5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e68_stage as stg  # noqa: E402

# A1 (R-107 B1): Modal 1.0 does not add local modules by itself; the launcher helper is added as the last step.
image = stg.build_image(modal).add_local_python_source("e68_stage")
app = modal.App(stg.APP_NAME, image=image)
vol = modal.Volume.from_name(stg.VOLUME, create_if_missing=True)
LIMITS = dict(startup_timeout=stg.STARTUP_TIMEOUT, scaledown_window=stg.SCALEDOWN_S, retries=0)
WANT = {"vllm": stg.VLLM, "torch": "2.14.0+cu130", "triton": "3.8.0"}


def launcher_sha() -> dict:
    """A1 (R-107 B1): the SHA-256 of the two launcher files as they are in the container."""
    return {"modal_e68.py": stg.file_sha(__file__), "e68_stage.py": stg.file_sha(stg.__file__)}


@app.function(cpu=1.0, memory=4_096, timeout=600, volumes={stg.RESULTS: vol}, **LIMITS)
def probe(run_id: str) -> dict:
    import subprocess
    out = {}
    for name, cmd in (("python", ["python", "--version"]), ("pip", ["sh", "-c", "command -v pip && python -m pip --version"]),
                      ("versions", ["python3", f"{stg.CODE}/gemma4_kv_decode_scaling/versions.py"]),
                      ("manifest", ["python3", f"{stg.CODE}/gemma4_kv_decode_scaling/check_manifest.py", stg.CODE,
                                    stg.MANIFEST_REMOTE]),
                      ("tools", ["sh", "-c", "command -v timeout; command -v nvidia-smi; true"])):
        p = subprocess.run(cmd, capture_output=True, text=True)
        out[name] = {"exit": p.returncode, "out": (p.stdout + p.stderr)[-2_000:]}
    lines = dict(ln.split(" ", 1) for ln in out["versions"]["out"].splitlines() if " " in ln)
    out["versions_ok"] = all(lines.get(k, "").strip() == v for k, v in WANT.items())   # A1 (R-107 F2)
    out["python_ok"] = out["python"]["exit"] == 0 and out["pip"]["exit"] == 0
    out["manifest_ok"] = out["manifest"]["exit"] == 0
    out["launcher_sha"] = launcher_sha()
    root = Path(stg.RESULTS) / run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe.json").write_text(json.dumps(out, indent=1) + "\n")
    vol.commit()
    return out


@app.function(cpu=0.25, memory=256, timeout=60, **LIMITS)
def sleeper() -> str:
    time.sleep(120)
    return "not stopped"


def _bench(run_id: str, stage: str) -> dict:
    out = stg.session(stg.CODE, stg.RESULTS, run_id, stage, launcher_sha=launcher_sha(), on_start=vol.commit)
    vol.commit()
    return out


@app.function(gpu=stg.GPU["reh"], cpu=stg.CPU_CORES, memory=stg.MEMORY_MIB, timeout=stg.TIMEOUT["reh"],
              volumes={stg.RESULTS: vol}, **LIMITS)
def bench_reh(run_id: str) -> dict:
    return _bench(run_id, "reh")


@app.function(gpu=stg.GPU["main"], cpu=stg.CPU_CORES, memory=stg.MEMORY_MIB, timeout=stg.TIMEOUT["main"],
              volumes={stg.RESULTS: vol}, **LIMITS)
def bench_main(run_id: str) -> dict:
    return _bench(run_id, "main")


@app.local_entrypoint()
def main(stage: str):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"e68-{stage}-{stamp}"
    t0 = time.time()
    local_sha = {"modal_e68.py": stg.file_sha(__file__), "e68_stage.py": stg.file_sha(stg.__file__)}
    if stage == "probe":
        res = probe.remote(run_id)
        res["launcher_sha_match"] = res["launcher_sha"] == local_sha
        dest = HERE / "results" / run_id
        dest.mkdir(parents=True)
        (dest / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"E-68 probe: python_ok {res['python_ok']} versions_ok {res['versions_ok']} manifest_ok "
              f"{res['manifest_ok']} launcher_sha_match {res['launcher_sha_match']} | {time.time() - t0:.0f} s")
    elif stage == "timeout":
        try:
            msg = sleeper.remote()
        except Exception as e:  # noqa: BLE001 (Modal raises FunctionTimeoutError)
            msg = f"stopped: {type(e).__name__}"
        dest = HERE / "results" / run_id
        dest.mkdir(parents=True)
        (dest / "timeout.txt").write_text(f"{msg} after {time.time() - t0:.0f} s\n")
        print(f"E-68 timeout check: {msg} after {time.time() - t0:.0f} s")
    elif stage in ("reh", "main"):
        fn = bench_reh if stage == "reh" else bench_main
        print(f"E-68 {stage}: {stg.GPU[stage]}, limits startup {stg.STARTUP_TIMEOUT} s + run {stg.TIMEOUT[stage]} s, "
              f"cap per attempt ${stg.cost_cap(stage):.2f}, worst case ${stg.worst_case(stage):.2f}", flush=True)
        res = fn.remote(run_id)
        rec = res["record"]
        if rec.get("restart"):
            print(f"E-68 {stage}: STOPPED, a restart of {run_id} was refused (a preemption); read the volume copy")
            return
        print(f"E-68 {stage}: results {len(res['tar'])} bytes", flush=True)
        dest = stg.unpack(res["tar"], HERE / "results")
        rec["launcher_sha_match"] = rec.get("launcher_sha") == local_sha
        (dest / "local_record.json").write_text(json.dumps({**rec, "wall_s_local": time.time() - t0}, indent=1) + "\n")
        print(f"E-68 {stage}: ok {rec['ok']} | launcher_sha_match {rec['launcher_sha_match']} | "
              f"{rec.get('decision', '')[:600]} | {time.time() - t0:.0f} s | results {dest}")
    else:
        raise SystemExit(f"unknown stage {stage}")
