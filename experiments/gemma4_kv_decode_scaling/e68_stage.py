"""E-68 (README_E68.md): the parts of the Modal launcher that do not import `modal`, so that the Mac tests can run them.

The GPU code is the frozen E-65 package (bench_e65.py, analyze_e65.py, session_e65.sh and the files they import),
unchanged. This file only lists the files, builds the container paths, runs the E-65 session in a container (or as a
stub on the Mac), and packs the results.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
APP_NAME = "msthesis-e68-readratio"
VOLUME = "msthesis-e68-results"
IMAGE = ("vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3")  # E-49 to E-59
VLLM = "0.33.0rc1.dev229+g3966924d5"
CODE = "/workspace/code"                     # the E-65 session paths
RESULTS = "/results"                         # the Modal volume mount
PY_LINK = "command -v python > /dev/null || ln -s \"$(command -v python3)\" /usr/local/bin/python"

# The files that run in the container (the E-65 upload list of run_pod_e65.sh), by folder.
FILES = {
    "gemma4_kv_decode_scaling": ("bench_e65.py", "analyze_e65.py", "session_e65.sh", "boot_end.py", "stop_pod.py",
                                 "versions.py", "check_manifest.py"),
    "gemma4_kv_microbench": ("bench_e58.py",),
    "gemma4_kv_engine": ("compact_engine.py", "compact_layout.py"),
    "gemma4_kv_model": ("splice_e55.py",),
    "gemma4_kv_capacity_model": ("e64.py", "poolsim.py", "sweep.py"),
    "runpod": ("hardware_record.sh",),
}
MANIFEST_LOCAL = HERE / "code_manifest_e68.json"
MANIFEST_REMOTE = f"{CODE}/gemma4_kv_decode_scaling/code_manifest.json"

# Stage settings (README_E68.md Section 9). POD_LIMIT is the E-65 session watchdog; TIMEOUT is the Modal hard limit
# of the function (Modal stops the container at it: the boot watchdog of RUNPOD.md Section 3).
GPU = {"reh": "A10", "main": "A100-80GB"}
POD_LIMIT = {"reh": 600, "main": 900}
TIMEOUT = {"reh": 900, "main": 1200}         # A1 (R-107 B2): 1,200 s, not 1,500 s
STARTUP_TIMEOUT = 300                        # A1 (R-107 B2): the startup is billed and has its own limit
SCALEDOWN_S = 2                              # A1 (R-107 B2): the billed idle time after the call (Modal minimum)
CPU_CORES, MEMORY_MIB = 4.0, 16_384
# Modal list prices (modal.com/pricing, read 2026-11-11), dollars per second.
PRICE = {"A10": 0.000306, "A100-80GB": 0.000694, "core": 0.0000131, "gib": 0.00000222}
# Only these container variables are recorded (the environment can hold a token).
MODAL_VARS = ("MODAL_CLOUD_PROVIDER", "MODAL_REGION", "MODAL_TASK_ID", "MODAL_IMAGE_ID", "MODAL_ENVIRONMENT",
              "NVIDIA_VISIBLE_DEVICES", "CUDA_VERSION")


def mounts() -> list[tuple[Path, str]]:
    """(local file, container path) for every file that runs, plus the manifest."""
    out = [(EXP / folder / name, f"{CODE}/{folder}/{name}") for folder, names in FILES.items() for name in names]
    out.append((MANIFEST_LOCAL, MANIFEST_REMOTE))
    return out


def build_image(modal):
    """The E-65 image by digest, with no ENTRYPOINT (Modal needs a compatible one) and the code files added."""
    img = modal.Image.from_registry(IMAGE).entrypoint([]).run_commands(PY_LINK)
    for local, remote in mounts():
        img = img.add_local_file(str(local), remote)
    return img


def cost_cap(stage: str) -> float:
    """The largest bill of one attempt: (startup limit + run limit + idle) x the GPU, CPU and memory prices."""
    per_s = PRICE[GPU[stage]] + CPU_CORES * PRICE["core"] + MEMORY_MIB / 1024 * PRICE["gib"]
    return (STARTUP_TIMEOUT + TIMEOUT[stage] + SCALEDOWN_S) * per_s


def worst_case(stage: str) -> float:
    """A1 (R-107 B2): a preemption can restart the input once on the same call, so two attempts."""
    return 2 * cost_cap(stage)


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tar_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(root), arcname=root.name)
    return buf.getvalue()


def sha_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def session(code: str, results: str, run_id: str, stage: str, stub: bool = False,
            launcher_sha: dict | None = None, on_start=None) -> dict:
    """Check the code hashes, then run the unchanged E-65 session. Returns the record and the packed results."""
    t0 = time.time()
    run_root = Path(results) / run_id
    if run_root.exists() and any(run_root.iterdir()):
        # A1 (R-107 B2): a preemption restarts the input on the same call; never run (or bill) a second attempt.
        return {"record": {"run_id": run_id, "stage": stage, "ok": False, "restart": True}, "tar": b""}
    run_root.mkdir(parents=True, exist_ok=True)
    rec: dict = {"run_id": run_id, "stage": stage, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "env": {v: os.environ.get(v, "") for v in MODAL_VARS}, "launcher_sha": launcher_sha or {}}
    (run_root / "started.txt").write_text(rec["started_at"] + "\n")
    if on_start:
        on_start()                                 # the Modal volume commit, so that a restart sees this folder
    man = Path(code) / "gemma4_kv_decode_scaling" / "code_manifest.json"
    chk = subprocess.run(["python3", f"{code}/gemma4_kv_decode_scaling/check_manifest.py", code, str(man)],
                         capture_output=True, text=True)
    (run_root / "manifest_check.txt").write_text(chk.stdout + chk.stderr)
    rec["manifest_ok"] = chk.returncode == 0
    if not rec["manifest_ok"]:                     # the frozen code is not the code that would run: stop at once
        rec["ok"], rec["seconds"] = False, time.time() - t0
        (run_root / "modal_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        return {"record": rec, "tar": tar_bytes(run_root)}
    env = {**os.environ, "CODE": code, "RES": str(run_root / "e65"), "REH": "1" if stage == "reh" else "0",
           "STUB": "1" if stub else "0", "POD_LIMIT": str(POD_LIMIT[stage]), "AUTO_STOP": "0",
           "BOOT_LOG": str(run_root / "no_boot_watchdog.log"), "E68_TIE_RULE": "ratio"}   # A3: the ratio tie
    try:
        p = subprocess.run(["sh", f"{code}/gemma4_kv_decode_scaling/session_e65.sh"], env=env, capture_output=True,
                           text=True, timeout=POD_LIMIT[stage] + 120)
        rec["session_exit"] = p.returncode
        (run_root / "session_out.txt").write_text(p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        rec["session_exit"] = "timeout"
        (run_root / "session_out.txt").write_text(str(e.stdout or "") + str(e.stderr or ""))
    dec = run_root / "e65" / "decision_gpu.txt"
    rec["decision"] = dec.read_text().strip().splitlines()[-1] if dec.exists() and dec.read_text().strip() else ""
    slog = run_root / "e65" / "session.log"
    rec["bench_line"] = next((ln for ln in (slog.read_text().splitlines() if slog.exists() else [])
                              if " bench: exit " in ln), "")
    # A1 (R-107 F4): ok only if the bench exited 0 and the GPU word is VALID or REHEARSAL.
    rec["ok"] = (rec["session_exit"] == 0 and " bench: exit 0 " in rec["bench_line"]
                 and (run_root / "e65" / "bench" / "fit.json").exists()
                 and rec["decision"].startswith(("E-65 GPU: VALID", "E-65 GPU: REHEARSAL")))
    rec["seconds"] = time.time() - t0
    (run_root / "modal_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    rec["sha256"] = sha_tree(run_root)
    return {"record": rec, "tar": tar_bytes(run_root)}


def unpack(blob: bytes, parent: Path) -> Path:
    """Write the packed run folder under parent. Refuse a member outside it, a link, or a run folder that exists."""
    parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        members = tf.getmembers()
        tops = {Path(m.name).parts[0] for m in members}
        if len(tops) != 1:
            raise ValueError(f"the results hold {len(tops)} top folders")
        top = parent / tops.pop()
        if top.exists():
            raise ValueError(f"{top} exists; results are never overwritten")
        for m in members:
            target = (parent / m.name).resolve()
            if not str(target).startswith(str(parent.resolve()) + os.sep) or not (m.isfile() or m.isdir()):
                raise ValueError(f"unsafe member in the results: {m.name}")
        tf.extractall(parent, filter="data")
    return top
