"""E-73 (README_E73.md Sections 4 and 8): the session without `modal` (the Mac tests run it): the restart guard, the code
hash check, the GPU list, the 31B snapshot, the arms S then C (P and D as two processes on GPU 0 and GPU 1), the
analysis, and the packed results. Shared helpers come from `e71_stage`.

Usage: e73_stage.py --manifest   (writes e73_code_manifest.json)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import e71_stage as s71  # noqa: E402

APP_NAME = "msthesis-e73-secondpoint"
RESULTS_VOLUME, WEIGHTS_VOLUME = s71.RESULTS_VOLUME, s71.WEIGHTS_VOLUME          # the registered E-71 volumes
IMAGE, CODE, RESULTS, WEIGHTS = s71.IMAGE, s71.CODE, s71.RESULTS, s71.WEIGHTS
FILES = {
    "gemma4_kv_decode_node": ("run_e73.py", "stub_e73.py", "analyze_e73.py", "e73_stage.py", "run_e72.py", "run_e71.py",
                              "stub_e71.py", "analyze_e71.py", "e71_stage.py", "prediction_interval_e73.json",
                              "prediction_interval.json", "prediction_frozen.json"),
    "gemma4_kv_disagg": ("nixl_compact.py", "run_e70.py"),
    "gemma4_kv_offload": ("e69_hooks.py", "run_e69.py"),
    "gemma4_kv_engine": ("compact_engine.py", "compact_layout.py"),
    "gemma4_kv_model": ("run_e55.py", "analyze_e55.py", "splice_e55.py"),
    "gemma4_kv_capacity_model": ("poolsim.py", "sweep.py"),
}
MANIFEST_LOCAL = HERE / "e73_code_manifest.json"
MANIFEST_REMOTE = f"{CODE}/gemma4_kv_decode_node/e73_code_manifest.json"
GPU, GPU_COUNT, CPU_CORES, MEMORY_MIB = "A100-80GB", 2, 8.0, 160 * 1024
TIMEOUT, STARTUP_TIMEOUT, SCALEDOWN_S, RESERVE_S = 5_000, 300, 2, 90       # A2: cap below the credits left
ARM_ORDER = ("S", "C")
ARM_LIMIT = {"S": 2_350, "C": 2_350}             # A2: the build plus the 2,050 s D watchdog


def code_files() -> list[tuple[Path, str]]:
    return [(EXP / f / n, f"{f}/{n}") for f, names in FILES.items() for n in names]


def mounts() -> list[tuple[Path, str]]:
    return [(local, f"{CODE}/{rel}") for local, rel in code_files()] + [(MANIFEST_LOCAL, MANIFEST_REMOTE)]


def make_manifest() -> dict:
    return {rel: s71.file_sha(local) for local, rel in code_files()}


def check_manifest(code: str) -> tuple[bool, list[str]]:
    man = json.loads((Path(code) / "gemma4_kv_decode_node" / "e73_code_manifest.json").read_text())
    lines, good = [], True
    for rel, want in sorted(man.items()):
        p = Path(code) / rel
        got = s71.file_sha(p) if p.exists() else "missing"
        good = good and got == want
        lines.append(("ok  " if got == want else "BAD ") + rel + " " + got)
    return good and bool(man), lines


def build_image(modal):
    img = modal.Image.from_registry(IMAGE).entrypoint([]).run_commands(s71.PY_LINK)
    for local, remote in mounts():
        img = img.add_local_file(str(local), remote)
    return img


def cost_cap() -> float:
    per_s = GPU_COUNT * s71.PRICE[GPU] + CPU_CORES * s71.PRICE["core"] + MEMORY_MIB / 1024 * s71.PRICE["gib"]
    return (STARTUP_TIMEOUT + TIMEOUT + SCALEDOWN_S) * per_s


def built(root: Path, arm: str) -> bool:
    """Both engines of the arm finished their build (the records hold `engine_seconds`)."""
    for role in ("p", "d"):
        try:
            if json.loads((root / arm / f"{role}_run.json").read_text()).get("engine_seconds") is None:
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


def run_arm(code: str, root: Path, arm: str, snap: str, stub: bool, limit: int) -> dict:
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    procs, t0 = {}, time.time()
    for role, gpu in (("p", "0"), ("d", "1")):
        cmd = [sys.executable, f"{code}/gemma4_kv_decode_node/run_e73.py", "--role", role, "--arm", arm, "--model",
               snap, "--dir", str(d)] + (["--stub"] if stub else [])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "E71_MANIFEST_OK": "1", "HF_HOME": WEIGHTS}
        procs[role] = subprocess.Popen(cmd, env=env, stdout=open(d / f"{role}_stdout.txt", "w"),
                                       stderr=subprocess.STDOUT)
    res = {}
    for role, cap in (("d", None), ("p", 30)):
        left = max(30, limit - (time.time() - t0))
        try:
            res[role] = procs[role].wait(timeout=left if cap is None else min(left, cap))
        except subprocess.TimeoutExpired:
            procs[role].kill()
            res[role] = "timeout" if cap is None else "stopped after D"
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = {role: s71.role_status(root, arm, role) for role in ("p", "d")}
    return res


def session(code: str, results: str, run_id: str, stub: bool = False, launcher_sha: dict | None = None,
            on_start=None, on_progress=None) -> dict:
    t0 = time.time()
    deadline = t0 + TIMEOUT - 60
    root = Path(results) / run_id
    if root.exists() and any(root.iterdir()):
        return {"record": {"run_id": run_id, "ok": False, "restart": True}, "tar": b""}
    root.mkdir(parents=True, exist_ok=True)
    rec: dict = {"run_id": run_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "env": {v: os.environ.get(v, "") for v in s71.MODAL_VARS}, "launcher_sha": launcher_sha or {},
                 "arms": {}}
    (root / "started.txt").write_text(rec["started_at"] + "\n")
    if on_start:
        on_start()

    def save():
        (root / "modal_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        if on_progress:
            on_progress()

    good, lines = check_manifest(code)
    (root / "manifest_check.txt").write_text("\n".join(lines) + "\n")
    rec["manifest_ok"] = good
    if good:
        if not stub:
            smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            (root / "nvidia-smi.txt").write_text(smi.stdout + smi.stderr)
            rec["gpus"] = smi.stdout.strip().splitlines()
        snap, log = s71.snapshot("main", stub)
        (root / "snapshot.txt").write_text(log)
        rec["snapshot"] = snap
        save()
        for arm in ARM_ORDER if snap else ():
            left = deadline - RESERVE_S - time.time()
            if left < 300:
                rec["arms"][arm] = {"skipped": round(left)}
                continue
            rec["arms"][arm] = run_arm(code, root, arm, snap, stub, min(ARM_LIMIT[arm], int(left)))
            save()
            if arm == "S" and rec["arms"][arm]["status"] != {"p": "ok", "d": "ok"}:     # C needs S
                rec["stop"] = "S failed; C is not run"
                break
        an = subprocess.run([sys.executable, f"{code}/gemma4_kv_decode_node/analyze_e73.py", str(root),
                             f"{code}/gemma4_kv_decode_node/prediction_interval_e73.json"],
                            capture_output=True, text=True)
        rec["decision"] = (an.stdout + an.stderr).strip()[-2_000:]
    rec["ok"] = good and (root / "summary.json").exists()
    rec["seconds"] = round(time.time() - t0, 1)
    save()
    return {"record": rec, "tar": s71.tar_bytes(root)}


if __name__ == "__main__" and "--manifest" in sys.argv:
    MANIFEST_LOCAL.write_text(json.dumps(make_manifest(), indent=1, sort_keys=True) + "\n")
    print(f"{len(make_manifest())} files -> {MANIFEST_LOCAL}")
