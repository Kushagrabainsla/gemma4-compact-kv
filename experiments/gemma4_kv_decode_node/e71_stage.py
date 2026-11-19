"""E-71 (README Sections 4 and 8): the session without `modal` (the Mac tests run it): the restart guard, the code hash
check, the GPU list, the model snapshot, the arms S1, C, S2 (P and D as two processes on GPU 0 and GPU 1), the
analysis against the frozen prediction, and the packed results.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
APP_NAME = "msthesis-e71-decodenode"
RESULTS_VOLUME, WEIGHTS_VOLUME = "msthesis-e71-results", "msthesis-e69-weights"
IMAGE = "vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3"
CODE, RESULTS, WEIGHTS = "/workspace/code", "/results", "/weights"
PY_LINK = "command -v python > /dev/null || ln -s \"$(command -v python3)\" /usr/local/bin/python"
FILES = {
    "gemma4_kv_decode_node": ("run_e71.py", "stub_e71.py", "analyze_e71.py", "e71_stage.py", "prediction_frozen.json",
                              "prediction_interval.json"),
    "gemma4_kv_disagg": ("nixl_compact.py", "run_e70.py"),
    "gemma4_kv_offload": ("e69_hooks.py", "run_e69.py"),
    "gemma4_kv_engine": ("compact_engine.py", "compact_layout.py"),
    "gemma4_kv_model": ("run_e55.py", "analyze_e55.py", "splice_e55.py"),
    "gemma4_kv_capacity_model": ("poolsim.py", "sweep.py"),
}
MANIFEST_LOCAL = HERE / "code_manifest.json"
MANIFEST_REMOTE = f"{CODE}/gemma4_kv_decode_node/code_manifest.json"
MODEL = {"main": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
         "reh": ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d")}
GPU = {"main": "A100-80GB", "reh": "A100-80GB"}     # D-125 (2026-11-16): no L40S capacity; the rehearsal moved
GPU_COUNT, CPU_CORES = 2, 8.0
MEMORY_MIB = {"main": 160 * 1024, "reh": 96 * 1024}
TIMEOUT = {"main": 7_800, "reh": 1_500}          # README A3: 130 and 25 minutes (the container watchdog)
STARTUP_TIMEOUT, SCALEDOWN_S = 300, 2
ARM_ORDER = ("C0", "S1", "C", "S2")               # A1 (R-118 F1): the short compact smoke arm C0 first
ARM_LIMIT = {"main": 2_650, "reh": 420}           # an arm's engine build plus its D watchdog fit in its limit
SMOKE_LIMIT = {"main": 900, "reh": 300}
ARM_STEPS = {"main": 22_000, "reh": 1_800}        # warm-up plus window work steps of S1, C and S2 (run_e71 plans)
ARM_WATCHDOG = {"main": 2_250, "reh": 300}        # the D watchdog of S1, C and S2 (run_e71 plans)
BUILD_S = {"main": 250, "reh": 60}                # A3: an engine build, for the speed gate (E-69: 45 to 204 s)
RESERVE_S = 90
PRICE = {"A100-80GB": 0.000694, "L40S": 0.000542, "core": 0.0000131, "gib": 0.00000222}
MODAL_VARS = ("MODAL_CLOUD_PROVIDER", "MODAL_REGION", "MODAL_TASK_ID", "MODAL_IMAGE_ID", "NVIDIA_VISIBLE_DEVICES")


def code_files() -> list[tuple[Path, str]]:
    return [(EXP / f / n, f"{f}/{n}") for f, names in FILES.items() for n in names]


def mounts() -> list[tuple[Path, str]]:
    return [(local, f"{CODE}/{rel}") for local, rel in code_files()] + [(MANIFEST_LOCAL, MANIFEST_REMOTE)]


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest() -> dict:
    return {rel: file_sha(local) for local, rel in code_files()}


def check_manifest(code: str) -> tuple[bool, list[str]]:
    man = json.loads((Path(code) / "gemma4_kv_decode_node" / "code_manifest.json").read_text())
    lines, good = [], True
    for rel, want in sorted(man.items()):
        p = Path(code) / rel
        got = file_sha(p) if p.exists() else "missing"
        good = good and got == want
        lines.append(("ok  " if got == want else "BAD ") + rel + " " + got)
    return good and bool(man), lines


def build_image(modal):
    img = modal.Image.from_registry(IMAGE).entrypoint([]).run_commands(PY_LINK)
    for local, remote in mounts():
        img = img.add_local_file(str(local), remote)
    return img


def cost_cap(stage: str) -> float:
    per_s = GPU_COUNT * PRICE[GPU[stage]] + CPU_CORES * PRICE["core"] + MEMORY_MIB[stage] / 1024 * PRICE["gib"]
    return (STARTUP_TIMEOUT + TIMEOUT[stage] + SCALEDOWN_S) * per_s


def tar_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(root), arcname=root.name, filter=lambda ti: None if "/q/" in ti.name else ti)
    return buf.getvalue()


def snapshot(stage: str, stub: bool) -> tuple[str | None, str]:
    if stub:
        return "stub", "stub"
    py = ("import sys\nfrom huggingface_hub import snapshot_download\n"
          "print(snapshot_download(sys.argv[1], revision=sys.argv[2]), flush=True)\n")
    script = Path("/tmp/e71_snapshot.py")
    script.write_text(py)
    p = subprocess.run([sys.executable, str(script), *MODEL[stage]], env={**os.environ, "HF_HOME": WEIGHTS},
                       capture_output=True, text=True, timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return (lines[-1] if p.returncode == 0 and lines else None), (p.stdout + p.stderr)[-3_000:]


def role_status(root: Path, arm: str, role: str) -> str:
    try:
        return json.loads((root / arm / f"{role}_run.json").read_text()).get("status", "missing")
    except Exception:  # noqa: BLE001
        return "missing"


def smoke_ok(root: Path) -> bool:
    try:
        w = json.loads((root / "C0" / "d_run.json").read_text()).get("window") or {}
    except Exception:  # noqa: BLE001
        return False
    return w.get("early_finishes") == 0 and (w.get("decode_tokens") or 0) > 0


def speed_gate(root: Path, stage: str, left: float) -> tuple[bool, dict]:
    """A3 (after the rehearsal): the C0 window time per work step, projected to the three full arms. In the
    rehearsal the C0 step (ramp, NIXL loads in flight) was 19% slower than the steady S1 step, so the projection is
    conservative. Pass if one arm fits its D watchdog and the three arms fit the time left."""
    try:
        w = json.loads((root / "C0" / "d_run.json").read_text()).get("window") or {}
        step_s = w["seconds"] / w["work_steps"]
    except Exception:  # noqa: BLE001
        return False, {"error": "no C0 window"}
    arm_s = step_s * ARM_STEPS[stage]
    info = {"c0_step_ms": round(1e3 * step_s, 2), "projected_arm_s": round(arm_s), "left_s": round(left),
            "watchdog_s": ARM_WATCHDOG[stage]}
    return arm_s <= ARM_WATCHDOG[stage] - 100 and 3 * (arm_s + BUILD_S[stage]) <= left, info


def run_arm(code: str, root: Path, arm: str, snap: str, stage: str, stub: bool, limit: int) -> dict:
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    procs, t0 = {}, time.time()
    for role, gpu in (("p", "0"), ("d", "1")):
        cmd = [sys.executable, f"{code}/gemma4_kv_decode_node/run_e71.py", "--role", role, "--arm", arm, "--model",
               snap, "--dir", str(d)] + (["--rehearsal"] if stage == "reh" else []) + (["--stub"] if stub else [])
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
    res["status"] = {role: role_status(root, arm, role) for role in ("p", "d")}
    return res


def session(code: str, results: str, run_id: str, stage: str, stub: bool = False, launcher_sha: dict | None = None,
            on_start=None, on_progress=None) -> dict:
    t0 = time.time()
    deadline = t0 + TIMEOUT[stage] - 60
    root = Path(results) / run_id
    if root.exists() and any(root.iterdir()):
        return {"record": {"run_id": run_id, "ok": False, "restart": True}, "tar": b""}
    root.mkdir(parents=True, exist_ok=True)
    rec: dict = {"run_id": run_id, "stage": stage, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "env": {v: os.environ.get(v, "") for v in MODAL_VARS}, "launcher_sha": launcher_sha or {}, "arms": {}}
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
        snap, log = snapshot(stage, stub)
        (root / "snapshot.txt").write_text(log)
        rec["snapshot"] = snap
        save()
        for arm in ARM_ORDER if snap else ():
            left = deadline - RESERVE_S - time.time()
            if left < 300:
                rec["arms"][arm] = {"skipped": round(left)}
                continue
            cap = SMOKE_LIMIT[stage] if arm == "C0" else ARM_LIMIT[stage]
            rec["arms"][arm] = run_arm(code, root, arm, snap, stage, stub, min(cap, int(left)))
            save()
            if rec["arms"][arm]["status"]["d"] != "ok":
                rec["stop"] = f"{arm} failed; the next arms are not run"
                break
            if arm == "C0" and not smoke_ok(root):                   # R-119 B1: the smoke gate reads the window
                rec["stop"] = "C0 smoke gate failed (early finishes or no decode tokens); the next arms are not run"
                break
            if arm == "C0":                                            # A3: the speed gate
                good_speed, rec["speed_gate"] = speed_gate(root, stage, deadline - RESERVE_S - time.time())
                save()
                if not good_speed:
                    rec["stop"] = f"C0 speed gate failed {rec['speed_gate']}; the next arms are not run"
                    break
        args = [sys.executable, f"{code}/gemma4_kv_decode_node/analyze_e71.py", str(root),
                f"{code}/gemma4_kv_decode_node/prediction_interval.json"] + (["--rehearsal"] if stage == "reh" else [])
        an = subprocess.run(args, capture_output=True, text=True)
        rec["decision"] = (an.stdout + an.stderr).strip()[-2_000:]
    rec["ok"] = good and (root / "summary.json").exists()
    rec["seconds"] = round(time.time() - t0, 1)
    save()
    return {"record": rec, "tar": tar_bytes(root)}


def unpack(blob: bytes, parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tops = {Path(m.name).parts[0] for m in tf.getmembers()}
        if len(tops) != 1:
            raise ValueError(f"the results hold {len(tops)} top folders")
        top = parent / tops.pop()
        if top.exists():
            raise ValueError(f"{top} exists; results are never overwritten")
        for m in tf.getmembers():
            target = (parent / m.name).resolve()
            if not str(target).startswith(str(parent.resolve()) + os.sep) or not (m.isfile() or m.isdir()):
                raise ValueError(f"unsafe member in the results: {m.name}")
        tf.extractall(parent, filter="data")
    return top
