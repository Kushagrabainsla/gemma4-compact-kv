"""E-70 stage 1 (README Sections 4 and 9): the session without `modal`, so the Mac tests run it.

`session`: the restart guard, the code hash check, the GPU list, the model snapshot from the weights volume, the pairs
in order (P and D as two processes on GPU 0 and GPU 1), the analysis, and the packed results.
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
APP_NAME = "msthesis-e70-nixl"
RESULTS_VOLUME, WEIGHTS_VOLUME = "msthesis-e70-results", "msthesis-e69-weights"
IMAGE = "vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3"
CODE, RESULTS, WEIGHTS = "/workspace/code", "/results", "/weights"
PY_LINK = "command -v python > /dev/null || ln -s \"$(command -v python3)\" /usr/local/bin/python"
FILES = {
    "gemma4_kv_disagg": ("run_e70.py", "nixl_compact.py", "stub_e70.py", "analyze_e70.py", "analyze_e70s2.py",
                         "e70_stage.py"),
    "gemma4_kv_offload": ("e69_hooks.py", "run_e69.py"),
    "gemma4_kv_engine": ("compact_engine.py", "compact_layout.py"),
    "gemma4_kv_model": ("run_e55.py", "analyze_e55.py", "splice_e55.py"),
    "gemma4_kv_tasks": ("data_e56.json",),
}
MANIFEST_LOCAL = HERE / "code_manifest.json"
MANIFEST_REMOTE = f"{CODE}/gemma4_kv_disagg/code_manifest.json"
MODEL = ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d")
GPU, GPU_COUNT, CPU_CORES, MEMORY_MIB = "L40S", 2, 8.0, 96 * 1024     # stage 2 A1 (R-116 F3): 96 GiB
TIMEOUT, STARTUP_TIMEOUT, SCALEDOWN_S = 1_500, 300, 2   # README Section 9: 25 minutes (the watchdog)
PAIR_ORDER = ("STOCK", "COMPACT-NOPATCH", "COMPACT")
STAGE2_PAIRS = ("STOCK2", "COMPACT2")                   # README_stage2.md
PAIR_LIMIT = {"STOCK": 600, "COMPACT-NOPATCH": 300, "COMPACT": 600, "STOCK2": 660, "COMPACT2": 660}
ROLE_WAIT = {"STOCK": 420, "COMPACT-NOPATCH": 120, "COMPACT": 420, "STOCK2": 640, "COMPACT2": 640}
RESERVE_S = 90
PRICE = {"L40S": 0.000542, "core": 0.0000131, "gib": 0.00000222}     # modal.com/pricing, read 2026-11-12, $/s
MODAL_VARS = ("MODAL_CLOUD_PROVIDER", "MODAL_REGION", "MODAL_TASK_ID", "MODAL_IMAGE_ID", "NVIDIA_VISIBLE_DEVICES",
              "CUDA_VERSION")


def code_files() -> list[tuple[Path, str]]:
    return [(EXP / f / n, f"{f}/{n}") for f, names in FILES.items() for n in names]


def mounts() -> list[tuple[Path, str]]:
    return [(local, f"{CODE}/{rel}") for local, rel in code_files()] + [(MANIFEST_LOCAL, MANIFEST_REMOTE)]


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest() -> dict:
    return {rel: file_sha(local) for local, rel in code_files()}


def check_manifest(code: str) -> tuple[bool, list[str]]:
    man = json.loads((Path(code) / "gemma4_kv_disagg" / "code_manifest.json").read_text())
    lines, ok = [], True
    for rel, want in sorted(man.items()):
        p = Path(code) / rel
        got = file_sha(p) if p.exists() else "missing"
        ok = ok and got == want
        lines.append(("ok  " if got == want else "BAD ") + rel + " " + got)
    return ok and bool(man), lines


def build_image(modal):
    img = modal.Image.from_registry(IMAGE).entrypoint([]).run_commands(PY_LINK)
    for local, remote in mounts():
        img = img.add_local_file(str(local), remote)
    return img


def cost_cap() -> float:
    per_s = GPU_COUNT * PRICE[GPU] + CPU_CORES * PRICE["core"] + MEMORY_MIB / 1024 * PRICE["gib"]
    return (STARTUP_TIMEOUT + TIMEOUT + SCALEDOWN_S) * per_s


def tar_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(root), arcname=root.name)
    return buf.getvalue()


def snapshot(stub: bool) -> tuple[str | None, str]:
    if stub:
        return "stub", "stub"
    py = ("import sys\nfrom huggingface_hub import snapshot_download\n"
          "print(snapshot_download(sys.argv[1], revision=sys.argv[2]), flush=True)\n")
    script = Path("/tmp/e70_snapshot.py")
    script.write_text(py)
    p = subprocess.run([sys.executable, str(script), *MODEL], env={**os.environ, "HF_HOME": WEIGHTS},
                       capture_output=True, text=True, timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return (lines[-1] if p.returncode == 0 and lines else None), (p.stdout + p.stderr)[-3_000:]


def role_status(root: Path, pair: str, role: str) -> str:
    """A1 (R-114 B3): the status that the role wrote, not its exit code (NIXL can crash at exit after a good run)."""
    f = root / pair / f"{role}_run.json"
    try:
        return json.loads(f.read_text()).get("status", "missing")
    except Exception:  # noqa: BLE001
        return "missing"


def run_pair(code: str, root: Path, pair: str, snap: str, stub: bool, limit: int, extra_env: dict | None = None) -> dict:
    d = root / pair
    if d.exists():
        k = 1
        while (root / f"{pair}.failed{k}").exists():
            k += 1
        d.rename(root / f"{pair}.failed{k}")
    d.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    procs = {}
    t0 = time.time()
    for role, gpu in (("p", "0"), ("d", "1")):
        cmd = [py, f"{code}/gemma4_kv_disagg/run_e70.py", "--role", role, "--pair", pair, "--model", snap,
               "--data", f"{code}/gemma4_kv_tasks/data_e56.json", "--dir", str(d), "--wait", str(ROLE_WAIT[pair])]
        if stub:
            cmd.append("--stub")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "E70_MANIFEST_OK": "1", "HF_HOME": WEIGHTS,
               **(extra_env or {})}
        procs[role] = subprocess.Popen(cmd, env=env, stdout=open(d / f"{role}_stdout.txt", "w"),
                                       stderr=subprocess.STDOUT)
    res = {}
    # A1 of stage 2 (R-116 note): wait for D first; P only serves D's reads, so stop it 20 s after D ends
    for role, cap in (("d", None), ("p", 20)):
        p = procs[role]
        left = max(30, limit - (time.time() - t0))
        try:
            res[role] = p.wait(timeout=left if cap is None else min(left, cap))
        except subprocess.TimeoutExpired:
            p.kill()
            res[role] = "timeout" if cap is None else "stopped after D"
    res["seconds"] = round(time.time() - t0, 1)
    res["status"] = {role: role_status(root, pair, role) for role in ("p", "d")}
    res["env"] = extra_env or {}
    return res


def session(code: str, results: str, run_id: str, stub: bool = False, launcher_sha: dict | None = None,
            on_start=None, on_progress=None, pairs: tuple | None = None) -> dict:
    t0 = time.time()
    deadline = t0 + TIMEOUT - 60
    root = Path(results) / run_id
    if root.exists() and any(root.iterdir()):
        return {"record": {"run_id": run_id, "ok": False, "restart": True}, "tar": b""}
    root.mkdir(parents=True, exist_ok=True)
    rec: dict = {"run_id": run_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "env": {v: os.environ.get(v, "") for v in MODAL_VARS}, "launcher_sha": launcher_sha or {}, "pairs": {}}
    (root / "started.txt").write_text(rec["started_at"] + "\n")
    if on_start:
        on_start()

    def save():
        (root / "modal_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        if on_progress:
            on_progress()

    ok_man, lines = check_manifest(code)
    (root / "manifest_check.txt").write_text("\n".join(lines) + "\n")
    rec["manifest_ok"] = ok_man
    if ok_man:
        if not stub:
            smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            (root / "nvidia-smi.txt").write_text(smi.stdout + smi.stderr)
            rec["gpus"] = smi.stdout.strip().splitlines()
        snap, log = snapshot(stub)
        (root / "snapshot.txt").write_text(log)
        rec["snapshot"] = snap
        save()
        extra: dict = {}
        order = pairs or PAIR_ORDER
        rec["stage"] = 2 if order == STAGE2_PAIRS else 1
        for pair in order if snap else ():
            left = deadline - RESERVE_S - time.time()
            if left < 120:
                rec["pairs"][pair] = {"skipped": round(left)}
                continue
            rec["pairs"][pair] = run_pair(code, root, pair, snap, stub, min(PAIR_LIMIT[pair], int(left)), extra)
            save()
            if pair == "STOCK" and rec["pairs"][pair]["status"]["d"] != "ok":
                # A1 (R-114 F3), as changed by A2: one retry of the stock pair without CUDA IPC (host shared memory;
                # gVisor has no TCP); the compact pairs then use the same transports
                left = deadline - RESERVE_S - time.time()
                if left > 240:
                    extra = {"UCX_TLS": "cuda_copy,sm,self"}
                    rec["pairs"]["STOCK-retry"] = run_pair(code, root, pair, snap, stub,
                                                           min(PAIR_LIMIT[pair], int(left)), extra)
                    save()
                if rec["pairs"].get("STOCK-retry", {}).get("status", {}).get("d") != "ok":
                    rec["stop"] = "the stock pair failed (transport or container); the compact pairs are not run"
                    break
            if pair == "STOCK2" and rec["pairs"][pair]["status"]["d"] != "ok":
                rec["stop"] = "the stock pair of stage 2 failed; COMPACT2 is not run"
                break
        script = "analyze_e70s2.py" if rec["stage"] == 2 else "analyze_e70.py"
        an = subprocess.run([sys.executable, f"{code}/gemma4_kv_disagg/{script}", str(root)],
                            capture_output=True, text=True)
        rec["decision"] = (an.stdout + an.stderr).strip()[-2_000:]
    rec["ok"] = ok_man and (root / "summary.json").exists()
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
