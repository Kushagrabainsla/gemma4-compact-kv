"""E-69 (README Sections 4 and 9): the parts of the Modal launcher that do not import `modal`, so the Mac tests run them.

`session` runs inside the container (or as a stub on the Mac): the restart guard, the code hash check, the probe, the
telemetry, the model download into the weights volume, the arms in the order of README Section 9 (one process each,
C-C tried twice), and the analysis. It returns the record and the packed results.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
APP_NAME = "msthesis-e69-offload"
RESULTS_VOLUME, WEIGHTS_VOLUME = "msthesis-e69-results", "msthesis-e69-weights"
IMAGE = "vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3"
VLLM = "0.33.0rc1.dev229+g3966924d5"
CODE, RESULTS, WEIGHTS = "/workspace/code", "/results", "/weights"
PY_LINK = "command -v python > /dev/null || ln -s \"$(command -v python3)\" /usr/local/bin/python"
FILES = {
    "gemma4_kv_offload": ("run_e69.py", "e69_hooks.py", "analyze_e69.py", "stub_e69.py", "probe_e69.py",
                          "e69_stage.py"),
    "gemma4_kv_engine": ("compact_engine.py", "compact_layout.py"),
    "gemma4_kv_model": ("run_e55.py", "analyze_e55.py", "splice_e55.py"),
}
MANIFEST_LOCAL = HERE / "code_manifest.json"
MANIFEST_REMOTE = f"{CODE}/gemma4_kv_offload/code_manifest.json"
MODEL = {"main": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
         "reh": ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d")}
GPU = {"reh": "L40S", "main": "A100-80GB"}
CPU_CORES = 8.0
MEMORY_MIB = {"reh": 128 * 1024, "main": 160 * 1024}   # A1 F6; A3: 160 GiB for the 96 GiB tier of the main run
TIMEOUT = {"reh": 2_700, "main": 4_500}         # A1 F8: 45 and 75 minutes (the Modal hard limit, the watchdog)
STARTUP_TIMEOUT = 300
SCALEDOWN_S = 2
ARM_ORDER = {"reh": ("S-C", "C-C", "C-C0", "C-CX", "C-C64"),   # A1 B2, B3, F6: the controls run in the rehearsal
             "main": ("C-C", "S-C", "S-R", "C-R")}             # A1 F4: C-C first, so a 31B failure shows at once
ARM_LIMIT = {"reh": 480, "main": 1_200}         # A1 F8: one arm (engine start, E2E, EQ set, W)
ARM_MIN = {"reh": 240, "main": 600}             # an arm starts only with this much time left (after the reserve)
RESERVE_S = 120                                 # the analysis and the packing
VERIFY_KEYS = ("KV cache layout", "mmap file", "MADV_POPULATE", "offload region", "cudaHostRegister", "pageable",
               "Traceback", "Error")
DOWNLOAD_LIMIT = {"reh": 900, "main": 1_500}
# Modal list prices (modal.com/pricing, read 2026-11-12), dollars per second.
PRICE = {"L40S": 0.000542, "A100-80GB": 0.000694, "core": 0.0000131, "gib": 0.00000222}
MODAL_VARS = ("MODAL_CLOUD_PROVIDER", "MODAL_REGION", "MODAL_TASK_ID", "MODAL_IMAGE_ID", "MODAL_ENVIRONMENT",
              "NVIDIA_VISIBLE_DEVICES", "CUDA_VERSION")


def code_files() -> list[tuple[Path, str]]:
    return [(EXP / f / n, f"{f}/{n}") for f, names in FILES.items() for n in names]


def mounts() -> list[tuple[Path, str]]:
    out = [(local, f"{CODE}/{rel}") for local, rel in code_files()]
    out.append((MANIFEST_LOCAL, MANIFEST_REMOTE))
    return out


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest() -> dict:
    return {rel: file_sha(local) for local, rel in code_files()}


def check_manifest(code: str) -> tuple[bool, list[str]]:
    man = json.loads((Path(code) / "gemma4_kv_offload" / "code_manifest.json").read_text())
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


def per_second(stage: str) -> float:
    return PRICE[GPU[stage]] + CPU_CORES * PRICE["core"] + MEMORY_MIB[stage] / 1024 * PRICE["gib"]


def cost_cap(stage: str) -> float:
    return (STARTUP_TIMEOUT + TIMEOUT[stage] + SCALEDOWN_S) * per_second(stage)


def tar_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(root), arcname=root.name, filter=lambda ti: None if ti.name.endswith(".mmap") else ti)
    return buf.getvalue()


def sha_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def telemetry(path: Path, stop: threading.Event) -> None:
    cmd = ["nvidia-smi", "--format=csv,nounits", "-lms", "1000", "--query-gpu=timestamp,power.draw,clocks.sm,"
           "temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used"]
    try:
        with open(path, "w") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            stop.wait()
            p.terminate()
    except FileNotFoundError:
        path.write_text("no nvidia-smi\n")


def download(model: str, rev: str, limit: int) -> tuple[str | None, str]:
    """The model snapshot in the weights volume (HF_HOME there; public model, no token)."""
    py = ("import sys\nfrom huggingface_hub import snapshot_download\n"
          "print(snapshot_download(sys.argv[1], revision=sys.argv[2]), flush=True)\n")
    script = Path("/tmp/e69_download.py")
    script.write_text(py)
    env = {**os.environ, "HF_HOME": WEIGHTS, "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")}
    try:
        p = subprocess.run([sys.executable, str(script), model, rev], env=env, capture_output=True, text=True,
                           timeout=limit)
    except subprocess.TimeoutExpired:
        return None, "download timeout"
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return (lines[-1] if p.returncode == 0 and lines else None), (p.stdout + p.stderr)[-3_000:]


def session(code: str, results: str, run_id: str, stage: str, stub: bool = False, launcher_sha: dict | None = None,
            on_start=None, on_progress=None, arms: tuple | None = None) -> dict:
    t0 = time.time()
    deadline = t0 + TIMEOUT[stage] - 60
    run_root = Path(results) / run_id
    if run_root.exists() and any(run_root.iterdir()):
        return {"record": {"run_id": run_id, "stage": stage, "ok": False, "restart": True}, "tar": b""}
    run_root.mkdir(parents=True, exist_ok=True)
    rec: dict = {"run_id": run_id, "stage": stage, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "env": {v: os.environ.get(v, "") for v in MODAL_VARS}, "launcher_sha": launcher_sha or {}, "arms": {}}
    (run_root / "started.txt").write_text(rec["started_at"] + "\n")
    if on_start:
        on_start()

    def save():
        (run_root / "modal_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        if on_progress:
            on_progress()

    ok_man, lines = check_manifest(code)
    (run_root / "manifest_check.txt").write_text("\n".join(lines) + "\n")
    rec["manifest_ok"] = ok_man
    if not ok_man:
        rec["ok"], rec["seconds"] = False, time.time() - t0
        save()
        return {"record": rec, "tar": tar_bytes(run_root)}
    py = sys.executable
    pr = subprocess.run([py, f"{code}/gemma4_kv_offload/probe_e69.py"], capture_output=True, text=True)
    (run_root / "probe.txt").write_text(pr.stdout + pr.stderr)
    try:
        rec["probe"] = json.loads(pr.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        rec["probe"] = {"error": (pr.stdout + pr.stderr)[-500:]}
    stop = threading.Event()
    tel = threading.Thread(target=telemetry, args=(run_root / "telemetry.csv", stop), daemon=True)
    if not stub:
        tel.start()
        smi = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
        (run_root / "nvidia-smi.txt").write_text(smi.stdout + smi.stderr)
    model, rev = MODEL[stage]
    if stub:
        snap, log = "stub", "stub"
    else:
        snap, log = download(model, rev, DOWNLOAD_LIMIT[stage])
    (run_root / "download.txt").write_text(log)
    rec["snapshot"], rec["download_s"] = snap, round(time.time() - t0, 1)
    save()
    if snap is None:
        rec["ok"] = False
        rec["stop"] = "model download failed"
    else:
        for arm in arms or ARM_ORDER[stage]:           # R-112 F2: a short check runs a subset (C-C only)
            left = deadline - RESERVE_S - time.time()
            if left < ARM_MIN[stage]:
                rec["arms"][arm] = {"status": "skipped", "left_s": round(left)}
                continue
            for k in (1, 2) if arm == "C-C" else (1,):
                out = run_root / arm
                if out.exists():
                    out.rename(run_root / f"{arm}.failed{k - 1}")
                cmd = [py, f"{code}/gemma4_kv_offload/run_e69.py", "--arm", arm, "--model", snap, "--out", str(out),
                       "--try", str(k)] + (["--rehearsal"] if stage == "reh" else []) + (["--stub"] if stub else [])
                env = {**os.environ, "E69_MANIFEST_OK": "1", "HF_HOME": WEIGHTS}
                limit = min(ARM_LIMIT[stage], max(60, int(deadline - RESERVE_S - time.time())))
                ta = time.time()
                try:
                    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=limit)
                    code_exit, txt = p.returncode, p.stdout + p.stderr
                except subprocess.TimeoutExpired as e:
                    code_exit, txt = "timeout", str(e.stdout or "") + str(e.stderr or "")
                out.mkdir(parents=True, exist_ok=True)
                (out / "stdout.txt").write_text(txt[-200_000:])
                rec["arms"][arm] = {"exit": code_exit, "seconds": round(time.time() - ta, 1), "try": k,
                                    "line": next((ln for ln in txt.splitlines()[::-1] if ln.startswith("E-69 ")), ""),
                                    "verify_lines": [ln[-300:] for ln in txt.splitlines()          # A2: VERIFY 1, 2
                                                     if any(key in ln for key in VERIFY_KEYS)][:20]}
                save()
                if code_exit == 0:
                    break
                if arm == "C-C" and k == 1 and deadline - RESERVE_S - time.time() < ARM_MIN[stage]:
                    rec["arms"][arm]["second_try"] = "no time"
                    break
            if arm == "S-C" and rec["arms"][arm].get("exit") != 0:
                rec["stop"] = "S-C failed; the connector fails without the compact layout"
                break
            if arm in ("C-C", "S-C") and rec["arms"][arm].get("exit") == 0:     # A1 F10 abort: no preemption
                try:
                    r = json.loads((run_root / arm / "run_e69.json").read_text())
                    if r["counters"]["preemptions"] == 0:
                        rec["stop"] = f"{arm} had 0 preemptions"
                        break
                except Exception:  # noqa: BLE001
                    pass
        an = subprocess.run([py, f"{code}/gemma4_kv_offload/analyze_e69.py", str(run_root)]
                            + (["--rehearsal"] if stage == "reh" else []), capture_output=True, text=True)
        rec["decision"] = (an.stdout + an.stderr).strip()[-2_000:]
        rec["ok"] = an.returncode == 0 and (run_root / "summary.json").exists()
    stop.set()
    rec["seconds"] = round(time.time() - t0, 1)
    save()
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
