"""E-69 probe (README Section 2 and 9): the container facts that can stop the run before a GPU is billed.

Prints one JSON object: versions, /dev/shm size, the cgroup memory limit, the memlock limit, and whether every vLLM
function that the hooks wrap exists at the pin. Runs on a CPU container (probe) and again at the start of each GPU
session.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
from pathlib import Path

HOOKS = (("vllm.v1.core.sched.scheduler", "Scheduler", "make_stats"),
         ("vllm.v1.core.sched.scheduler", "Scheduler", "_preempt_request"),
         ("vllm.v1.core.sched.scheduler", "Scheduler", "_update_waiting_for_remote_kv"),
         ("vllm.v1.core.sched.scheduler", "Scheduler", "schedule"),
         ("vllm.v1.core.kv_cache_manager", "KVCacheManager", "get_block_ids"),
         ("vllm.v1.core.kv_cache_manager", "KVCacheManager", "allocate_slots"),
         ("vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker", "OffloadingConnectorWorker",
          "register_kv_caches"),
         ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "get_kv_cache_spec"))


def cgroup_memory() -> str | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if Path(p).exists():
            return Path(p).read_text().strip()
    return None


def main() -> None:
    out: dict = {}
    try:
        import torch
        import triton
        import vllm
        out["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__, "triton": triton.__version__}
    except Exception as e:  # noqa: BLE001
        out["versions"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    du = shutil.disk_usage("/dev/shm") if Path("/dev/shm").exists() else None
    out["dev_shm"] = {"total_gib": du.total / 2**30, "free_gib": du.free / 2**30} if du else None
    out["cgroup_memory"] = cgroup_memory()
    out["memlock"] = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    out["cpus"] = os.cpu_count()
    try:
        out["mem_total_gib"] = int(next(ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemTotal"))) / 2**20
    except Exception:  # noqa: BLE001
        out["mem_total_gib"] = None
    hooks = {}
    import importlib
    for mod, cls, fn in HOOKS:
        try:
            hooks[f"{cls}.{fn}"] = callable(getattr(getattr(importlib.import_module(mod), cls), fn, None))
        except Exception as e:  # noqa: BLE001
            hooks[f"{cls}.{fn}"] = f"{type(e).__name__}: {e}"[:200]
    out["hooks"] = hooks
    out["hooks_ok"] = all(v is True for v in hooks.values())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
