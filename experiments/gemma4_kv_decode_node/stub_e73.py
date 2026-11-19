"""E-73 stub (stage S3): the E-71 stub with the D3Q sequence (routed by `run_e73.install`), plus the P lease line that
vLLM prints at the start (the W3 positive control). `STUB_RATIO` sets C over S tokens per step (default 1.24, the
expected engine value with the D3H bias); `STUB_LEASE=1` writes one lease-expiry line."""
from __future__ import annotations

import os

import stub_e71


def run(args, plan: dict, rec: dict) -> None:
    os.environ.setdefault("STUB_RATIO", "1.24")
    if args.role == "p":
        print("INFO [startup.py:96] non-default args: {'kv_connector_extra_config': {'ucx_backend_params': {}, "
              "'kv_lease_duration': 600}}", flush=True)
        if os.environ.get("STUB_LEASE") == "1":
            print("WARNING [base_worker.py:3402] Releasing expired KV blocks for request p3 which were retrieved by 0 "
                  "remote worker(s) before lease expired.", flush=True)
    stub_e71.run(args, plan, rec)
    rec["hooks"] = {"hooked": [5], "problems": []}
    rec["mode"] = "VIEWS" if args.arm == "C" else "OFF"
