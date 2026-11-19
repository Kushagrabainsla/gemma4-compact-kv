"""Build `code_manifest.json` on the Mac (L-25): the SHA-256 of each file that runs on the Pod, as committed at the design
commit (read from git, not from a working copy). The run script uploads it with the package and checks the hashes on
the Pod before the start; the session copies it into the results.

Usage: make_manifest_e65.py --commit <sha> --out <file>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PATHS = {name: f"experiments/{folder}/{name}" for folder, names in (
    ("gemma4_kv_decode_scaling", ("bench_e65.py", "analyze_e65.py", "session_e65.sh", "boot_end.py", "stop_pod.py",
                                  "versions.py")),
    ("gemma4_kv_microbench", ("bench_e58.py",)),
    ("gemma4_kv_engine", ("compact_engine.py", "compact_layout.py")),
    ("gemma4_kv_model", ("splice_e55.py",)),
    ("gemma4_kv_capacity_model", ("e64.py", "poolsim.py", "sweep.py"))) for name in names}


def manifest(commit: str) -> dict[str, str]:
    """The SHA-256 of the blob of each file at the commit (`git cat-file blob`: the bytes as committed)."""
    out = {}
    for name, path in PATHS.items():
        blob = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{commit}:{path}"], capture_output=True,
                              check=True).stdout
        out[name] = hashlib.sha256(blob).hexdigest()
    return out


def manifest_working() -> dict[str, str]:
    """A1: the SHA-256 of each file in the working copy (the design is frozen by these hashes, with no commit)."""
    return {name: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for name, path in PATHS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="")
    ap.add_argument("--working", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = manifest_working() if args.working else manifest(args.commit)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{args.out}: {len(PATHS)} files at {'the working copy' if args.working else args.commit}")


if __name__ == "__main__":
    main()
