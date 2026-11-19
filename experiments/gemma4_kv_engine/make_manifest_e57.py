"""Build `code_manifest.json` on the Mac (L-25): the SHA-256 of each file that runs on the Pod, as committed at the design
commit (read from git, not from a working copy). The run script uploads it with the package and checks the hashes on
the Pod before the start; the session copies it into the results.

Usage: make_manifest_e57.py --commit <sha> --out <file>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PATHS = {name: f"experiments/{folder}/{name}" for folder, names in (
    ("gemma4_kv_engine", ("compact_layout.py", "compact_engine.py", "run_e57.py", "analyze_e57.py", "session_e57.sh")),
    ("gemma4_kv_tasks", ("analyze_e56.py", "data_e56.json")),
    ("gemma4_kv_model", ("splice_e55.py", "run_e55.py", "analyze_e55.py"))) for name in names}


def manifest(commit: str) -> dict[str, str]:
    """The SHA-256 of the blob of each file at the commit (`git cat-file blob`: the bytes as committed)."""
    out = {}
    for name, path in PATHS.items():
        blob = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{commit}:{path}"], capture_output=True,
                              check=True).stdout
        out[name] = hashlib.sha256(blob).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(json.dumps(manifest(args.commit), indent=1, sort_keys=True) + "\n")
    print(f"{args.out}: {len(PATHS)} files at {args.commit}")


if __name__ == "__main__":
    main()
