"""On the Pod, before the session: the SHA-256 of each uploaded file must equal `code_manifest.json` (PROTOCOL.md
principle 6, R-76 C7). Prints one line per file and exits 1 on any difference or a missing file.

Usage: check_manifest.py <code root> <manifest json>
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT, MAN = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
FOLDERS = ("gemma4_kv_decode_scaling", "gemma4_kv_microbench", "gemma4_kv_engine", "gemma4_kv_model",
           "gemma4_kv_capacity_model")
bad = 0
for name, want in sorted(MAN.items()):
    hits = [ROOT / f / name for f in FOLDERS if (ROOT / f / name).exists()]
    got = hashlib.sha256(hits[0].read_bytes()).hexdigest() if hits else "missing"
    ok = got == want
    bad += not ok
    print(("ok  " if ok else "BAD ") + name + " " + got)
print(f"manifest: {len(MAN) - bad} of {len(MAN)} match")
sys.exit(1 if bad else 0)
