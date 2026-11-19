"""Write code_manifest.json: the SHA-256 of every file that runs in the E-69 container (PROTOCOL.md principle 6)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e69_stage as stg  # noqa: E402

stg.MANIFEST_LOCAL.write_text(json.dumps(stg.make_manifest(), indent=1, sort_keys=True) + "\n")
print(f"{len(stg.make_manifest())} files -> {stg.MANIFEST_LOCAL}")
