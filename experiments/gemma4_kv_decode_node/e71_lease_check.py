"""E-71 lease check (R-120 F8): for each main arm, the gap from P's q-file write to P's NIXL lease release, and from
the release to D's late read notice, for every request whose lease expired.

Inputs: `results/e71-main-20261116T202539Z/lease_check/q_<arm>.json` (`modal volume ls --json` of the arm's q folder;
times in the local zone) and `<arm>/p_stdout.txt` (log times in UTC).
Usage: e71_lease_check.py   -> prints a JSON summary
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

RES = Path(__file__).resolve().parent / "results" / "e71-main-20261116T202539Z"
REL = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[base_worker.py:\d+\] Releasing expired KV blocks for request p(\d+)")
LATE = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[pull_worker.py:\d+\] Potentially invalid KV blocks for unrecognized "
                  r"request p(\d+)")


def utc(stamp: str) -> datetime:
    return datetime.strptime("2026-" + stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def main() -> None:
    out = {}
    for arm in ("S1", "C", "S2"):
        q = {int(Path(e["filename"]).stem): datetime.fromisoformat(e["created_modified"]).astimezone(timezone.utc)
             for e in json.loads((RES / "lease_check" / f"q_{arm}.json").read_text())}
        log = (RES / arm / "p_stdout.txt").read_text()
        rel = {int(m.group(2)): utc(m.group(1)) for m in REL.finditer(log)}
        late = {int(m.group(2)): utc(m.group(1)) for m in LATE.finditer(log)}
        lease = sorted(round((rel[i] - q[i]).total_seconds()) for i in rel if i in q)
        after = sorted(round((late[i] - rel[i]).total_seconds()) for i in late if i in rel)
        added = json.loads((RES / arm / "d_run.json").read_text())["window"]["added"]
        out[arm] = {"requests_prefilled": len(q), "added_by_d": added, "expired": len(rel), "late_reads": len(late),
                    "late_share_of_added": round(len(late) / added, 3),
                    "late_without_release": sorted(set(late) - set(rel)),
                    "release_without_late": sorted(set(rel) - set(late)),
                    "write_to_release_s": [min(lease), max(lease)] if lease else None,
                    "release_to_read_s_min_median_max": [after[0], after[len(after) // 2], after[-1]] if after else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
