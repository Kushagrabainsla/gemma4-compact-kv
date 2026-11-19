#!/bin/sh
# E-68 part 2 on the Mac (README_E68.md Section 4.4): one core at nice 10, after the main GPU run.
# 0. A2: the analysis with the amended SL and H spread rule (fit_a2.json); the frozen fit.json is kept.
# 1. e65_model.py with the measured table: the `none` labels (no offload) of all 10 workloads (about 1 hour).
# 2. e68.py prompt200 and all200 with the measured table and the two bracket tables (about 50 minutes each).
# 3. the E-68 word.
# Usage: run_e68.sh <run folder with e65/bench/fit.json>
set -u
E=$(cd "$(dirname "$0")" && pwd)
UV=${UV:-~/.local/bin/uv}
RUN=$1
OUT=$RUN/part2
mkdir -p "$OUT"
PY="nice -n 10 $UV run --no-project python -B"
$PY "$E/e68.py" --amend "$RUN/e65/bench" "$OUT/fit_amended.json" > "$OUT/amend.txt" 2>&1 || { cat "$OUT/amend.txt"; exit 2; }
cat "$OUT/amend.txt"
FIT=$OUT/fit_amended.json
$PY "$E/e68.py" --ratio "$FIT" | tee "$OUT/read_ratio.txt"     # report only (A1, R-107 F6)
$PY "$E/e65_model.py" "$FIT" "$E/prefill_fit.json" "$OUT/e65_meas.json" > "$OUT/e65_meas.log" 2>&1 || { tail -3 "$OUT/e65_meas.log"; exit 2; }
for mode in prompt200 all200; do
    $PY "$E/e68.py" "$FIT" "$mode" "$OUT/e68_$mode.json" > "$OUT/e68_$mode.log" 2>&1 || { tail -3 "$OUT/e68_$mode.log"; exit 2; }
done
$PY "$E/e68.py" --word "$OUT/e68_prompt200.json" "$OUT/e68_all200.json" "$OUT/e65_meas.json" "$FIT" > "$OUT/word.txt" 2>&1
head -1 "$OUT/word.txt"
