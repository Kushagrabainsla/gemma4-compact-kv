#!/bin/sh
# E-68 part 2, EXPLORATORY (explore_e68.py), after S6 attempt 3: the frozen word is INVALID (V3), so run_e68.sh stops.
# The same steps as run_e68.sh on one core at nice 10, with every output labelled EXPLORATORY. It decides nothing.
# Usage: run_explore_e68.sh <run folder with fit_amended.json>
set -u
E=experiments/gemma4_kv_decode_scaling
UV=${UV:-~/.local/bin/uv}
RUN=$1
OUT=$RUN/part2_exploratory
mkdir -p "$OUT"
PY="nice -n 10 $UV run --no-project python -B"
X=$E/explore_e68.py
FIT=$OUT/fit_explore.json
$PY "$X" fit "$RUN/fit_amended.json" "$FIT" | tee "$OUT/fit.txt"
$PY "$E/e68.py" --ratio "$FIT" | tee "$OUT/read_ratio.txt"
$PY "$X" none "$FIT" "$E/prefill_fit.json" "$OUT/e65_meas.json" > "$OUT/e65_meas.log" 2>&1 || { tail -3 "$OUT/e65_meas.log"; exit 2; }
for mode in prompt200 all200; do
    $PY "$X" mode "$FIT" "$mode" "$OUT/e68_$mode.json" > "$OUT/e68_$mode.log" 2>&1 || { tail -3 "$OUT/e68_$mode.log"; exit 2; }
done
$PY "$X" word "$OUT/e68_prompt200.json" "$OUT/e68_all200.json" "$OUT/e65_meas.json" "$FIT" > "$OUT/word.txt" 2>&1
head -1 "$OUT/word.txt"
