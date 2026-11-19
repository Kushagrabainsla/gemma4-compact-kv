#!/bin/sh
# E-65 A1 (R-101 B1): the free Mac bracket, r = 1.0 then r = 0.625 (about 1 hour each on one core at nice 10).
set -u
E65=$(cd "$(dirname "$0")" && pwd)
OUT=$E65/results/bracket
UV=${UV:-~/.local/bin/uv}
mkdir -p "$OUT"
for r in 1.0 0.625; do
    "$UV" run --no-project python -B "$E65/make_bracket.py" "$r" "$OUT/fit_r$r.json"
    nice -n 10 "$UV" run --no-project python -B "$E65/e65_model.py" "$OUT/fit_r$r.json" "$E65/prefill_fit.json" \
        "$OUT/e65_bracket_r$r.json"
done
