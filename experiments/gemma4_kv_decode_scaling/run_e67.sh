#!/bin/sh
# E-67: the three CPU offload modes, one after the other, on one core at nice 10 (about 35 minutes each).
set -u
E=$(cd "$(dirname "$0")" && pwd)
UV=${UV:-/Users/kushagrabainsls/.local/bin/uv}
mkdir -p "$E/results/e67"
MODES=${*:-prompt200 all200 all64}
for mode in $MODES; do
    nice -n 10 "$UV" run --no-project python -B "$E/e67.py" "$mode" "$E/results/e67/e67_$mode.json"
done
