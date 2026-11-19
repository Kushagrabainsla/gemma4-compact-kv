#!/bin/sh
# E-66: the three prefix-cache modes, one after the other, on one core at nice 10 (about 40 minutes each).
set -u
E=$(cd "$(dirname "$0")" && pwd)
UV=${UV:-/Users/kushagrabainsls/.local/bin/uv}
mkdir -p "$E/results/e66"
for mode in engine global perfect; do
    nice -n 10 "$UV" run --no-project python -B "$E/e66.py" "$mode" "$E/results/e66/e66_$mode.json"
done
