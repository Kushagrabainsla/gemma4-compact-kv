#!/bin/sh
# E-58 session (README Sections 4 and 9): the version and hardware record, telemetry, the SHA-256 of the code that runs,
# the weight-free benchmark (bench_e58.py, one process, no model download), the frozen analysis, and AUTO_STOP (the run
# script sets AUTO_STOP=1 on the Pod). The Pod start command adds the boot watchdog (pod_cmd.py --cap 900); the deadline
# is the earlier of POD_LIMIT after the start and BOOT_MARGIN seconds before that watchdog (its "armed cap" line in
# BOOT_LOG). STUB=1 runs the same flow on a Mac (no GPU, no vLLM). REH=1 is the S5 rehearsal (the word REHEARSAL).
set -u
CODE=${CODE:-/workspace/code}
E58=$CODE/gemma4_kv_microbench
RES=${RES:-/workspace/results/e58}
STUB=${STUB:-0}
REH=${REH:-0}
POD_LIMIT=${POD_LIMIT:-900}             # session watchdog
BOOT_LOG=${BOOT_LOG:-/workspace/boot_watchdog.log}
BOOT_MARGIN=${BOOT_MARGIN:-120}
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-240}  # the copy window for RES
AUTO_STOP=${AUTO_STOP:-0}
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
fi
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH
if [ -r /proc/1/environ ]; then
    for v in RUNPOD_POD_ID RUNPOD_API_KEY; do
        eval "cur=\${$v:-}"; [ -n "$cur" ] && continue
        val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1); [ -n "$val" ] && export "$v=$val"
    done
fi
mkdir -p "$RES"; START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ); DEADLINE=$((START + POD_LIMIT))
BOOT_END=""
if [ -r "$BOOT_LOG" ]; then  # the last "armed cap" line: its time plus the cap is when the boot watchdog stops the Pod
    BOOT_END=$(python3 -c "
import calendar, sys, time
end = ''
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) >= 4 and p[1] == 'armed' and p[2] == 'cap':
        end = calendar.timegm(time.strptime(p[0], '%Y-%m-%dT%H:%M:%SZ')) + int(p[3])
print(end)
" "$BOOT_LOG" 2>/dev/null)
    if [ -n "$BOOT_END" ] && [ $((BOOT_END - BOOT_MARGIN)) -lt "$DEADLINE" ]; then DEADLINE=$((BOOT_END - BOOT_MARGIN)); fi
fi
elapsed() { echo $(( $(date +%s) - START )); }
left() { echo $(( DEADLINE - $(date +%s) )); }
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RES/session.log"; }
run_to() {  # $1 seconds, then the command: with GNU timeout where it exists (the Pod), else without (a Mac stub)
    s=$1; shift
    if command -v timeout > /dev/null 2>&1; then timeout -k 10 "$s" "$@"; else "$@"; fi
}
sha_list() {  # the SHA-256 of the named files under $1, with paths relative to $1 (RUNPOD.md Section 6 format)
    d=$1; shift
    for f in "$@"; do
        if command -v sha256sum > /dev/null 2>&1; then sha256sum "$d/$f"; else shasum -a 256 "$d/$f"; fi
    done | sed "s|$d/||"
}
stop_pod() {
    [ "$AUTO_STOP" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || return 0
    python3 -c "
import json, os, urllib.request
q = {'query': 'mutation { podStop(input: {podId: \\\"%s\\\"}) { id } }' % os.environ['RUNPOD_POD_ID']}
r = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps(q).encode(), method='POST',
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']})
print(urllib.request.urlopen(r, timeout=30).status)
" > "$RES/stop.txt" 2>&1; log "AUTO_STOP ($1): $(tail -1 "$RES/stop.txt")"
}
TEL=""; WD=""
finish() {
    [ -n "$TEL" ] && kill "$TEL" 2>/dev/null; [ -n "$WD" ] && kill "$WD" 2>/dev/null
    [ "$AUTO_STOP" = 1 ] && { log "AUTO_STOP in $AUTO_STOP_DELAY s; copy RES now"; sleep "$AUTO_STOP_DELAY"; }
    stop_pod exit
}
trap finish EXIT; trap 'exit 130' INT TERM
[ "$AUTO_STOP" = 1 ] && { ( sleep "$POD_LIMIT"; log "watchdog"; stop_pod watchdog ) & WD=$!; }

# The code that runs: the E-58 package and the E-57 and E-55 files that it imports.
{ sha_list "$E58" bench_e58.py analyze_e58.py session_e58.sh
  sha_list "$CODE/gemma4_kv_engine" compact_engine.py compact_layout.py
  sha_list "$CODE/gemma4_kv_model" splice_e55.py; } > "$RES/code_sha.txt"
[ -f "$E58/code_manifest.json" ] && cp "$E58/code_manifest.json" "$RES/code_manifest.json"
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
FLAGS="--out $RES/bench"
[ "$REH" = 1 ] && FLAGS="$FLAGS --rehearsal"
if [ "$STUB" = 1 ]; then
    FLAGS="$FLAGS --stub --layers 2 --rounds 2"
    echo "vllm STUB" > "$RES/versions.txt"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 500 --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    { python3 -c "import vllm, torch, triton; print('vllm', vllm.__version__); print('torch', torch.__version__); print('triton', triton.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"
log "deadline in $(left) s (session watchdog $POD_LIMIT s; boot watchdog end ${BOOT_END:-unknown}, margin $BOOT_MARGIN s)"
run_to $(( $(left) - 45 )) python3 "$E58/bench_e58.py" $FLAGS > "$RES/bench_log.txt" 2>&1; rc=$?
log "bench: exit $rc after $(elapsed) s; $(tail -1 "$RES/bench_log.txt" | cut -c1-300)"
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E58/analyze_e58.py" --res "$RES/bench" > "$RES/decision.txt" 2>&1 || true
log "decision: $(tail -1 "$RES/decision.txt" | cut -c1-1200)"
log "results: $(du -sk "$RES" | cut -f1) KB in $RES"
log "done after $(elapsed) s"
