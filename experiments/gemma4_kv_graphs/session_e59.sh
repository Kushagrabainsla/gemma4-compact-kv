#!/bin/sh
# E-59 session (README Sections 4 and 9): the version and hardware record, telemetry, the SHA-256 of the code that runs
# (L-25), the pinned checkpoint download to the container disk (L-32), the seven engine processes in the order of
# README Section 4.5 (the graph block G1 to G4 first, then the eager block E1 to E3), the frozen analysis, and the same
# safety as E-57: a session watchdog, a timeout for each engine, AUTO_STOP (the run script sets AUTO_STOP=1 on the
# Pod) and a deadline BOOT_MARGIN seconds before the boot watchdog (pod_cmd.py; its "armed cap" line in BOOT_LOG). An
# engine does not start if less than MIN_START seconds are left. Exit 3 of run_e59.py is a pre-flight stop (the
# record says why). STUB=1 runs the same flow on a Mac (no GPU, no vLLM, no download). REH=1 is the S5 rehearsal on
# Gemma 4 12B at 16K tokens.
set -u
CODE=${CODE:-/workspace/code}
E59=$CODE/gemma4_kv_graphs
RES=${RES:-/workspace/results/e59}
STUB=${STUB:-0}
STUB_PLANT=${STUB_PLANT:-}
REH=${REH:-0}
if [ "$REH" = 1 ]; then
    M59=${M59:-google/gemma-4-12B-it}; R59=${R59:-43c9d014b231474fa7212dbb48a762566c2b255d}
else
    M59=${M59:-google/gemma-4-31B-it}; R59=${R59:-842da3794eaa0b77d5f08bae87a17459d91ff475}
fi
WEIGHTS=${WEIGHTS:-/root/hf}            # L-32: the weights on the container disk, not on the network volume
DL_TIMEOUT=${DL_TIMEOUT:-1200}          # 62.6 GB in up to 3 resumed tries (E-56: 220 s; E-57: 555 s)
POD_LIMIT=${POD_LIMIT:-3300}            # session watchdog
BOOT_LOG=${BOOT_LOG:-/workspace/boot_watchdog.log}
BOOT_MARGIN=${BOOT_MARGIN:-240}
AN_RESERVE=${AN_RESERVE:-60}            # seconds kept for the analysis
MIN_START=${MIN_START:-300}             # A1 (R-86 F8): an engine does not start with less time left
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-240} # the copy window for RES
AUTO_STOP=${AUTO_STOP:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -n "${VLLM_CACHE_ROOT:-}" ]; then
    export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
    export FLASHINFER_WORKSPACE_BASE=$VLLM_CACHE_ROOT/flashinfer
fi
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH
if [ -r /proc/1/environ ]; then
    for v in RUNPOD_POD_ID RUNPOD_API_KEY HF_TOKEN; do
        eval "cur=\${$v:-}"; [ -n "$cur" ] && continue
        val=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1); [ -n "$val" ] && export "$v=$val"
    done
fi
mkdir -p "$RES"; START=$(date +%s); STAMP=$(date -u +%Y%m%dT%H%M%SZ); DEADLINE=$((START + POD_LIMIT))
BOOT_END=""
if [ -r "$BOOT_LOG" ]; then
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
run_to() {
    s=$1; shift
    if command -v timeout > /dev/null 2>&1; then timeout -k 10 "$s" "$@"; else "$@"; fi
}
sha_list() {
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

# The code that runs (L-25): the E-59 package and the E-57 and E-55 files that it imports.
{ sha_list "$E59" run_e59.py analyze_e59.py session_e59.sh
  sha_list "$CODE/gemma4_kv_engine" compact_engine.py compact_layout.py run_e57.py
  sha_list "$CODE/gemma4_kv_model" splice_e55.py run_e55.py analyze_e55.py; } > "$RES/code_sha.txt"
[ -f "$E59/code_manifest.json" ] && cp "$E59/code_manifest.json" "$RES/code_manifest.json"
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
SNAP=""; FLAGS="--model-id $M59 --revision $R59"; [ "$REH" = 1 ] && FLAGS="$FLAGS --rehearsal"
if [ "$STUB" = 1 ]; then
    echo "vllm STUB" > "$RES/versions.txt"
    FLAGS="$FLAGS --stub"; [ -n "$STUB_PLANT" ] && FLAGS="$FLAGS --stub-plant $STUB_PLANT"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    { python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1
    mkdir -p "$WEIGHTS"
    log "disk before the download: $(df -h "$WEIGHTS" 2>/dev/null | tail -1)"
    DL_END=$(( $(date +%s) + DL_TIMEOUT )); k=0
    while [ -z "$SNAP" ] && [ $k -lt 3 ] && [ "$(date +%s)" -lt "$DL_END" ]; do
        k=$((k + 1))
        HF_HOME=$WEIGHTS run_to $(( DL_END - $(date +%s) )) python3 -c "
import sys
from huggingface_hub import snapshot_download
for m, r in zip(sys.argv[1::2], sys.argv[2::2]):
    print(m, r, snapshot_download(m, revision=r), flush=True)
" "$M59" "$R59" >> "$ENV/models.txt" 2>&1 || log "WARNING: model download try $k failed (exit $?)"
        SNAP=$(awk -v m="$M59" '$1 == m {print $3}' "$ENV/models.txt" | tail -1)
    done
    log "download: ${SNAP:-none} after $(elapsed) s; $(df -h "$WEIGHTS" 2>/dev/null | tail -1)"
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"
log "deadline in $(left) s (session watchdog $POD_LIMIT s; boot watchdog end ${BOOT_END:-unknown}, margin $BOOT_MARGIN s)"

STOP=""
engine() {  # $1 name, $2 layout, $3 graphs, $4 capture mode, $5 rows, $6 profiled rows
    name=$1; out="$RES/$1"; mkdir -p "$out"
    if [ -n "$STOP" ]; then log "$name: SKIPPED ($STOP)"; return 0; fi
    if [ -f "$out/run_e59.json" ] && grep -q '"status": "ok"' "$out/run_e59.json"; then
        log "$name: ok before, skip"; return 0
    fi
    [ -e "$out/run_e59.json" ] && mv "$out/run_e59.json" "$out/run_e59.failed.$(date -u +%Y%m%dT%H%M%SZ).json"
    if [ "$(left)" -lt $(( MIN_START + AN_RESERVE )) ]; then log "$name: SKIPPED ($(left) s left)"; return 0; fi
    dl=$(( DEADLINE - AN_RESERVE ))
    if [ "$STUB" = 1 ]; then
        python3 "$E59/run_e59.py" --engine "$name" --layout "$2" --graphs "$3" --capture "$4" --rows "$5" \
            --profile-rows "$6" --out "$out" --deadline "$dl" $FLAGS > "$out/log.txt" 2>&1; rc=$?
    elif [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        run_to $(( dl - $(date +%s) + 20 )) python3 "$E59/run_e59.py" --engine "$name" --layout "$2" --graphs "$3" \
            --capture "$4" --rows "$5" --profile-rows "$6" --out "$out" --deadline "$dl" --snapshot "$SNAP" $FLAGS \
            > "$out/log.txt" 2>&1; rc=$?
    else
        rc=2; echo "no snapshot folder after the download" > "$out/log.txt"
    fi
    [ "$rc" = 3 ] && log "$name: PREFLIGHT STOP; see $out/run_e59.json"
    grep -E "GPU KV cache size|Maximum concurrency|cudagraph|Capturing|Graph capturing finished" "$out/log.txt" \
        2>/dev/null | head -4 | while IFS= read -r l; do log "$name engine: $(printf '%s' "$l" | cut -c1-200)"; done
    log "$name: exit $rc after $(elapsed) s; $(tail -1 "$out/log.txt" | cut -c1-300)"
    return 0
}
engine G1 stock 1 OFF "STOCK:3,STOCK:3" 1
# A1 (R-86 B3): without a working G1 the decision cannot exist; the session stops and saves the money.
grep -q '"status": "ok"' "$RES/G1/run_e59.json" 2>/dev/null || STOP="G1 did not end ok"
engine G2 stock 1 VIEWS "EMULV:3,EMULV:3" ""
engine G3 compact 1 VIEWS "COMPACT:3,COMPACT:3,COMPACT:4" 1
engine G4 stock 1 OFF "STOCK:3" ""
engine E1 stock 0 OFF "STOCK:3,EMULV:3,STOCK:3" 1
engine E2 compact 0 VIEWS "COMPACT:3,COMPACT:3" ""
engine E3 stock 0 OFF "STOCK:3,EMULV:3,STOCK:3" ""   # A3: the same row positions as E1
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E59/analyze_e59.py" --res "$RES" > "$RES/decision.txt" 2>&1 || true
log "decision: $(tail -1 "$RES/decision.txt" | cut -c1-900)"
log "results: $(du -sk "$RES" | cut -f1) KB in $RES; disk: $(df -h "$RES" 2>/dev/null | tail -1)"
log "done after $(elapsed) s"
