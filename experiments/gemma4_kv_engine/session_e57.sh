#!/bin/sh
# E-57 session (README Sections 4 and 9): the version and hardware record, telemetry, the SHA-256 of the code that runs
# (L-25), the pinned checkpoint download, engine A (the stock cache: STOCK, EMULV, DET, the capacity phase), engine B
# (the compact cache: COMPACT, the capacity phase with the equal-batch rows, GROSS and NAIVE while the time allows), and
# the frozen decision. Each engine runs in its own process. The same safety as E-56: a session watchdog, a timeout for
# each engine and AUTO_STOP (the run script sets AUTO_STOP=1 on the Pod). The Pod start command adds the boot watchdog
# (experiments/runpod/pod_cmd.py --cap 3780); A1: the deadline also ends BOOT_MARGIN seconds before that watchdog
# (its "armed cap" line in BOOT_LOG). Exit 3 of run_e57.py is a pre-flight or health stop. STUB=1 runs the
# same flow on a Mac (no GPU, no vLLM, no download). REH=1 is the S5 rehearsal on Gemma 4 12B (the same attention code
# and K=V global layers; M57 and R57 name it).
set -u
CODE=${CODE:-/workspace/code}
E57=$CODE/gemma4_kv_engine
E56=$CODE/gemma4_kv_tasks
E55=$CODE/gemma4_kv_model
RES=${RES:-/workspace/results/e57}
STUB=${STUB:-0}
STUB_ARGS=${STUB_ARGS:-}
REH=${REH:-0}
if [ "$REH" = 1 ]; then
    M57=${M57:-google/gemma-4-12B-it}; R57=${R57:-43c9d014b231474fa7212dbb48a762566c2b255d}
else
    M57=${M57:-google/gemma-4-31B-it}; R57=${R57:-842da3794eaa0b77d5f08bae87a17459d91ff475}
fi
DL_TIMEOUT=${DL_TIMEOUT:-1500}          # 62.6 GB (31B) in up to 3 resumed tries; E-56 read it in under 220 s
POD_LIMIT=${POD_LIMIT:-3600}            # session watchdog at 60 min
BOOT_LOG=${BOOT_LOG:-/workspace/boot_watchdog.log}  # the boot watchdog's "armed cap" line (pod_cmd.py)
BOOT_MARGIN=${BOOT_MARGIN:-240}         # A1 (R-82 F5): the session ends this long before the boot watchdog stops the Pod
AN_RESERVE=${AN_RESERVE:-120}           # seconds kept for the analysis after engine B
A_SHARE=${A_SHARE:-52}                  # percent of the time after the setup for engine A (A1: 45 before)
AUTO_STOP_DELAY=${AUTO_STOP_DELAY:-300} # the copy window for RES
AUTO_STOP=${AUTO_STOP:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=0  # the engine core and the worker in this process (E-53 README Section 4)
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
sha_list() {  # the SHA-256 of every file under $1, with paths relative to $1 (RUNPOD.md Section 6 format)
    if command -v sha256sum > /dev/null 2>&1; then
        find "$1" -type f -exec sha256sum {} + | sed "s|$1/||" | sort -k 2
    else
        find "$1" -type f -exec shasum -a 256 {} + | sed "s|$1/||" | sort -k 2
    fi
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

# The code that runs (L-25): the E-57 package, the E-56 files that it imports or reads, and the E-55 files.
{ sha_list "$E57"; sha_list "$E56" | awk '$2 == "analyze_e56.py" || $2 == "data_e56.json"';
  sha_list "$E55" | awk '$2 == "splice_e55.py" || $2 == "run_e55.py" || $2 == "analyze_e55.py"'; } > "$RES/code_sha.txt"
[ -f "$E57/code_manifest.json" ] && cp "$E57/code_manifest.json" "$RES/code_manifest.json"
ENV="$RES/env-$STAMP"; mkdir -p "$ENV"
SNAP=""; RUN_FLAGS=""; [ "$REH" = 1 ] && RUN_FLAGS="--rehearsal"
if [ "$STUB" = 1 ]; then
    echo "vllm STUB" > "$RES/versions.txt"
else
    nvidia-smi -q > "$ENV/nvidia-smi.txt" 2>&1
    [ -f "$CODE/runpod/hardware_record.sh" ] && sh "$CODE/runpod/hardware_record.sh" "$ENV" > /dev/null 2>&1
    TZ=UTC nvidia-smi --format=csv,nounits -lms 1000 --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used > "$ENV/telemetry.csv" 2>&1 &
    TEL=$!
    { python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
      echo "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unset}"; } > "$RES/versions.txt" 2>&1
    log "disk before the download: $(df -h /workspace 2>/dev/null | tail -1)"
    DL_END=$(( $(date +%s) + DL_TIMEOUT )); k=0
    while [ -z "$SNAP" ] && [ $k -lt 3 ] && [ "$(date +%s)" -lt "$DL_END" ]; do   # a try resumes
        k=$((k + 1))
        run_to $(( DL_END - $(date +%s) )) python3 -c "
import sys
from huggingface_hub import snapshot_download
for m, r in zip(sys.argv[1::2], sys.argv[2::2]):
    print(m, r, snapshot_download(m, revision=r), flush=True)
" "$M57" "$R57" >> "$ENV/models.txt" 2>&1 || log "WARNING: model download try $k failed (exit $?)"
        SNAP=$(awk -v m="$M57" '$1 == m {print $3}' "$ENV/models.txt" | tail -1)
    done
fi
log "setup done after $(elapsed) s; versions: $(tr '\n' ' ' < "$RES/versions.txt")"
log "deadline in $(left) s (session watchdog $POD_LIMIT s; boot watchdog end ${BOOT_END:-unknown}, margin $BOOT_MARGIN s)"

AVAIL=$(( $(left) - AN_RESERVE )); A_END=$(( $(date +%s) + AVAIL * A_SHARE / 100 ))
FLAGS="--data $E56/data_e56.json --model-id $M57 --revision $R57 --start $START --reserve 60 $RUN_FLAGS"
engine() {  # $1 layout, $2 deadline (epoch), then extra flags
    layout=$1; dl=$2; shift 2
    out="$RES/$layout"; mkdir -p "$out"
    if [ -f "$out/run_main.json" ] && grep -q '"status": "ok"' "$out/run_main.json"; then
        log "$layout: ok before, skip"; return 0
    fi
    [ -e "$out/run_main.json" ] && mv "$out/run_main.json" "$out/run_main.failed.$(date -u +%Y%m%dT%H%M%SZ).json"
    if [ "$STUB" = 1 ]; then
        python3 "$E57/run_e57.py" --layout "$layout" --out "$out" --deadline "$dl" $FLAGS --stub $STUB_ARGS "$@" \
            > "$out/main_log.txt" 2>&1; rc=$?
    elif [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        run_to $(( dl - $(date +%s) + 30 )) python3 "$E57/run_e57.py" --layout "$layout" --out "$out" \
            --deadline "$dl" --snapshot "$SNAP" $FLAGS "$@" > "$out/main_log.txt" 2>&1; rc=$?
    else
        rc=2; echo "no snapshot folder after the download" > "$out/main_log.txt"
    fi
    [ "$rc" = 3 ] && log "$layout: PREFLIGHT OR HEALTH STOP; see $out/run_main.json"
    grep -E "GPU KV cache size|Maximum concurrency|kv cache group sizes|kv lcm block sizes" "$out/main_log.txt" \
        2>/dev/null | head -6 | while IFS= read -r l; do log "$layout engine: $(printf '%s' "$l" | cut -c1-200)"; done
    log "$layout: exit $rc after $(elapsed) s; $(tail -1 "$out/main_log.txt" | cut -c1-300)"
    return 0
}
engine stock "$A_END"
engine compact $(( DEADLINE - AN_RESERVE )) --stock-capacity "$RES/stock/capacity.json"
[ -n "$TEL" ] && kill "$TEL" 2>/dev/null
python3 "$E57/analyze_e57.py" --res "$RES" --data "$E56/data_e56.json" > "$RES/decision.txt" 2>&1 || true
log "decision: $(tail -1 "$RES/decision.txt" | cut -c1-900)"
log "results: $(du -sk "$RES" | cut -f1) KB in $RES; disk: $(df -h "$RES" 2>/dev/null | tail -1)"
log "done after $(elapsed) s"
