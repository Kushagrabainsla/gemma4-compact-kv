#!/bin/sh
# E-65 Mac side (RUNPOD.md Sections 3 and 5): upload the code, check the hashes on the Pod, start the session in its
# own process session, and check that it runs. Exits at once with a non-zero code if a step fails (E-54 attempt 1).
# Usage: run_pod_e65.sh <ssh host> <ssh port> <reh|main>
set -u
HOST=$1; PORT=$2; MODE=$3
REPO=.
EXP=$REPO/experiments
KEY=$HOME/.ssh/runpod_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$HOST"
REH=0; [ "$MODE" = reh ] && REH=1
fail() { echo "FAIL: $*"; exit 2; }
$SSH "mkdir -p /workspace/code/gemma4_kv_decode_scaling /workspace/code/gemma4_kv_microbench /workspace/code/gemma4_kv_engine /workspace/code/gemma4_kv_model /workspace/code/gemma4_kv_capacity_model /workspace/code/runpod" || fail "mkdir"
D=$EXP/gemma4_kv_decode_scaling
scp -i "$KEY" -P "$PORT" -q "$D/bench_e65.py" "$D/analyze_e65.py" "$D/session_e65.sh" "$D/boot_end.py" "$D/stop_pod.py" \
    "$D/versions.py" "$D/check_manifest.py" "$D/code_manifest.json" "root@$HOST:/workspace/code/gemma4_kv_decode_scaling/" || fail "upload e65"
scp -i "$KEY" -P "$PORT" -q "$EXP/gemma4_kv_microbench/bench_e58.py" "root@$HOST:/workspace/code/gemma4_kv_microbench/" || fail "upload e58"
scp -i "$KEY" -P "$PORT" -q "$EXP/gemma4_kv_engine/compact_engine.py" "$EXP/gemma4_kv_engine/compact_layout.py" \
    "root@$HOST:/workspace/code/gemma4_kv_engine/" || fail "upload engine"
scp -i "$KEY" -P "$PORT" -q "$EXP/gemma4_kv_model/splice_e55.py" "root@$HOST:/workspace/code/gemma4_kv_model/" || fail "upload model"
scp -i "$KEY" -P "$PORT" -q "$EXP/gemma4_kv_capacity_model/e64.py" "$EXP/gemma4_kv_capacity_model/poolsim.py" \
    "$EXP/gemma4_kv_capacity_model/sweep.py" "root@$HOST:/workspace/code/gemma4_kv_capacity_model/" || fail "upload capacity"
scp -i "$KEY" -P "$PORT" -q "$EXP/runpod/hardware_record.sh" "root@$HOST:/workspace/code/runpod/" || fail "upload runpod"
$SSH "python3 /workspace/code/gemma4_kv_decode_scaling/check_manifest.py /workspace/code /workspace/code/gemma4_kv_decode_scaling/code_manifest.json" \
    > "$D/results/pod_sha_prestart_$MODE.txt" 2>&1 || { cat "$D/results/pod_sha_prestart_$MODE.txt"; fail "manifest"; }
tail -1 "$D/results/pod_sha_prestart_$MODE.txt"
$SSH "cd /workspace && AUTO_STOP=1 REH=$REH nohup python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' sh /workspace/code/gemma4_kv_decode_scaling/session_e65.sh > /workspace/session_out.txt 2>&1 < /dev/null &" || fail "start"
sleep 15
$SSH "pgrep -f session_e65.sh > /dev/null && tail -3 /workspace/results/e65/session.log" || fail "the session is not running"
echo "STARTED $MODE"
