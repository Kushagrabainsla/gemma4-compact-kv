#!/bin/sh
# E-68 attempt 3 (D-120) on one Runpod A100 SXM: the E-65 Mac-side upload (run_pod_e65.sh) with the E-68 manifest.
# It uploads the code, checks the 13 hashes on the Pod against code_manifest_e68.json, starts the unchanged E-65
# session in its own process session (AUTO_STOP=1, REH=0, no E68_TIE_RULE: the frozen absolute tie), and checks that
# it runs. Exits at once with a non-zero code if a step fails (E-54 attempt 1).
# Usage: run_pod_e68.sh <ssh host> <ssh port> <local results dir>
set -u
HOST=$1; PORT=$2; OUT=$3
REPO=.
EXP=$REPO/experiments
KEY=$HOME/.ssh/runpod_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$HOST"
SCP="scp -i $KEY -P $PORT -o StrictHostKeyChecking=accept-new -q"
fail() { echo "FAIL: $*"; exit 2; }
mkdir -p "$OUT"
$SSH "mkdir -p /workspace/code/gemma4_kv_decode_scaling /workspace/code/gemma4_kv_microbench /workspace/code/gemma4_kv_engine /workspace/code/gemma4_kv_model /workspace/code/gemma4_kv_capacity_model /workspace/code/runpod" || fail "mkdir"
D=$EXP/gemma4_kv_decode_scaling
$SCP "$D/bench_e65.py" "$D/analyze_e65.py" "$D/session_e65.sh" "$D/boot_end.py" "$D/stop_pod.py" "$D/versions.py" \
    "$D/check_manifest.py" "root@$HOST:/workspace/code/gemma4_kv_decode_scaling/" || fail "upload e65"
$SCP "$D/code_manifest_e68.json" "root@$HOST:/workspace/code/gemma4_kv_decode_scaling/code_manifest.json" || fail "upload manifest"
$SCP "$EXP/gemma4_kv_microbench/bench_e58.py" "root@$HOST:/workspace/code/gemma4_kv_microbench/" || fail "upload e58"
$SCP "$EXP/gemma4_kv_engine/compact_engine.py" "$EXP/gemma4_kv_engine/compact_layout.py" \
    "root@$HOST:/workspace/code/gemma4_kv_engine/" || fail "upload engine"
$SCP "$EXP/gemma4_kv_model/splice_e55.py" "root@$HOST:/workspace/code/gemma4_kv_model/" || fail "upload model"
$SCP "$EXP/gemma4_kv_capacity_model/e64.py" "$EXP/gemma4_kv_capacity_model/poolsim.py" \
    "$EXP/gemma4_kv_capacity_model/sweep.py" "root@$HOST:/workspace/code/gemma4_kv_capacity_model/" || fail "upload capacity"
$SCP "$EXP/runpod/hardware_record.sh" "root@$HOST:/workspace/code/runpod/" || fail "upload runpod"
$SSH "python3 /workspace/code/gemma4_kv_decode_scaling/check_manifest.py /workspace/code /workspace/code/gemma4_kv_decode_scaling/code_manifest.json" \
    > "$OUT/pod_sha_prestart.txt" 2>&1 || { cat "$OUT/pod_sha_prestart.txt"; fail "manifest"; }
tail -1 "$OUT/pod_sha_prestart.txt"
$SSH "cd /workspace && AUTO_STOP=1 REH=0 nohup python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' sh /workspace/code/gemma4_kv_decode_scaling/session_e65.sh > /workspace/session_out.txt 2>&1 < /dev/null &" || fail "start"
sleep 15
$SSH "pgrep -f session_e65.sh > /dev/null && tail -3 /workspace/results/e65/session.log" || fail "the session is not running"
echo "STARTED"
