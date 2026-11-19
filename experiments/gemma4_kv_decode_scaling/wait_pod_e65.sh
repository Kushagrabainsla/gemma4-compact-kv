#!/bin/sh
# E-65 Mac side: wait for the session to end ("done after" in session.log, or a non-zero bench exit), with a limit,
# then copy the results with rsync -rlt and compare SHA-256 lists on both sides (RUNPOD.md Section 6).
# Usage: wait_pod_e65.sh <ssh host> <ssh port> <local results dir> <limit seconds>
set -u
HOST=$1; PORT=$2; OUT=$3; LIMIT=$4
KEY=$HOME/.ssh/runpod_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$HOST"
START=$(date +%s)
while :; do
    LOG=$($SSH "tail -2 /workspace/results/e65/session.log" 2>/dev/null)
    echo "$(date -u +%H:%M:%S) $LOG" | tail -1
    case "$LOG" in *"done after"*) break ;; esac
    if [ $(( $(date +%s) - START )) -gt "$LIMIT" ]; then echo "WAIT LIMIT"; break; fi
    sleep 20
done
mkdir -p "$OUT"
rsync -rlt -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" "root@$HOST:/workspace/results/" "$OUT/" || echo "RSYNC FAILED"
rsync -rlt -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" "root@$HOST:/workspace/boot_watchdog.log" "$OUT/" 2>/dev/null
$SSH "cd /workspace/results && find . -type f -exec sha256sum {} + | sort -k 2" > "$OUT/remote_sha.txt" 2>&1
( cd "$OUT/" && find . -type f ! -name remote_sha.txt ! -name local_sha.txt ! -name boot_watchdog.log -exec shasum -a 256 {} + | sort -k 2 ) > "$OUT/local_sha.txt"
if diff -q "$OUT/remote_sha.txt" "$OUT/local_sha.txt" > /dev/null; then echo "CHECKSUMS MATCH"; else echo "CHECKSUMS DIFFER"; fi
tail -4 "$OUT/e65/session.log"
