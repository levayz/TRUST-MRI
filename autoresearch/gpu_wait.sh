#!/usr/bin/env bash
# Wait until GPU $1 is free for one of our runs, cooperating with gpu_hold.py (which reserves the GPU between runs):
# touch the request file, wait for the holder's acknowledgement (it deletes the file and stays off the GPU for 5 min; the
# request is renewed every 2 min while we wait), then wait until nothing else is on the GPU (polling every 5 s).
# Exit 3 after AR_WAIT_MINUTES (default 720) of waiting.
#   bash gpu_wait.sh 1 || exit 3
GPU=${1:?gpu index}
cd "$(dirname "$0")"
WAIT_MINUTES=${AR_WAIT_MINUTES:-720}
HOLD_DIR=/fast/lev/gpu_hold; mkdir -p "$HOLD_DIR"      # shared by every checkout and worktree: one holder per GPU
RELEASE_FILE="$HOLD_DIR/.gpu_release_${GPU}"
HOLD_PID_FILE="$HOLD_DIR/gpu_hold_${GPU}.pid"

holder_alive() { [ -f "$HOLD_PID_FILE" ] && kill -0 "$(cat "$HOLD_PID_FILE")" 2>/dev/null; }
# memory used on the GPU by everything except our holder (which keeps a block during the hand-over on purpose)
gpu_used_mb() {
    local hp=none; holder_alive && hp=$(cat "$HOLD_PID_FILE")
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU" | awk -F', *' -v hp="$hp" '$1 != hp {s += $2} END {print s + 0}'
}
request_gpu() {
    touch "$RELEASE_FILE"
    holder_alive || return 0
    for _ in $(seq 1 24); do [ -e "$RELEASE_FILE" ] || return 0; sleep 5; done
    echo "$(date '+%F %T') warning: gpu_hold.py did not acknowledge the release request"
}
request_gpu; last_request=$(date +%s); polls=0
while [ "$(gpu_used_mb)" -gt 2000 ]; do
    if [ "$polls" -ge $((WAIT_MINUTES * 12)) ]; then echo "GPU $GPU still busy after $WAIT_MINUTES min, giving up"; rm -f "$RELEASE_FILE"; exit 3; fi
    [ $((polls % 120)) -eq 0 ] && echo "$(date '+%F %T') waiting: GPU $GPU has $(gpu_used_mb) MiB in use"
    sleep 5; polls=$((polls + 1))
    if [ $(( $(date +%s) - last_request )) -ge 120 ]; then request_gpu; last_request=$(date +%s); fi
done
rm -f "$RELEASE_FILE"
exit 0
