#!/bin/bash
# The A30 session at IAG: test, then the full measurement, then the 2k image, all logged.
# Run it inside tmux (or with nohup) so it survives the SSH connection dropping:
#   tmux new -s a30 'bash run_a30.sh'        or        nohup bash run_a30.sh > a30.out 2>&1 &
# REPEATS=1 bash run_a30.sh shortens the measurement; SKIP_2K=1 skips the 2k image.
cd "$(dirname "$0")"
LOG=runs/a30_$(date +%Y-%m-%d_%H-%M-%S).log
mkdir -p runs
REPEATS=${REPEATS:-3}

step() {
    echo "=== $(date '+%F %T') $*" | tee -a "$LOG"
}

step "the machine"
{ hostname; nvidia-smi; df -h .; command -v nvcc && nvcc --version | tail -2; } 2>&1 | tee -a "$LOG"

step "test"
python main.py test --yes 2>&1 | tee -a "$LOG"
if ! grep -q "all .* tests passed" "$LOG"; then
    step "test FAILED: stopping before the measurement (read $LOG)"
    exit 1
fi

step "measure (repeats $REPEATS)"
python main.py measure --repeats "$REPEATS" --yes 2>&1 | tee -a "$LOG"

if [ -z "$SKIP_2K" ]; then
    step "make 2k"
    python main.py make --size 2048 --snr 10 --observers 1.5708,1.0472,0.5236,0 --yes 2>&1 | tee -a "$LOG"
fi

step "done; bring back these folders:"
{ ls -dt runs/measure/*/ | head -1; [ -z "$SKIP_2K" ] && ls -dt runs/make/*/ | head -1; echo "$LOG"; } 2>/dev/null | tee -a "$LOG"
