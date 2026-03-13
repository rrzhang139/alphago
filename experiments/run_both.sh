#!/bin/bash
# Run both Go 9x9 experiments sequentially, auto-push results after each.
set -e
export PATH="/usr/games:$HOME/.local/bin:$PATH"
cd /workspace/code/alphago
source .venv/bin/activate

echo "=== Starting experiment A: From scratch (LR 0.002) ==="
echo "Start time: $(date)"
python -u experiments/20260312_go9_scratch/run.py 2>&1 | tee /workspace/scratch.log
echo "=== Experiment A complete: $(date) ==="

# Push A results
git add -f experiments/20260312_go9_scratch/
git commit -m "Experiment A: Go 9x9 from scratch (50 iters, LR 0.002)" || true
git push || true

echo ""
echo "=== Starting experiment B: Warm-start low LR (0.0005) ==="
echo "Start time: $(date)"
python -u experiments/20260312_go9_warmstart_lowlr/run.py 2>&1 | tee /workspace/warmstart.log
echo "=== Experiment B complete: $(date) ==="

# Push B results
git add -f experiments/20260312_go9_warmstart_lowlr/
git commit -m "Experiment B: Go 9x9 warm-start low LR (50 iters, LR 0.0005)" || true
git push || true

echo ""
echo "=== Both experiments complete: $(date) ==="
echo "Check /workspace/scratch.log and /workspace/warmstart.log for details"
