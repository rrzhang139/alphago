#!/bin/bash
# Run fix experiments C and D sequentially, auto-push after each.
set -e
export PATH="/usr/games:$HOME/.local/bin:$PATH"
cd /workspace/code/alphago
source .venv/bin/activate

echo "=== Fix C: Constant LR 0.001, 2 epochs, FIFO 200K ==="
echo "Start: $(date)"
python -u experiments/20260313_go9_fix_c/run.py 2>&1 | tee /workspace/fix_c.log
echo "=== Fix C complete: $(date) ==="

git add -f experiments/20260313_go9_fix_c/
git commit -m "Fix C results: constant LR + 2 epochs on FIFO 200K" || true
git push || true

echo ""
echo "=== Fix D: Constant LR 0.001, 5 epochs, window 10 ==="
echo "Start: $(date)"
python -u experiments/20260313_go9_fix_d/run.py 2>&1 | tee /workspace/fix_d.log
echo "=== Fix D complete: $(date) ==="

git add -f experiments/20260313_go9_fix_d/
git commit -m "Fix D results: constant LR + window buffer (10 iters)" || true
git push || true

echo ""
echo "=== Both fixes complete: $(date) ==="
