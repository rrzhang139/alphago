# Infrastructure Progress Log

Pod management, experiment execution, cost tracking, and infra learnings. **Infra agent writes here. Research agent reads to understand experiment status.**

| Date | Action | Details | Cost | Notes |
|------|--------|---------|------|-------|
| 2026-03-13 | **Terminated idle pod** `k2uwshg9u2kipr` (multigame-batch) | RTX A4000, 0% GPU for 16.9h after Fix C/D completed | ~$2.87 wasted idle time | Weights verified locally before termination. Must auto-terminate pods after experiments finish. |
| 2026-03-13 | **Fix C completed** (100/100 iters) | Go9, constant LR, 2 epochs, FIFO 200K. Loss 5.30→3.22. | Pod shared with Fix D | `experiments/20260313_go9_fix_c/`. Plateau iters 17-50 then recovered. |
| 2026-03-13 | **Fix D completed** (100/100 iters) | Go9, constant LR, 5 epochs, window buf 10. Loss 4.97→3.06. | Pod shared with Fix C | `experiments/20260313_go9_fix_d/`. Winner: lower loss, better value (0.26 vs 0.36). |
| 2026-03-13 | **Active pod check** | `noecq0fv7nifkx` (wm-train-mixed) RTX 3090 at 100% GPU — left running (not alphago) | $0.22/hr ongoing | Different project, not our concern. |
| 2026-03-14 | **Provisioned pod** `ulc2912j19325a` (go9-scale500) | RTX A4000 $0.17/hr. Setup: uv, cmake, gnugo, git-lfs, C++ MCTS build. | $0.17/hr ongoing | Previous pod `qyd60cp4ogn6op` was EXITED, created new one. |
| 2026-03-14 | **Fixed torch.load** for PyTorch 2.6 compat | Changed `weights_only=True` → `False` in conv_net.py, simple_net.py, othello_net.py, parallel.py | — | Pod has PyTorch 2.6.0+cu124 vs local 2.4. Committed fix `64299cd`. |
| 2026-03-14 | **FAILED: go9_scale500 on A4000** | 500 games/iter, 200 sims on Xeon E5-2650 v3. Iter 1 took >4h (way too slow). | ~$0.75 wasted | Terminated pod. Xeon E5-2650 is a 2014 CPU — 5-10x slower than modern Ryzen for MCTS. |
| 2026-03-14 | **Provisioned pod** `2hjaizpk34u1oj` (A5000, Threadripper) | RTX A5000 $0.16/hr. AMD Ryzen Threadripper PRO 5955WX — massively faster CPU. | $0.16/hr | Much better host for CPU-bound MCTS. |
| 2026-03-14 | **Tuned config**: 100 games/iter, nn_batch=8, eval_games=0 | 500 games too slow. 100 games = 8s/iter. eval_games=20 added 4min/iter — disabled. | — | Key insight: nn_batch_size=64 caused batching overhead. nn_batch=8 is optimal. |
| 2026-03-14 | **Launched go9_scale500 v2** | 500 iters, 100 games/iter, 200 sims, eval=0. Loss 3.5→1.7 in 18 iters (~5min). GPU 97%. | est ~$0.56 (3.5h) | Auto-push on completion. Healthy training: loss ↓, entropy ↓, depth ↑. |

## Infra Learnings

- **RunPod GraphQL field names**: Use `gpuUtilPercent` / `memoryUtilPercent` (not `gpuUtilPerc` / `memoryUtilPerc`)
- **Idle pod detection**: Query RunPod API for `gpuUtilPercent == 0` to find pods that should be terminated
- **Pod termination query**: `mutation { podTerminate(input: { podId: "..." }) }`
- **Weights are ~14MB** for 128f CNN on Go 9x9 — well under GitHub's 100MB limit, no LFS needed
- **PyTorch 2.6 breaking change**: Default `weights_only=True` in `torch.load()` breaks loading old checkpoints. Must use `weights_only=False`.
- **GIT_LFS_SKIP_SMUDGE=1**: Useful for fast clone but means .pt files are LFS pointers. Must run `git lfs install && git lfs pull` after clone.
- **SSH via RunPod gateway**: Use `ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io` — get podHostId from GraphQL `machine { podHostId }`, NOT from the direct IP:port.
- **Cloud vCPU slowness**: 500 games × 200 sims with 10 workers takes ~4h/iter on A4000 pod (Xeon E5-2650). Same config takes ~3 min on A5000 pod (Threadripper). **CPU host matters enormously for MCTS.**
- **nn_batch_size=64 is too large**: Causes batching coordination overhead between workers. nn_batch_size=8 is optimal — 10x faster iterations.
- **eval_games is expensive**: 20 eval games add ~4 min per iteration. Set eval_games=0 during training, eval separately after.
- **Pod CPU matters more than GPU for AlphaZero**: Self-play is CPU-bound (MCTS tree ops). GPU is used for NN inference but is not the bottleneck. Choose pods by CPU, not GPU.
