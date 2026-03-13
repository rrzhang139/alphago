# Infrastructure Progress Log

Pod management, experiment execution, cost tracking, and infra learnings. **Infra agent writes here. Research agent reads to understand experiment status.**

| Date | Action | Details | Cost | Notes |
|------|--------|---------|------|-------|
| 2026-03-13 | **Terminated idle pod** `k2uwshg9u2kipr` (multigame-batch) | RTX A4000, 0% GPU for 16.9h after Fix C/D completed | ~$2.87 wasted idle time | Weights verified locally before termination. Must auto-terminate pods after experiments finish. |
| 2026-03-13 | **Fix C completed** (100/100 iters) | Go9, constant LR, 2 epochs, FIFO 200K. Loss 5.30→3.22. | Pod shared with Fix D | `experiments/20260313_go9_fix_c/`. Plateau iters 17-50 then recovered. |
| 2026-03-13 | **Fix D completed** (100/100 iters) | Go9, constant LR, 5 epochs, window buf 10. Loss 4.97→3.06. | Pod shared with Fix C | `experiments/20260313_go9_fix_d/`. Winner: lower loss, better value (0.26 vs 0.36). |
| 2026-03-13 | **Active pod check** | `noecq0fv7nifkx` (wm-train-mixed) RTX 3090 at 100% GPU — left running (not alphago) | $0.22/hr ongoing | Different project, not our concern. |

## Infra Learnings

- **RunPod GraphQL field names**: Use `gpuUtilPercent` / `memoryUtilPercent` (not `gpuUtilPerc` / `memoryUtilPerc`)
- **Idle pod detection**: Query RunPod API for `gpuUtilPercent == 0` to find pods that should be terminated
- **Pod termination query**: `mutation { podTerminate(input: { podId: "..." }) }`
- **Weights are ~14MB** for 128f CNN on Go 9x9 — well under GitHub's 100MB limit, no LFS needed
