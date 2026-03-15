# GPU Allocation Reference

Measured resource usage for AlphaZero experiments on RunPod. Use this to pick the cheapest pod that fits each workload.

## Go 9x9 — 128f 4-block CNN (standard)

**Workload**: 100 games/iter, 200 sims, 10 workers, nn_batch=8, C++ MCTS engine

| Resource | Measured | Notes |
|----------|----------|-------|
| **GPU Util** | 20-25% (self-play), 97% (training) | GPU is idle during CPU-bound MCTS, spikes during NN training |
| **VRAM** | ~930 MiB / 24 GiB (4%) | 128f CNN is tiny. **16GB GPU is more than enough.** A4000 (16GB) works fine. |
| **CPU** | 10 workers, each ~1 core | MCTS is the bottleneck. **10+ cores required.** Single-threaded perf matters more than core count. |
| **RAM** | ~2 GiB process RSS | Window buffer adds ~1GB. **4GB RAM minimum**, 8GB comfortable. |
| **Disk** | ~6.5 GB (code + venv + checkpoints) | 20GB volume is sufficient. 500 iters × 25 checkpoints × 14MB = ~350MB weights. |
| **Iter time** | 25-27s (Threadripper), 35-50s (Xeon E5-2699) | **CPU single-thread perf is the #1 factor** |

### Recommended pod config
```
GPU: RTX A4000 (16GB) — cheapest that works ($0.17/hr)
     RTX A5000 (24GB) — if A4000 unavailable ($0.16/hr, sometimes cheaper)
CPU: AMD Threadripper / Ryzen 7000+ preferred (2x faster than old Xeons)
     Intel Xeon E5-2699 v3 (64 cores) workable but 2x slower per-iter
     Intel Xeon E5-2650 v3 — AVOID. Way too slow for MCTS.
RAM: 8GB+ (only ~2GB used)
Disk: 20GB volume sufficient
```

## Go 9x9 — 128f 6-block CNN + SE blocks + global pool

**Workload**: Same as above but deeper network with SE attention

| Resource | Measured | Notes |
|----------|----------|-------|
| **GPU Util** | 20-72% | Higher than standard due to SE blocks adding compute |
| **VRAM** | ~930 MiB | Slightly more params (1.88M vs ~1.2M) but same VRAM class |
| **CPU** | Same as standard | MCTS still dominates |
| **RAM** | ~2 GiB | Same |
| **Iter time** | 45-66s (Xeon E5-2699), grows with buffer | ~40-60% slower than standard CNN. Window buffer slows training as it fills. |

### Recommended: Same as standard. SE blocks don't need more GPU.

## Key Insights

### CPU matters more than GPU for AlphaZero
Self-play uses MCTS (tree search on CPU). GPU is only used for NN inference during search.
With nn_batch_size=8, GPU batching is efficient but GPU sits idle 75-80% of the time.

| CPU | Single-thread perf | Iter time (100 games, 200 sims) | Verdict |
|-----|-------------------|--------------------------------|---------|
| AMD Threadripper PRO 5955WX | Fast | 25-27s | **Best value** |
| Intel Xeon E5-2699 v3 (64 cores) | Slow (2014) | 35-52s | Workable |
| Intel Xeon E5-2650 v3 (20 cores) | Very slow (2014) | >4h (500 games) | **AVOID** |

### GPU VRAM requirements by model size
| Model | Params | VRAM Used | Min GPU |
|-------|--------|-----------|---------|
| 128f 4-block CNN (Go 9x9) | ~1.2M | ~930 MiB | RTX A4000 (16GB) |
| 128f 6-block CNN + SE (Go 9x9) | 1.88M | ~930 MiB | RTX A4000 (16GB) |
| 512f OthelloNNet (10x10) | ~85M | ~2 GiB | RTX A4000 (16GB) |
| Projected: 256f 20-block (Go 19x19) | ~25M | ~4-8 GiB | RTX 3090 (24GB) |

### nn_batch_size tuning
| Batch size | Iter time | GPU Util | Notes |
|-----------|-----------|----------|-------|
| 1 | Very slow | <5% | Transfer overhead dominates |
| 8 | **Optimal** | 20-25% | Sweet spot for C++ MCTS |
| 64 | 10x slower | Variable | Coordination overhead between workers |

### Cost optimization
- **Never provision more GPU than needed**: 128f CNN uses <1GB VRAM. An A100 ($1.64/hr) would waste 79GB.
- **Pick pods by CPU, not GPU**: Sort RunPod by CPU model. Threadripper/Ryzen >> old Xeons.
- **eval_games=0 during training**: Saves ~4 min/iter. Evaluate checkpoints separately after training.
- **Terminate pods immediately**: Idle A4000 costs $2.87 if left overnight (learned the hard way).

## Historical Experiment Costs

| Experiment | GPU | CPU | Iters | Time | Cost | Loss |
|-----------|-----|-----|-------|------|------|------|
| Fix C (100 iter) | A4000 | shared | 100 | shared | shared | 3.22 |
| Fix D (100 iter) | A4000 | shared | 100 | shared | shared | 3.06 |
| scale500 (FAILED) | A4000 | Xeon E5-2650 | 1/500 | 4h+ | $0.75 wasted | — |
| scale500 v2 | A5000 | Threadripper | 500 | 3.6h | $0.58 | 1.60 |
| fresh_correct | A5000 | Xeon E5-2699 | 300 | 3.1h | $0.50 | 1.64 |
| se_globalpool | A5000 | Xeon E5-2699 | 200 | 3.4h | $0.54 | **1.389** |
