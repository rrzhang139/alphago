# Goal Checklist

Current objectives, evaluation criteria, and progression plan. Research agent maintains this.

## Current Goal

**Beat GnuGo Level 10 on 9x9 Go (~5 kyu amateur)**

| Metric | Target | Current Best | How to Measure |
|--------|--------|-------------|----------------|
| vs GnuGo L1 | >80% win | 0% (all models so far) | `python scripts/eval_vs_gnugo.py --weights <path> --gnugo-level 1 --num-games 50 --num-sims 400 --use-se --global-pool-value --num-res-blocks 6` |
| vs GnuGo L5 | >50% win | untested | Same, `--gnugo-level 5` |
| vs GnuGo L10 | >50% win | untested | Same, `--gnugo-level 10` |
| Training loss | <1.0 | **0.960** (se_10ep_rom, iter 286) | history.json |
| Search depth | >8.0 | **9.1** (se_10ep_rom, iter 286) | C++ MCTS diagnostics |

---

## Bottleneck-First Research Strategy

**Philosophy**: Instead of tweaking dozens of small parameters, identify the ONE biggest bottleneck at each stage and fix it. KataGo achieved 50x efficiency over comparable methods by stacking ~6 high-impact improvements (each 1.25x-1.65x). We should focus on the same.

### KataGo's Improvements Ranked by Impact (from the paper)

| Rank | Technique | Speedup | Our Status | Notes |
|------|-----------|---------|------------|-------|
| 1 | **Auxiliary ownership/score targets** | 1.65x | ✅ se_ownership running | Biggest single win |
| 2 | **Global pooling** | 1.60x | ✅ Implemented | In all SE+GP experiments |
| 3 | **Game-specific input features** (liberties, ladder, pass-alive) | 1.55x | ⚠️ Liberty planes implemented, **ladders & pass-alive missing** | High priority gap |
| 4 | **Playout cap randomization** | 1.37x | ✅ Implemented | In all GPU experiments |
| 5 | **Auxiliary policy targets** (opponent's next move) | 1.30x | ❌ Not implemented | Medium priority |
| 6 | **Policy target pruning + forced playouts** | 1.25x | ✅ Pruning done | Forced playouts not done |

**Combined theoretical speedup**: 9.1x. We have ~4 of 6. The remaining ~2.5x comes from game-specific features and auxiliary policy targets.

### Reference Implementation Comparison

| System | Network | Games | GPU-hours | Strength |
|--------|---------|-------|-----------|----------|
| **michaelnny/alpha_zero** | 10 blocks, 128f | 1M+ | 320 (8×3090×40h) | Amateur 1-dan (beat CrazyStone 16/20) |
| **ELF OpenGo** | 20 blocks, 256f | 20M | 48,000 (2000×V100) | Superhuman (20-0 vs top pros) |
| **KataGo** | 6-20 blocks, 128-256f | ~5M | ~10,000 | Superhuman (50x more efficient than ELF) |
| **Us (se_10ep_rom)** | 6 blocks, 128f | ~50K (500×100) | ~4 (1×A4000×8h) | Loss 0.96, 0/20 vs GnuGo L1 |

**The gap is clear**: we've generated ~50K games. The reference needed 1M+. We need **20x more training data/games**, not parameter tweaks.

---

## What Actually Matters (ordered by impact)

### Tier 1: Training Scale (10x+ impact) 🔴

These are the "forest" — each one is worth more than all parameter tweaks combined.

**1. More self-play games** — We have ~50K games. Reference used 1M+. This is probably our #1 bottleneck.
- Action: Run 2000+ iterations (not 500), or increase games_per_iteration to 200-400
- Cost: ~$5-10 for 2000 iters on A4000
- Expected impact: Massive — more diverse positions = better generalization

**2. Network capacity** — Our 6-block 128-filter network has ~1.9M params. Reference used 10 blocks (3.1M params). ELF used 20 blocks 256f (~25M params). Diminishing returns exist, but we may be undersized.
- Action: Test 10 blocks 128f or 6 blocks 256f (double filters = 4x params for conv layers)
- Cost: ~$3-5 for a 500-iter experiment
- Expected impact: High — "doubling rollouts still boosts strength by ~200 ELO" (ELF finding, indicating model capacity is a bottleneck)

**3. More simulations** — We use 200 sims. AlphaZero used 800. KataGo used 600-800 for training.
- Action: Test 400 or 600 sims (with playout cap to manage cost)
- Cost: ~2x per iteration
- Expected impact: High — deeper search = higher quality training data

### Tier 2: High-Impact Features (1.3-1.65x each) 🟡

These are the KataGo improvements we haven't done yet.

**4. Game-specific input features** — Ladders, pass-alive detection, ko threat count
- Ladder detection alone is a huge deal: ELF found ladders were "learned slowly and never fully mastered" even at superhuman level
- Pass-alive regions help the model understand which groups are unconditionally alive
- Expected: 1.55x (KataGo measured)

**5. Auxiliary policy targets** — Predict opponent's next move as an extra training signal
- Provides regularization and forces the network to model both sides
- Expected: 1.30x (KataGo measured)

**6. Forced playouts** — Force MCTS to visit top-N policy moves at least once
- Ensures search explores reasonable moves even when priors are wrong
- Expected: Combined with pruning ~1.25x

### Tier 3: Parameter Tuning (1-10% each) 🟢

These are what we've been spending most time on. Each gives 5-15% at best.

- FPU reduction, c_puct_base, gradient clipping, temperature schedule, etc.
- **Lesson learned**: Parameters interact unpredictably. Individual A/B tests can mislead (c_puct_base + FPU conflict).
- **Rule**: Only tune parameters AFTER Tier 1 and Tier 2 are addressed. Tuning a weak model is polishing a turd.

---

## Progression Plan (Revised)

### Phase 1: Stable Training ✅
- Loss monotonically decreasing for 100+ iterations
- **Done** — Fix D showed stable training, se_10ep_rom confirmed at 500 iters

### Phase 2: Beat GnuGo Level 1 (~15 kyu) — CURRENT
- **Goal**: >80% win rate vs GnuGo level 1
- **Eval**: `eval_vs_gnugo.py --gnugo-level 1 --num-games 50`

#### Current GPU Experiments
| Experiment | Iter | Loss | Depth | Status |
|-----------|------|------|-------|--------|
| **se_10ep_rom** | 286/500 | **0.960** | 9.1 | 🔥 Phase transition — broke below 1.0 |
| **se_ownership** | 266/500 | 1.125 | 8.2 | Catching up, no phase transition yet |

#### When These Complete: Decision Tree
1. **Eval both vs GnuGo L1** with `eval_sweep.py`
2. **If >50% vs GnuGo L1** → Move to Phase 3. Current approach works, just needs more scale.
3. **If 0% vs GnuGo L1** → The bottleneck is NOT loss/training quality. It's one of:
   - **Scale** (50K games isn't enough) → Run 2000-iter experiment
   - **Network capacity** (6 blocks too small) → Test 10 blocks
   - **Game-specific features** (no ladder/pass-alive understanding) → Implement ladders
4. **If 10-40% vs GnuGo L1** → Promising but needs refinement. Try:
   - More sims at eval time (800-1600)
   - ROM-fixed follow-up experiment
   - Ownership might help (if se_ownership does better than se_10ep_rom vs GnuGo)

#### Next Experiment Priority (after current runs finish)
1. **Scale test**: 2000 iters, same config as se_10ep_rom (highest priority if 0% vs GnuGo)
2. **ROM-fixed**: 200 iters warm-start (queued, tests if ROM helps at GPU scale)
3. **Larger network**: 10 blocks 128f or 6 blocks 256f (if scale alone doesn't help)
4. **Ladder detection**: Input feature that directly addresses tactical blindness

### Phase 3: Beat GnuGo Level 5 (~10 kyu)
- **Goal**: >50% win rate vs GnuGo level 5
- **Likely needs**: 1000+ iterations, possibly larger network, ownership
- **Status**: Not started

### Phase 4: Beat GnuGo Level 10 (~5 kyu)
- **Goal**: >50% win rate vs GnuGo level 10
- **Likely needs**: Larger network (10+ blocks), 2000+ iterations, game-specific features
- **Status**: Not started

### Phase 5: Go 19x19 (stretch)
- **Goal**: Functional training on full-size Go board
- **Needs**: Multi-GPU, 20+ blocks, months of training
- **Status**: Future

---

## Key Learnings

### What Actually Moved the Needle (ordered by impact)
1. **10 epochs vs 5 epochs**: 47% better loss. Biggest single finding.
2. **SE blocks + global pool**: Broke through CNN plateau (1.6→1.4). Architecture > hyperparameters.
3. **C++ MCTS engine**: 5x faster self-play. Enables more training.
4. **Window buffer**: Keeps fresh data, prevents stale gradient signal.
5. **Playout cap**: 4-5x faster games with equal quality.

### What Didn't Matter Much
- FPU reduction: 12.6% locally, but interacts with other params
- c_puct_base: 12.8% alone, but conflicts with FPU
- Gradient clipping: neutral
- Progressive sims: marginal
- Shaped Dirichlet: hurt when combined with other tricks (kitchen_sink)

### Meta-Learnings
- **Test parameters in combination**, not individually. Interactions can reverse individual results.
- **Local tests (50 sims, 5 iters) can't validate search parameters** — these depend on sim count and training length.
- **Architecture changes > hyperparameter tuning** at our scale.
- **The model plays reasonable Go in self-play** (0 premature passes). Problems only appear vs external opponents it hasn't seen.

---

## GnuGo Strength Reference

| Level | Approx Rank | Notes |
|-------|-------------|-------|
| 1 | ~15 kyu | Very weak, makes obvious mistakes |
| 3 | ~12 kyu | Basic territory sense |
| 5 | ~10 kyu | Decent amateur |
| 8 | ~7 kyu | Strong club player |
| 10 | ~5 kyu | Strong amateur, our target |
