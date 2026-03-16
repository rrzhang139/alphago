# Goal Checklist

Current objectives, evaluation criteria, and progression plan. Research agent maintains this. Both agents read it.

## Current Goal

**Beat GnuGo Level 10 on 9x9 Go (~5 kyu amateur)**

| Metric | Target | Current Best | How to Measure |
|--------|--------|-------------|----------------|
| vs GnuGo L1 | >80% win | 0% (Fix D, 100 iters, 200-400 sims) | `python scripts/eval_vs_gnugo.py --weights <path> --gnugo-level 1 --num-games 50 --num-sims 400` |
| vs GnuGo L5 | >50% win | untested | Same, `--gnugo-level 5` |
| vs GnuGo L10 | >50% win | untested | Same, `--gnugo-level 10` |
| vs Random | >95% win | 65% (Fix D, 100 iters, 200 sims) | Pipeline eval_games or manual |
| Training loss | monotonically decreasing | 3.06 final, hump resolved (Fix D) | history.json from training run |
| Policy entropy H(pi) | <1.5 (focused search) | ~2.2 (Fix D, 100 iters) | C++ MCTS diagnostics |
| Search depth | >5.0 | ~3.0 (Fix D, 100 iters) | C++ MCTS diagnostics |

### Blocking Issues (RESOLVED)
- ~~Training loss U-shapes after ~30 iterations (cosine LR + stale buffer)~~
- Fix C and Fix D both completed 100 iters (2026-03-13). Both show a mid-training loss hump (iters 15-40) but **recover and continue decreasing**. This hump is natural: as the model improves, self-play generates harder data (P1 win rate rises 14%→40%, games lengthen), temporarily raising loss before the model adapts.
- **Fix D (window buffer) is the winner**: final loss 3.06 vs 3.22 (Fix C). Value loss 0.26 vs 0.36. Window buffer keeps fresh data.
- **Next blocker**: no vs_random eval was run (eval_games=0). Need to eval the Fix D model to measure actual play strength before scaling further.

### Learnings
- The loss "U-shape" is not a bug — it's a natural phase transition in self-play training where improving models produce harder training data
- Window buffer (last 10 iters) outperforms FIFO 200K for Go 9x9 training
- Constant LR 0.001 is stable for 100 iterations (no cosine decay needed at this stage)
- 5 epochs on window buffer > 2 epochs on FIFO (more thorough training on fresh data)

## Progression Plan

Each goal builds on the previous. Don't skip ahead — validate each before moving on.

### Phase 1: Stable Training ✅
- **Goal**: Loss monotonically decreasing for 100+ iterations on Go 9x9
- **Eval**: Training curves (loss, entropy, depth) from history.json
- **Status**: Complete — Fix D (window buffer) shows loss 4.97→3.06 over 100 iters. Mid-training hump is natural (resolves by iter 50). Fix D model is current best.
- **Next**: Eval Fix D model vs random, then proceed to Phase 2

### Phase 2: Beat GnuGo Level 1 (~15 kyu) (current)
- **Goal**: >80% win rate vs GnuGo level 1
- **Eval**: `eval_vs_gnugo.py --gnugo-level 1 --num-games 50`
- **Estimated**: 500+ iterations with SE architecture
- **Status**: All 3 initial GPU experiments completed. 0/20 vs GnuGo L1 for all models. Model plays coherent Go (beats random easily) but lacks tactical depth.

#### GPU Experiment Results (March 15)
| Experiment | Iters | Architecture | Best Loss | vsGnuGo L1 | Cost | Key Finding |
|-----------|-------|-------------|-----------|------------|------|-------------|
| scale500 | 500 | CNN 4b 128f | 1.605 (iter 32) | 0/20 | $0.58 | **Plateaued at iter ~25**. 475 wasted iterations. |
| fresh_correct | 300 | CNN 4b 128f | 1.639 (iter 250) | 0/20 | $0.50 | Plateau ~1.65-1.80 after iter 86. |
| se_globalpool | 200 | **CNN 6b SE+GP** | **1.389** (iter 137) | 0/20 | $0.54 | **BEST. Still improving at iter 176 (1.416).** |
| kitchen_sink | running | CNN 4b 128f GP | 1.80 (plateau iter 18-56+) | - | ongoing | PSW+VLW+shaped_dirichlet may be too aggressive |

#### Critical Findings
1. **Plain CNN architecture hits a wall at loss ~1.6-1.7** — scale500 plateaued at iter 25, never improved in 475 more iters
2. **SE blocks + global pool break through that wall** — reached 1.389 (15% better) in only 200 iters, still improving
3. **More sims don't help**: 800 sims still 0/20 vs GnuGo L1 — network quality is the bottleneck
4. **Model does play Go**: beats random easily, builds territory, reaches rootV=1.0 by move 23. But no tactical reading
5. **Reference impl** (michaelnny/alpha_zero) used 10 res blocks, 150K gradient steps, 1M+ games to reach amateur 1-dan

#### Breakthrough Local Findings (March 16)
- **10 epochs >> 5 epochs**: loss 1.337 vs 2.537 in 5 iters (**47% better!**). All GPU experiments used 5 epochs — we've been under-training.
- **Random opening moves (ROM=6)**: loss 2.393 vs 2.532 (**5.5% better**), more diverse search (H=1.32 vs 0.51), 20% faster
- **10 res blocks marginal**: only 2.4% better loss but 43% slower. Not worth the cost.

#### Current Priorities (ordered)
1. **se_best** (RUNNING on pod): 500 iters SE+GP warm-start, 5 epochs (may be under-trained)
2. **se_10ep_rom** (QUEUED, highest priority): SE+GP + **10 epochs** + ROM=6, from scratch, 500 iters. Should significantly outperform se_best.
3. liberty_planes: after se_10ep_rom
4. ownership: lowest priority

### Phase 3: Beat GnuGo Level 5 (~10 kyu)
- **Goal**: >50% win rate vs GnuGo level 5
- **Eval**: `eval_vs_gnugo.py --gnugo-level 5 --num-games 50`
- **Estimated**: 200+ iterations, may need larger network or more sims
- **Status**: Not started

### Phase 4: Beat GnuGo Level 10 (~5 kyu)
- **Goal**: >50% win rate vs GnuGo level 10
- **Eval**: `eval_vs_gnugo.py --gnugo-level 10 --num-games 50`
- **Estimated**: 500+ iterations or architectural improvements
- **Status**: Not started

### Phase 5: Go 19x19 (stretch)
- **Goal**: Functional training on full-size Go board
- **Eval**: vs GnuGo on 19x19, training stability
- **Estimated**: Multi-GPU, larger network, weeks of training
- **Status**: Future

## GnuGo Strength Reference

| Level | Approx Rank | Notes |
|-------|-------------|-------|
| 1 | ~15 kyu | Very weak, makes obvious mistakes |
| 3 | ~12 kyu | Basic territory sense |
| 5 | ~10 kyu | Decent amateur |
| 8 | ~7 kyu | Strong club player |
| 10 | ~5 kyu | Strong amateur, our target |
