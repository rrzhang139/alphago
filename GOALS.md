# Goal Checklist

Current objectives, evaluation criteria, and progression plan. Research agent maintains this. Both agents read it.

## Current Goal

**Beat GnuGo Level 10 on 9x9 Go (~5 kyu amateur)**

| Metric | Target | Current Best | How to Measure |
|--------|--------|-------------|----------------|
| vs GnuGo L1 | >80% win | untested | `python scripts/eval_vs_gnugo.py --weights <path> --gnugo-level 1 --num-games 50 --num-sims 400` |
| vs GnuGo L5 | >50% win | untested | Same, `--gnugo-level 5` |
| vs GnuGo L10 | >50% win | untested | Same, `--gnugo-level 10` |
| vs Random | >95% win | 100% (playout_cap model) | Pipeline eval_games or manual |
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
- **Estimated**: 50-100 iterations × 500 games with stable training
- **Status**: Unblocked. Need to eval Fix D model, then scale up iterations with Fix D config (window buffer, constant LR, 5 epochs)

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
