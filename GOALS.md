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
| Training loss | monotonically decreasing | U-shaped (broken, fixing) | history.json from training run |
| Policy entropy H(pi) | <1.5 (focused search) | ~2.0-2.3 | C++ MCTS diagnostics |
| Search depth | >5.0 | 2.5-3.0 | C++ MCTS diagnostics |

### Blocking Issues
- Training loss U-shapes after ~30 iterations (cosine LR + stale buffer)
- Fix C (constant LR, 2 epochs) and Fix D (window buffer) running now
- Need stable training before scaling up iterations

## Progression Plan

Each goal builds on the previous. Don't skip ahead — validate each before moving on.

### Phase 1: Stable Training (current)
- **Goal**: Loss monotonically decreasing for 100+ iterations on Go 9x9
- **Eval**: Training curves (loss, entropy, depth) from history.json
- **Status**: In progress — Fix C and D experiments running

### Phase 2: Beat GnuGo Level 1 (~15 kyu)
- **Goal**: >80% win rate vs GnuGo level 1
- **Eval**: `eval_vs_gnugo.py --gnugo-level 1 --num-games 50`
- **Estimated**: 50-100 iterations × 500 games with stable training
- **Status**: Blocked on Phase 1

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
