# Experiment: c_puct Sweep

## Hypothesis

`c_puct` controls the exploration/exploitation balance in PUCT selection:

```
PUCT(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
                 ───────────────────────────────────────────
                        exploration bonus (scaled by c_puct)
```

**Prediction:**
- **c_puct too low (0.1):** Search commits early to the best-looking move, producing narrow deep trees. The MCTS policy will be sharply peaked (low entropy), converging to near one-hot targets quickly. Fast loss reduction, but the network might lock in suboptimal patterns.
- **c_puct too high (5.0):** Search spreads budget across many moves, producing broad shallow trees. The MCTS policy stays diffuse (high entropy) — the network trains on soft targets that barely improve on its own prior. Loss stays high because the targets are noisy.
- **Sweet spot (~1.0):** Enough exploration to correct network mistakes without wasting simulations.

For tic-tac-toe (9 actions, simple game), the sweet spot may be broad since there just aren't many moves to waste sims on.

## Setup

- **Baseline**: `c_puct = 1.0` (default)
- **Sweep**: `c_puct ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`
- **Fixed parameters**: 25 sims, 15 iters × 50 games, 128×4 MLP, lr=0.001, seed=42

### Config diff from defaults (shared across all runs)
| Parameter | Default | Experiment |
|-----------|---------|------------|
| `training.num_iterations` | 25 | 15 |
| `training.games_per_iteration` | 100 | 50 |
| `arena.arena_games` | 40 | 30 |

## Results

| c_puct | Final vs Random | Best vs Random | Final Loss | Final H(π) | Final Depth | Time |
|--------|-----------------|----------------|------------|------------|-------------|------|
| **0.1** | 96% | 98% | **1.284** | **0.53** | **4.6** | 56s |
| **0.5** | 94% | 98% | 1.303 | 0.65 | 4.7 | 56s |
| **1.0** | 95% | 97% | 1.451 | 0.87 | 4.0 | 58s |
| **2.0** | 97% | **99%** | 1.799 | 1.27 | 3.1 | 57s |
| **5.0** | 97% | **99%** | 1.969 | 1.36 | 3.2 | 58s |

### Key observations

**1. Win rate is essentially flat across the entire range.** All values achieve 94–99% vs random. The game is too simple for c_puct to differentiate play strength — tic-tac-toe only has ~4,520 distinct game states and perfect play results in a draw.

**2. Loss tells a different story.** Low c_puct produces dramatically lower loss:
- c_puct=0.1: final loss **1.284** (policy loss 0.960)
- c_puct=5.0: final loss **1.969** (policy loss 1.478)

This is a 54% gap. The reason: low c_puct concentrates visits on 1-2 moves, producing sharp (near one-hot) training targets. The network can easily match sharp targets → low cross-entropy loss. High c_puct spreads visits, producing soft targets → higher cross-entropy even when the network is doing well.

**3. Policy entropy confirms the mechanism.**
- c_puct=0.1: H(π) = **0.53** nats (very focused search)
- c_puct=5.0: H(π) = **1.36** nats (diffuse search, close to uniform over ~4 moves)

The MCTS policy at low c_puct has exp(0.53) ≈ 1.7 "effective moves" — essentially picking one move with high confidence. At c_puct=5.0, exp(1.36) ≈ 3.9 effective moves — almost uniform across legal actions.

**4. Search depth inversely correlates with c_puct.**
- c_puct=0.1: depth **4.6** (narrow deep trees)
- c_puct=5.0: depth **3.2** (broad shallow trees)

With the same 25-sim budget, low c_puct focuses sims down one line, seeing deeper. High c_puct spreads them wide, seeing broadly but shallowly.

**5. Arena dynamics are revealing.** c_puct=0.1 accepted only **3/15** models — nearly all arena games were draws (0W/30D/0L pattern). The model converged so quickly that new models couldn't beat the old one. c_puct=2.0 and 5.0 also accepted only 2-3 models but via a different mechanism: the new models were too similar to distinguish because both played broad, exploratory MCTS.

See: `figures/cpuct_comparison.png`

## Analysis

### The paradox: low loss ≠ better learning

c_puct=0.1 has the lowest loss but this is **misleading**. Low c_puct makes the MCTS policy nearly deterministic (one-hot), so the cross-entropy target is easy to fit. But these sharp targets may not contain as much information about the value of alternative moves. The network learns "play this specific move" rather than "these 3 moves are all reasonable."

For tic-tac-toe, this doesn't matter — there's typically one clearly best move, and the network learns it regardless of c_puct. But for harder games (Go, chess), this distinction becomes critical: the network needs to learn the landscape of move quality, not just the peak.

### The real effect: training target quality vs diversity

| c_puct | MCTS targets are... | Network learns... |
|--------|--------------------|--------------------|
| 0.1 | Sharp, confident, narrow | One "best" move per position |
| 1.0 | Moderately concentrated | Top 2-3 moves with relative quality |
| 5.0 | Diffuse, exploratory | Many moves are viable (overestimates diversity) |

### Why tic-tac-toe can't differentiate

The game saturates too quickly. All c_puct values reach >93% vs random within 1-2 iterations. The remaining training is just polishing an already-strong player. A harder game would likely show:
- Very low c_puct: converges fast but plateaus early (misses non-obvious moves)
- Very high c_puct: converges slowly, may not converge at all with limited sims
- Medium c_puct: best long-run performance

## Next Steps

1. **Repeat on Connect4** — c_puct should matter more on a harder game with deeper tactics and larger action space (7 vs 9 moves, but much deeper lookahead needed).
2. **c_puct × num_simulations interaction** — low c_puct with many sims goes very deep; high c_puct with few sims goes very shallow. The interaction may be more important than either knob alone.
3. **Adaptive c_puct** — start high (broad exploration when network is bad) and anneal down as training progresses (focused search when network improves). This is used in some KataGo configurations.
4. **Measure policy quality directly** — instead of loss (which conflates target sharpness with prediction accuracy), measure how often the network's top-1 move matches the "true best" move from a high-sim-count reference search.
