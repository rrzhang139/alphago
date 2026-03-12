# Experiment: num_simulations Sweep

## Hypothesis

More MCTS simulations per move should produce stronger play, but with diminishing returns. At 1 sim, MCTS adds almost nothing over the raw network policy. At some point (the "knee"), additional simulations stop helping for a game as simple as tic-tac-toe.

## Setup

- **Game**: tic-tac-toe
- **Swept**: `num_simulations` = [1, 5, 10, 25, 50, 100]
- **Fixed**: 4x128 MLP, lr=0.001, 15 iterations, 50 games/iter, 30 arena games
- **Baseline**: default config (25 sims)

## Results

| Sims | Final vs Random | Best vs Random | Final Loss | Final H(pi) | Wall Time |
|------|----------------|----------------|------------|-------------|-----------|
| 1    | 61%            | 74%            | 0.530      | 0.00        | 42s       |
| 5    | 90%            | 96%            | 1.911      | 0.79        | 43s       |
| 10   | 95%            | 98%            | 1.680      | 0.81        | 48s       |
| 25   | 95%            | 99%            | 1.589      | 0.89        | 60s       |
| 50   | 99%            | 99%            | 1.240      | 0.85        | 71s       |
| 100  | 97%            | 100%           | 1.235      | 0.79        | 96s       |

See `figures/num_sims_comparison.png` for plots.

## Analysis

### 1 sim is catastrophically bad
With only 1 simulation, MCTS can't search at all. The policy is just the raw network prior (no visit count distribution — there's only 1 visit). H(pi) = 0.00 confirms this: every search produces a one-hot policy (the single sim visits exactly one child). The model never accepted a single new model in 15 iterations (0/15 accepted). It plateaus at ~61% vs random — barely better than chance.

**Key insight**: At 1 sim, the MCTS "policy improvement operator" doesn't improve anything. The network trains on its own raw output, which is circular. This is why search matters.

### The jump from 1→5 sims is massive
Going from 1 to 5 simulations jumps from 61% to 90% vs random — the single biggest improvement in the sweep. 5 sims is enough for MCTS to do minimal lookahead: visit a few children, backprop values, form a non-trivial visit distribution. The policy target is now meaningfully better than the raw network, so the training loop actually works.

### 10-25 sims: the practical sweet spot for tic-tac-toe
At 10 sims, the model hits 95% vs random. Going to 25 doesn't improve the win rate much (also 95% final, 99% best) but loss continues dropping (1.680 → 1.589), meaning the network's predictions are getting more accurate even if play strength has saturated.

### 50-100 sims: diminishing returns
50 and 100 sims achieve similar final loss (~1.24) and win rate (~99%). The 100-sim run reaches the best possible performance (100% at one point) but costs 96s vs 71s for 50 sims — 35% more time for negligible improvement. For tic-tac-toe, search beyond 50 sims is wasted compute.

### Loss behaves unexpectedly
The 1-sim run has the *lowest* loss (0.530) despite being the weakest player. This is because with 1 sim, the policy targets are one-hot (H=0) and easy to fit — the network just memorizes "always play this move from this state." The targets are garbage, but they're easy to match. Higher-sim runs have softer, more informative policy targets that are harder to fit (higher loss) but produce much stronger play. **Low loss ≠ good model.** The quality of targets matters more than how well you fit them.

### Search depth increases with simulations
- 1 sim: depth 1.0 (trivially, can only visit one child)
- 5 sims: depth ~1.8
- 25 sims: depth ~3.8
- 100 sims: depth ~4.7

More simulations let MCTS build a deeper tree, looking further ahead. For tic-tac-toe (max 9 moves), depth 4-5 covers most of the game tree.

## Next Steps

1. **c_puct sweep**: Now that we know 25 sims is sufficient for tic-tac-toe, vary c_puct to see how exploration/exploitation balance affects convergence speed.
2. **No-noise experiment**: Compare dirichlet_epsilon=0.25 vs 0.0 — does root noise matter for this simple game?
3. **Network capacity sweep**: Try 2x32 vs 4x128 — how small can the network be before tic-tac-toe performance degrades?
