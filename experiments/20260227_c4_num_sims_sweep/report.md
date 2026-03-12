# Experiment: num_simulations Sweep — Connect Four

## Hypothesis

Connect Four is harder than tic-tac-toe: 6×7 board, 42 possible cells, games lasting 20-40 moves with tactical traps (double threats, forced wins). We expect:

1. **The "knee" shifts right** — more sims needed to reach strong play.
2. **Low sim counts are weaker** — 1 and 5 sims can't discover multi-move traps.
3. **High sims show bigger gains** — deeper search matters when tactics go 5+ moves deep.
4. **The loss paradox repeats** — 1 sim = lowest loss but weakest play.

Direct comparison with `experiments/20260227_num_sims_sweep/` (tic-tac-toe version).

## Setup

- **Game**: Connect Four (6×7 board, 7 actions, gravity-based)
- **Swept**: `num_simulations` = [1, 5, 10, 25, 50, 100]
- **Fixed**: 4×128 MLP, lr=0.001, 15 iterations, 50 games/iter, 30 arena games
- **Identical to TTT sweep** for direct comparison

## Results

| Sims | Final vs Random | Best vs Random | Final Loss | Final H(π) | Final Depth | Time |
|------|-----------------|----------------|------------|------------|-------------|------|
| 1    | 76%             | 86%            | **0.066**  | 0.00       | 1.0         | 34s  |
| 5    | 80%             | 90%            | 1.990      | 0.82       | 1.7         | 44s  |
| 10   | 94%             | 94%            | 1.953      | 0.95       | 2.6         | 50s  |
| 25   | 94%             | 98%            | 1.799      | 0.88       | 4.2         | 78s  |
| 50   | 94%             | **100%**       | 2.030      | 0.89       | 5.3         | 2.5m |
| 100  | **100%**        | **100%**       | 1.626      | 0.77       | **7.3**     | 4.8m |

See `figures/num_sims_comparison.png` for plots.

## Cross-Game Comparison (TTT vs C4)

| Sims | TTT Final | C4 Final | TTT Best | C4 Best | TTT Depth | C4 Depth |
|------|-----------|----------|----------|---------|-----------|----------|
| 1    | 61%       | 76%      | 74%      | 86%     | 1.0       | 1.0      |
| 5    | 90%       | 80%      | 96%      | 90%     | 1.8       | 1.7      |
| 10   | 95%       | 94%      | 98%      | 94%     | ~2.5      | 2.6      |
| 25   | 95%       | 94%      | 99%      | 98%     | ~3.8      | 4.2      |
| 50   | 99%       | 94%      | 99%      | 100%    | ~4.5      | 5.3      |
| 100  | 97%       | **100%** | 100%     | **100%**| ~4.7      | **7.3**  |

## Analysis

### 1. The knee shifts right — confirmed

For TTT, the knee was at **5 sims** (61% → 90%, the biggest single jump). For C4, the biggest jump is at **10 sims** (80% → 94%), and even that doesn't reach TTT's saturation point. The game's tactical depth demands deeper search.

At 25 sims (the default), TTT already saturates at 95-99%. C4 is still only at 94% and needs 100 sims to reliably hit 100%. **For Connect Four, 25 sims is under-provisioned.**

### 2. 100 sims is not overkill — unlike TTT

In TTT, going from 50 to 100 sims gave almost nothing (99% → 97%, within noise). In C4, **100 sims is the only config that reliably hits 100% vs random.** The search depth at 100 sims (7.3) is dramatically deeper than at 50 (5.3), meaning MCTS is discovering multi-move forced wins that shorter searches miss.

### 3. Connect Four's 1-sim is less broken than TTT's

Paradoxically, 1-sim C4 (76%) beats 1-sim TTT (61%). This is likely because Connect Four has gravity constraints (columns fill up), so a random untrained network has fewer obviously terrible moves to make. In TTT, without search, the network might repeatedly play in already-occupied squares (masked out, but the policy over legal moves is poor).

### 4. The loss paradox is even more extreme

1-sim loss: **0.066** — absurdly low because the training targets are one-hot (H=0.00) and the network memorizes them perfectly. But it only wins 76% vs random.

100-sim loss: **1.626** — much higher because MCTS produces soft, informative targets. But it wins 100% vs random.

The gap is 25× (0.066 vs 1.626), even larger than TTT's gap. **Loss is completely uncorrelated with play strength.**

### 5. Model acceptance rate tells the story

| Sims | Models Accepted | Interpretation |
|------|-----------------|----------------|
| 1    | 0/15            | No model ever passes arena — network can't improve without search |
| 5    | 4/15            | Occasional improvement, but arena results are noisy at low sims |
| 10   | 4/15            | Similar — marginal improvements |
| 25   | 7/15            | Active learning — almost half the models are better |
| 50   | 4/15            | Fewer acceptances but each accepted model is stronger |
| 100  | **9/15**        | Consistent improvement — search is strong enough that better networks reliably produce better play |

The 100-sim run accepted 9/15 models — the learning loop is working most efficiently here. The arena with 100 sims on each side can reliably distinguish better from worse models.

### 6. Search depth is the key differentiator

C4 games can last 30+ moves with forced-win sequences 6-8 moves deep. At 100 sims, mean search depth is **7.3** — deep enough to discover these tactics. At 25 sims, depth is only 4.2 — many forced wins are invisible.

This explains why 100 sims is consistently 100% vs random while 25-50 sims fluctuate between 94-100%: the deeper search finds winning tactics that shallower searches miss.

## Key Takeaways

1. **Game complexity determines sim requirements.** TTT saturates at 10-25 sims. C4 needs 50-100. Chess/Go need 800+.
2. **Search depth is the mechanism.** More sims → deeper trees → discovers longer tactical sequences.
3. **Loss is misleading.** Always evaluate by play strength, not by loss on training targets.
4. **The baseline should use 50+ sims for C4.** The current default of 25 is under-provisioned.

## Next Steps

1. **Update C4 baseline** — re-run `baselines/connect4` with 50 or 100 sims to get a stronger reference model.
2. **c_puct sweep on C4** — now that we know 50-100 sims is the right range, sweep c_puct to find the C4-optimal exploration constant.
3. **Network capacity sweep on C4** — 4×128 MLP may be too small for C4's larger board (42 inputs vs 9). Try deeper/wider networks.
4. **Residual CNN** — MLP doesn't exploit board structure. A small ResNet could dramatically improve sample efficiency on C4.
