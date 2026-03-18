#!/usr/bin/env python3
"""Go 9x9 SE+GP with 10 epochs, ROM, and OWNERSHIP prediction.

Combines ALL proven improvements plus KataGo's biggest training efficiency gain:
- SE blocks + global pool (proven: 1.389 vs 1.639)
- 10 epochs per iteration (proven: 47% better loss)
- Random opening moves 6 (proven: 5.5% better)
- Policy target pruning 0.03 (proven: 9.3% better)
- Value loss weight 0.5 (proven: 7% better policy)
- BS=256 (proven: 13% better loss)
- Constant LR 0.001

PLUS:
- Auxiliary ownership prediction head (KataGo's 1.65x improvement)
  Predicts per-intersection ownership at game end.
  361 bits of learning signal per game vs 1 bit from win/loss.

From scratch, 500 iterations.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ArenaConfig,
)
from alpha_go.training.pipeline import run_pipeline

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, 'data', 'checkpoints')


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    game = Go(size=9)

    config = AlphaZeroConfig(
        game="go9",
        seed=42,
        mcts=MCTSConfig(
            num_simulations=200,
            c_puct=1.5,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
            temp_threshold=30,
            temp_decay_halflife=19,
            nn_batch_size=8,
            playout_cap_prob=0.125,
            playout_cap_cheap_fraction=0.15,
            fpu_reduction=0.2,
            root_fpu_reduction=0.1,
            coordinator_wait_us=25,
            policy_target_pruning=0.03,
            c_puct_base=19652.0,  # AlphaZero log-scaling: 12.8% better locally
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=6,
            use_se=True,
            global_pool_value=True,
            use_ownership_head=True,  # KataGo's 1.65x improvement
        ),
        training=TrainingConfig(
            lr=0.001,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=10,
            value_loss_weight=0.5,
            ownership_loss_weight=0.02,  # KataGo: 1.5/b^2 ~ 0.019
            num_iterations=500,
            games_per_iteration=100,
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=10,
            random_opening_moves=6,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=25,
            resume_from_checkpoint=True,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=0),
        num_workers=10,
        use_cpp_mcts=True,
        use_wandb=False,
    )

    from dataclasses import asdict
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    model = create_model(game, config.network, lr=config.training.lr,
                         weight_decay=config.training.weight_decay)
    print(f"Device: {model.net.device}")

    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"SE+GP, 10 epochs, ROM=6, OWNERSHIP HEAD, from scratch, 500 iters")
    print(f"All improvements: pruning=0.03, VLW=0.5, BS=256, ownership_weight=0.02")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
