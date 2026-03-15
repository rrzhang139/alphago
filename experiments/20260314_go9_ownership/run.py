#!/usr/bin/env python3
"""Go 9x9 with ownership prediction (KataGo's biggest improvement).

Same as kitchen sink PLUS auxiliary ownership head:
- Predicts per-intersection ownership at game end
- 361 bits of learning signal per game (vs 1 bit from win/loss)
- KataGo found this gives ~1.65x training efficiency

Network has 3 heads: policy, value, and ownership.
Training data includes ownership maps computed from terminal positions.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model_from_config
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ArenaConfig,
)
from alpha_go.training.pipeline import run_pipeline

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, 'data', 'checkpoints')


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    game = Go(size=9, use_liberty_planes=True)

    config = AlphaZeroConfig(
        game="go9_ext",
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
            policy_target_pruning=0.03,  # KataGo: prune <3% visit moves from target
            shaped_dirichlet=True,
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
            use_ownership_head=True,  # KataGo auxiliary ownership
            global_pool_value=True,  # KataGo 1.60x improvement
        ),
        training=TrainingConfig(
            lr=0.001,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=5,
            num_iterations=300,
            games_per_iteration=100,
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=10,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=25,
            resume_from_checkpoint=True,
            policy_surprise_weight=0.5,
            ownership_loss_weight=0.02,  # KataGo: 1.5/b² ≈ 0.019 for 9x9
        ),
        arena=ArenaConfig(arena_games=0, eval_games=0),
        num_workers=10,
        use_cpp_mcts=True,
        use_wandb=True,
    )

    from dataclasses import asdict
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    model = create_model_from_config(game, config)
    model._ownership_loss_weight = config.training.ownership_loss_weight
    print(f"Device: {model.net.device}")
    print(f"Input: {game.get_board_shape()} = {game.num_planes} planes")
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"OWNERSHIP + kitchen sink: all KataGo improvements")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
