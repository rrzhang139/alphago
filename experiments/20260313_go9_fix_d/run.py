#!/usr/bin/env python3
"""Go 9x9 Fix D: Constant LR + window buffer (recent data only).

Diagnosis: FIFO 200K buffer fills with stale data from weaker models.
Fix: window buffer (keep last 10 iterations), 5 epochs, constant LR 0.001.
This ensures training data is always from recent (strong) models.

100 iters × 500 games = 50K games. From scratch.
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
            c_puct=1.0,
            dirichlet_alpha=0.12,
            dirichlet_epsilon=0.25,
            temp_threshold=30,
            temp_decay_halflife=19,
            nn_batch_size=64,
            playout_cap_prob=0.125,
            playout_cap_cheap_fraction=0.15,
            fpu_reduction=0.2,
            root_fpu_reduction=0.1,
            coordinator_wait_us=25,
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
        ),
        training=TrainingConfig(
            lr=0.001,                       # Constant LR
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=5,         # 5 epochs on smaller window
            num_iterations=100,
            games_per_iteration=500,
            max_buffer_size=200000,         # Not used for window
            buffer_strategy="window",       # Window buffer
            buffer_window=10,               # Keep last 10 iterations
            checkpoint_dir=CHECKPOINT_DIR,
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
    print("Fix D: Constant LR 0.001, 5 epochs, window buffer (10 iters), from scratch")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
