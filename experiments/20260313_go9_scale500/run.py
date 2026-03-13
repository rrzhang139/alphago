#!/usr/bin/env python3
"""Go 9x9 Scale: 500 iters warm-started from Fix D best model.

Fix D (100 iters) reached loss 3.06 but only 50% vs random.
Scale up to 500 iterations with eval_games every 25 iters to track strength.
Same config as Fix D: constant LR 0.001, 5 epochs, window buffer (10 iters).
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
WARM_START = os.path.join(os.path.dirname(__file__), '..', '20260313_go9_fix_d', 'data', 'checkpoints', 'best.pt')


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
            lr=0.001,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=5,
            num_iterations=500,
            games_per_iteration=500,
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=10,
            checkpoint_dir=CHECKPOINT_DIR,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=20),  # eval every iter
        num_workers=10,
        use_cpp_mcts=True,
        use_wandb=True,
    )

    from dataclasses import asdict
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    model = create_model(game, config.network, lr=config.training.lr,
                         weight_decay=config.training.weight_decay)

    # Warm-start from Fix D best model
    warm_start_path = os.path.abspath(WARM_START)
    if os.path.exists(warm_start_path):
        model.load(warm_start_path)
        print(f"Warm-started from {warm_start_path}")
    else:
        print(f"WARNING: warm-start not found at {warm_start_path}, training from scratch!")

    print(f"Device: {model.net.device}")
    print(f"Scale: 500 iters, warm-start from Fix D, eval_games=20")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
