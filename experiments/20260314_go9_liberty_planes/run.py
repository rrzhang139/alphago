#!/usr/bin/env python3
"""Go 9x9 with liberty planes (KataGo-style input features).

23 input planes: 17 base (AlphaGo Zero) + 6 liberty planes:
- 3 own stone planes: exactly 1 liberty (atari), 2 liberties, 3+ liberties
- 3 opponent stone planes: same

KataGo found input features gave 1.55x training efficiency.
Liberty planes directly encode group health which the network
otherwise must learn from raw stone patterns.

C++ MCTS engine supports liberty planes - no speed penalty.
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
            num_iterations=300,
            games_per_iteration=500,
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=10,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=25,
            resume_from_checkpoint=True,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=20),
        num_workers=10,
        use_cpp_mcts=True,  # C++ engine now supports liberty planes
        use_wandb=True,
    )

    from dataclasses import asdict
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    model = create_model(game, config.network, lr=config.training.lr,
                         weight_decay=config.training.weight_decay)
    print(f"Device: {model.net.device}")
    print(f"Input: {game.get_board_shape()} = 23 planes (17 base + 6 liberty)")
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"FROM SCRATCH with liberty planes, correct alpha=0.03, c_puct=1.5")
    print(f"NOTE: Python MCTS (no C++ support for liberty planes)")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
