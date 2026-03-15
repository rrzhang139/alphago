#!/usr/bin/env python3
"""Go 9x9 from scratch with ALL correct parameters.

Previous experiments had dirichlet_alpha=0.12 (negligible exploration for 82 actions).
This uses the correct alpha=0.03 (AlphaGo Zero standard) + c_puct=1.5 (proven better).
From scratch - no warm start from Fix D which learned with broken exploration.

Key config matches the successful playout_cap experiment:
- dirichlet_alpha=0.03 (concentrated noise)
- c_puct=1.5 (more exploration)
- playout_cap (12.5% full/200 sims, 87.5% cheap/30 sims)
Plus proven improvements from Fix D:
- window buffer (last 10 iters, fresh data)
- constant LR 0.001
- FPU reduction 0.2/0.1
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
            c_puct=1.5,                     # Proven better than 1.0
            dirichlet_alpha=0.03,            # Standard for Go (82 actions)
            dirichlet_epsilon=0.25,
            temp_threshold=30,
            temp_decay_halflife=19,
            nn_batch_size=8,  # Optimal for C++ engine (infra agent found 64 is too large)
            playout_cap_prob=0.125,
            playout_cap_cheap_fraction=0.15,
            fpu_reduction=0.2,
            root_fpu_reduction=0.1,
            coordinator_wait_us=25,
            policy_target_pruning=0.03,  # KataGo: prune <3% visit moves from target
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
            num_iterations=300,              # 300 iters from scratch
            games_per_iteration=100,  # Tuned for A5000+Threadripper (500 was too slow)
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=10,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=25,
            resume_from_checkpoint=True,     # Auto-resume on crash
        ),
        arena=ArenaConfig(arena_games=0, eval_games=0),  # Eval separately (saves 4 min/iter)
        num_workers=10,
        use_cpp_mcts=True,
        use_wandb=True,
    )

    from dataclasses import asdict
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    model = create_model(game, config.network, lr=config.training.lr,
                         weight_decay=config.training.weight_decay)
    print(f"Device: {model.net.device}")
    print(f"FROM SCRATCH with correct alpha=0.03, c_puct=1.5")
    print(f"300 iters, 500 games/iter, window buffer, FPU 0.2/0.1")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
