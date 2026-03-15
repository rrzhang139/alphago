#!/usr/bin/env python3
"""Go 9x9 Kitchen Sink: ALL proven improvements combined.

Combines every technique that showed promise in local experiments:
- Correct alpha=0.03, c_puct=1.5 (proven in playout_cap experiment)
- Window buffer (proven superior to FIFO in Fix D)
- FPU reduction 0.2/0.1 (proven in playout_cap)
- Playout cap randomization (12.5% full/200 sims)
- Liberty planes (KataGo input features, 23 planes)
- Shaped Dirichlet noise (KataGo)
- Policy surprise weighting (KataGo, lambda=0.5)
- Checkpoint/resume for crash recovery
- Constant LR 0.001 (proven superior to cosine)

From scratch — no warm start from any previous model.
300 iterations, 100 games/iter (tuned for A5000+Threadripper).
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
            nn_batch_size=8,  # Optimal for C++ engine on GPU
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
            policy_surprise_weight=0.5,  # KataGo PSW
        ),
        arena=ArenaConfig(arena_games=0, eval_games=0),  # Eval separately
        num_workers=10,
        use_cpp_mcts=True,
        use_wandb=True,
    )

    from dataclasses import asdict
    with open(os.path.join(EXPERIMENT_DIR, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    model = create_model_from_config(game, config)
    print(f"Device: {model.net.device}")
    print(f"Input: {game.get_board_shape()} = {game.num_planes} planes "
          f"({'17 base + 6 liberty' if game.use_liberty_planes else '17 base'})")
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"KITCHEN SINK: liberty planes + shaped Dirichlet + PSW=0.5")
    print(f"FROM SCRATCH, 300 iters, 100 games/iter, nn_batch=8")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
