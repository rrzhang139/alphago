#!/usr/bin/env python3
"""Go 9x9 SE+GlobalPool extended: resume se_globalpool to 500 iters.

se_globalpool (200 iters) achieved the best loss of ALL experiments: 1.389
It was still improving at iter 200 (unlike scale500 which plateaued at iter 125).

This extends the same architecture to 500 iterations to see how far it can go.
Warm-starts from the se_globalpool final checkpoint.

Same config but with longer training. All proven improvements included:
- SE blocks + global pool (1.389 vs 1.639 baseline = architecture win)
- Policy target pruning 0.03 (9.3% better loss)
- Value loss weight 0.5 (7% better policy)
- BS=256 (13% better loss)
- Constant LR 0.001 (proven stable)
"""
import json
import os
import shutil
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
SOURCE_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), '..', '20260314_go9_se_globalpool', 'data', 'checkpoints', 'checkpoint.pt'
)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Copy checkpoint from se_globalpool if we don't have one yet
    dest_ckpt = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    if not os.path.exists(dest_ckpt) and os.path.exists(SOURCE_CHECKPOINT):
        print(f"Warm-starting from se_globalpool checkpoint")
        shutil.copy2(SOURCE_CHECKPOINT, dest_ckpt)
    elif os.path.exists(dest_ckpt):
        print(f"Resuming from existing checkpoint")
    else:
        print(f"WARNING: No source checkpoint found at {SOURCE_CHECKPOINT}")
        print(f"Training from scratch (not ideal)")

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
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=6,
            use_se=True,
            global_pool_value=True,
        ),
        training=TrainingConfig(
            lr=0.001,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=5,
            value_loss_weight=0.5,
            num_iterations=500,  # Extended from 200 to 500
            games_per_iteration=100,
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=10,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=25,
            resume_from_checkpoint=True,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=0),
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

    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"SE+GlobalPool extended: 500 iters, warm-start from se_globalpool (200 iters)")
    print(f"All proven improvements: pruning=0.03, VLW=0.5, BS=256")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
