#!/usr/bin/env python3
"""Go 9x9 CAPACITY BREAKTHROUGH — larger network + ownership + LR decay.

Scale2000 proved that more games at the same architecture plateaus at loss ~0.67.
The bottleneck is model capacity and training sophistication, not data volume.

Changes from se_10ep_rom/scale2000:
1. **10 res blocks** (was 6) — reference impl used 10 blocks to reach 1-dan
2. **Ownership head** — se_ownership beat se_10ep_rom at 500 iters (0.677 vs 0.684)
3. **LR decay** — cosine schedule 0.001→0.0001 over training. Constant LR plateaus.
4. **Warm-start from se_ownership** (best model: loss 0.677)
5. 100 games/iter (not 200 — scale2000 proved 200 doesn't help)

Target: Break through the 0.67 plateau → beat GnuGo Level 1
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

# Warm-start from se_ownership (winner at 500 iters)
WARM_START_CANDIDATES = [
    os.path.join(os.path.dirname(EXPERIMENT_DIR),
                 '20260316_go9_se_ownership', 'data', 'checkpoints', 'best.pt'),
    os.path.join(os.path.dirname(EXPERIMENT_DIR),
                 '20260316_go9_se_ownership', 'data', 'checkpoints', 'checkpoint.pt'),
    # Fallback to se_10ep_rom
    os.path.join(os.path.dirname(EXPERIMENT_DIR),
                 '20260316_go9_se_10ep_rom', 'data', 'checkpoints', 'best.pt'),
]


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
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=10,         # KEY: 10 blocks (was 6). Reference used 10.
            use_se=True,
            global_pool_value=True,
            use_ownership_head=True,    # KEY: ownership was the winner
        ),
        training=TrainingConfig(
            lr=0.001,
            lr_schedule="cosine",       # KEY: LR decay (was constant)
            lr_min=0.0001,              # Decay to 0.0001 over training
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=10,
            value_loss_weight=0.5,
            ownership_loss_weight=0.02,
            num_iterations=1000,
            games_per_iteration=100,    # 100 (not 200 — proved 200 doesn't help)
            max_buffer_size=200000,
            buffer_strategy="window",
            buffer_window=15,
            random_opening_moves=6,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=50,
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

    # Warm-start
    warm_started = False
    for ws_path in WARM_START_CANDIDATES:
        if os.path.exists(ws_path):
            try:
                model.load(ws_path)
                print(f"Warm-started from {ws_path}")
                warm_started = True
                break
            except Exception as e:
                print(f"Failed to load {ws_path}: {e}")
                # Architecture mismatch (6→10 blocks) — can't warm-start, train from scratch
                continue
    if not warm_started:
        print("No compatible warm-start checkpoint. Training from scratch.")
        print("(Expected: 10-block network is different from 6-block checkpoints)")

    print(f"Device: {model.net.device}")
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"CAPACITY BREAKTHROUGH: 10 blocks, ownership, LR decay")
    print(f"Config: SE+GP 10b 128f, 10ep, ROM=6, cosine LR")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")


if __name__ == "__main__":
    main()
