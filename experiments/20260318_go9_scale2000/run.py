#!/usr/bin/env python3
"""Go 9x9 SCALE RUN — 2000 iterations, 200 games/iter.

The bottleneck analysis is clear: we have ~50K self-play games, the reference
implementation (michaelnny/alpha_zero) used 1M+. We need 20x more training data.

This experiment:
- 2000 iterations × 200 games = 400K games (8x our current best)
- Warm-starts from se_10ep_rom's best checkpoint (loss 0.96)
- All proven improvements (SE+GP, 10ep, ROM, pruning, WD, VLW)
- LR decay at iter 1000 and 1500 (needed for long training)
- Checkpoint every 50 iters + GnuGo eval every 100 iters

Target: Beat GnuGo Level 1 (>80% win rate)
Stretch: Beat GnuGo Level 5

Hardware: A100 80GB recommended (fastest inference, large VRAM for batching)
Estimated time: ~24-36h on A100, ~48-72h on A4000
Estimated cost: $30-45 on A100
"""
import json
import os
import subprocess
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

# Warm-start from se_10ep_rom's best checkpoint
WARM_START_CANDIDATES = [
    os.path.join(os.path.dirname(EXPERIMENT_DIR),
                 '20260316_go9_se_10ep_rom', 'data', 'checkpoints', 'best.pt'),
    os.path.join(os.path.dirname(EXPERIMENT_DIR),
                 '20260316_go9_se_10ep_rom', 'data', 'checkpoints', 'checkpoint.pt'),
]


def run_gnugo_eval(weights_path, num_games=20, num_sims=400, level=1):
    """Run GnuGo evaluation and return win rate."""
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/eval_vs_gnugo.py',
             '--weights', weights_path,
             '--use-se', '--global-pool-value', '--num-res-blocks', '6',
             '--gnugo-level', str(level),
             '--num-games', str(num_games),
             '--num-sims', str(num_sims)],
            capture_output=True, text=True, timeout=1800)
        output = result.stdout
        # Parse win rate from output
        for line in output.split('\n'):
            if 'Final:' in line and 'win rate' in line:
                print(f"  GnuGo L{level}: {line.strip()}")
                return line
        print(f"  GnuGo L{level} eval failed: {output[-200:]}")
    except Exception as e:
        print(f"  GnuGo L{level} eval error: {e}")
    return None


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
            num_res_blocks=6,
            use_se=True,
            global_pool_value=True,
        ),
        training=TrainingConfig(
            lr=0.001,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=10,
            value_loss_weight=0.5,
            num_iterations=2000,
            games_per_iteration=200,       # 2x current → 400K total games
            max_buffer_size=500000,         # Larger buffer for more data
            buffer_strategy="window",
            buffer_window=15,              # 15 × 200 = 3000 games in window
            random_opening_moves=6,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=50,        # Checkpoint every 50 iters
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

    # Warm-start from se_10ep_rom
    warm_started = False
    for ws_path in WARM_START_CANDIDATES:
        if os.path.exists(ws_path):
            model.load(ws_path)
            print(f"Warm-started from {ws_path}")
            warm_started = True
            break
    if not warm_started:
        print("WARNING: No warm-start checkpoint found. Training from scratch.")

    print(f"Device: {model.net.device}")
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"SCALE RUN: 2000 iters × 200 games = 400K games")
    print(f"Config: SE+GP 6b 128f, 10ep, ROM=6, WD=1e-4, BS=256")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f}m ({total/3600:.1f}h)")

    # Final GnuGo eval
    best_path = os.path.join(CHECKPOINT_DIR, 'best.pt')
    if os.path.exists(best_path):
        print("\n=== Final GnuGo Evaluation ===")
        run_gnugo_eval(best_path, num_games=50, level=1)
        run_gnugo_eval(best_path, num_games=20, level=3)
        run_gnugo_eval(best_path, num_games=20, level=5)


if __name__ == "__main__":
    main()
