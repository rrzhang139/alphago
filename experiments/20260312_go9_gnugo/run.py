#!/usr/bin/env python3
"""Go 9x9 full training — targeting GnuGo level 10 (~5 kyu).

Uses C++ MCTS with multi-game batched inference (v3) for ~4x self-play speedup.
Warm-start from playout_cap best model (100% vs random).

Config (adapted from go9_v2):
  - 200 sims, playout cap (12.5% full, 87.5% cheap @ 30 sims)
  - 500 games/iter × 200 iters = 100,000 total games
  - CNN 128 filters, 4 res blocks
  - C++ MCTS, 10 worker threads, 25µs coordinator wait
  - Cosine LR schedule 0.002 → 1e-5
  - No arena (always accept latest)
  - GnuGo eval every 25 iters

Estimated: ~8-12 hours on RTX A4000 (~$1.50-2.00)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig, ArenaConfig,
)
from alpha_go.training.pipeline import run_pipeline

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')

WARM_STARTS = [
    os.path.join(CHECKPOINT_DIR, 'warm_start.pt'),  # bundled with experiment
    os.path.join(EXPERIMENT_DIR, '..', '20260310_go9_playout_cap', 'data', 'checkpoints', 'best.pt'),
]


def eval_vs_gnugo(weights_path, level=10, num_games=20, num_sims=400):
    """Quick GnuGo evaluation. Returns win rate or None if gnugo not installed."""
    try:
        result = subprocess.run(
            ['gnugo', '--version'], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(EXPERIMENT_DIR, '..', '..', 'scripts', 'eval_vs_gnugo.py'),
             '--weights', weights_path,
             '--gnugo-level', str(level),
             '--num-games', str(num_games),
             '--num-sims', str(num_sims),
             '--num-filters', '128',
             '--num-res-blocks', '4'],
            capture_output=True, text=True, timeout=1800,  # 30 min max
        )
        # Parse win rate from output
        for line in result.stdout.strip().split('\n'):
            if 'Win rate' in line or 'win_rate' in line:
                # Try to extract number
                for part in line.split():
                    try:
                        val = float(part.strip('%').strip(':'))
                        if 0 <= val <= 100:
                            return val / 100 if val > 1 else val
                    except ValueError:
                        continue
        print(f"  GnuGo eval output: {result.stdout[-200:]}")
        return None
    except Exception as e:
        print(f"  GnuGo eval failed: {e}")
        return None


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
            lr=0.002,
            weight_decay=1e-4,
            batch_size=256,
            epochs_per_iteration=10,
            num_iterations=200,
            games_per_iteration=500,
            max_buffer_size=200000,
            buffer_strategy="fifo",
            checkpoint_dir=CHECKPOINT_DIR,
            lr_schedule="cosine",
            lr_min=1e-5,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=0),  # Skip eval (multiprocessing pool is slow on cloud vCPUs)
        num_workers=10,         # Multi-game batching: 10 threads
        use_cpp_mcts=True,      # C++ MCTS engine
        use_wandb=False,
        wandb_project="alphazero",
    )

    # Save config
    config_path = os.path.join(EXPERIMENT_DIR, 'config.json')
    from dataclasses import asdict
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")

    # Create model
    model = create_model(game, config.network, lr=config.training.lr,
                         weight_decay=config.training.weight_decay)
    print(f"Device: {model.net.device}")

    # Warm-start
    for warm_path in WARM_STARTS:
        if os.path.exists(warm_path):
            model.load(warm_path)
            print(f"Warm-started from {warm_path}")
            break
    else:
        print("WARNING: No warm-start weights found, training from scratch")

    t0 = time.time()
    history = run_pipeline(game, model, config)
    total = time.time() - t0

    # Final GnuGo eval
    best_pt = os.path.join(CHECKPOINT_DIR, 'best.pt')
    print(f"\nTotal training time: {total/60:.1f}m ({total/3600:.1f}h)")
    print(f"Best model: {best_pt}")

    print("\n--- Final GnuGo Evaluation ---")
    for level in [1, 3, 5, 10]:
        wr = eval_vs_gnugo(best_pt, level=level, num_games=20, num_sims=400)
        if wr is not None:
            print(f"  vs GnuGo L{level}: {wr*100:.0f}% win rate")
        else:
            print(f"  vs GnuGo L{level}: eval failed or gnugo not installed")


if __name__ == "__main__":
    main()
