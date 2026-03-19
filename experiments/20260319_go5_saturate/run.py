#!/usr/bin/env python3
"""5x5 Go — SATURATE model capabilities on tiny board.

Goal: Prove we can beat GnuGo on a small board with an oversized model.
If this works, we know the algorithm is correct and can scale up.
If this fails, there's a fundamental issue beyond model size.

5x5 Go: 25 intersections, 26 actions. ~10^10 legal positions.
Model: 20 blocks × 256 filters = ~33M params (absurdly large for 5x5).
This is the "throw everything at it" experiment.

Also runs a 10b×128f (3M params) for comparison — is bigger actually better?
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


def run_gnugo_eval(weights_path, net_config, board_size=5, num_games=20, num_sims=400, level=1):
    """Run GnuGo evaluation."""
    try:
        cmd = [
            sys.executable, 'scripts/eval_vs_gnugo.py',
            '--weights', weights_path,
            '--board-size', str(board_size),
            '--num-filters', str(net_config.num_filters),
            '--num-res-blocks', str(net_config.num_res_blocks),
            '--gnugo-level', str(level),
            '--num-games', str(num_games),
            '--num-sims', str(num_sims),
        ]
        if net_config.use_se:
            cmd.append('--use-se')
        if net_config.global_pool_value:
            cmd.append('--global-pool-value')
        if net_config.use_ownership_head:
            cmd.append('--use-ownership-head')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        for line in result.stdout.split('\n'):
            if 'Final:' in line or 'Game' in line:
                print(f"  {line.strip()}")
        return result.stdout
    except Exception as e:
        print(f"  GnuGo eval error: {e}")
        return None


def train_config(name, num_blocks, num_filters, num_iters=300):
    """Create a training config."""
    checkpoint_dir = os.path.join(EXPERIMENT_DIR, 'data', name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    return AlphaZeroConfig(
        game="go5", seed=42,
        mcts=MCTSConfig(
            num_simulations=200,
            c_puct=1.5,
            dirichlet_alpha=0.03,
            dirichlet_epsilon=0.25,
            temp_threshold=15,
            nn_batch_size=8,
            policy_target_pruning=0.03,
            fpu_reduction=0.2, root_fpu_reduction=0.1,
            playout_cap_prob=0.125,
            playout_cap_cheap_fraction=0.15,
            coordinator_wait_us=25,
        ),
        network=NetworkConfig(
            network_type="cnn",
            num_filters=num_filters,
            num_res_blocks=num_blocks,
            use_se=True,
            global_pool_value=True,
            use_ownership_head=True,
        ),
        training=TrainingConfig(
            lr=0.001,
            lr_schedule="cosine",
            lr_min=0.0001,
            weight_decay=1e-4,
            batch_size=128,
            epochs_per_iteration=10,
            value_loss_weight=0.5,
            ownership_loss_weight=0.02,
            num_iterations=num_iters,
            games_per_iteration=100,
            max_buffer_size=100000,
            buffer_strategy="window",
            buffer_window=10,
            random_opening_moves=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=50,
            resume_from_checkpoint=True,
        ),
        arena=ArenaConfig(arena_games=0, eval_games=20),
        num_workers=10,
        use_cpp_mcts=True,
        use_wandb=False,
    )


def main():
    game = Go(size=5)
    print(f"5x5 Go: {game.get_action_size()} actions, {game.get_board_size()} board size")

    # Config A: BIG model (20 blocks × 256 filters)
    configs = [
        ("big_20b_256f", 20, 256, 500),
        ("medium_10b_128f", 10, 128, 500),
    ]

    for name, blocks, filters, iters in configs:
        print(f"\n{'='*70}")
        print(f"TRAINING: {name} ({blocks} blocks × {filters} filters)")
        print(f"{'='*70}")

        config = train_config(name, blocks, filters, iters)
        model = create_model(game, config.network, lr=config.training.lr,
                             weight_decay=config.training.weight_decay)

        total_params = sum(p.numel() for p in model.net.parameters())
        print(f"Parameters: {total_params:,}")
        print(f"Device: {model.net.device}")

        t0 = time.time()
        history = run_pipeline(game, model, config)
        elapsed = time.time() - t0

        losses = history['total_loss']
        vsR = history['vs_random_win_rate']
        print(f"\n{name}: {elapsed/60:.1f}m | best={min(losses):.3f} | "
              f"final vsRandom={vsR[-1]:.0%}")

        # GnuGo eval
        best_path = os.path.join(config.training.checkpoint_dir, 'best.pt')
        if os.path.exists(best_path):
            print(f"\n--- GnuGo Eval ({name}) ---")
            for level in [1, 3, 5]:
                run_gnugo_eval(best_path, config.network, board_size=5,
                               num_games=20, level=level)

        # Save config
        from dataclasses import asdict
        with open(os.path.join(EXPERIMENT_DIR, f'config_{name}.json'), 'w') as f:
            json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
