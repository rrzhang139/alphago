#!/usr/bin/env python3
"""Experiment: How does c_puct (exploration constant) affect training?

Sweep c_puct = [0.1, 0.5, 1.0, 2.0, 5.0] with everything else at defaults.
Measures: vs_random win rate, loss curves, policy entropy, search depth, training time.

Hypothesis: c_puct controls the explore/exploit tradeoff in MCTS.
- Too low (0.1): search barely explores — commits to the first decent-looking move.
  Narrow trees, fast convergence, but might miss strong lines.
- Too high (5.0): search wastes simulations on bad moves the network dislikes.
  Broad shallow trees, noisy training targets.
- Sweet spot (~1.0): enough exploration to correct network mistakes without
  wasting budget. For tic-tac-toe (small action space), the sweet spot may
  be broad since there are only 9 moves to explore.
"""

import json
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games import get_game
from alpha_go.neural_net.simple_net import SimpleNetWrapper
from alpha_go.training.pipeline import run_pipeline
from alpha_go.utils.config import (
    AlphaZeroConfig, ArenaConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, 'data')
FIG_DIR = os.path.join(EXPERIMENT_DIR, 'figures')

CPUCT_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

# Fixed training params — same as num_sims sweep for comparability
TRAINING = TrainingConfig(
    num_iterations=15,
    games_per_iteration=50,
    epochs_per_iteration=10,
    batch_size=64,
    lr=0.001,
)
ARENA = ArenaConfig(arena_games=30, update_threshold=0.55)
NETWORK = NetworkConfig(hidden_size=128, num_layers=4)


def run_single(c_puct: float, seed: int = 42) -> dict:
    """Run one training session with the given c_puct value."""
    tag = f'cpuct_{c_puct}'
    config = AlphaZeroConfig(
        mcts=MCTSConfig(c_puct=c_puct),
        network=NETWORK,
        training=TrainingConfig(
            num_iterations=TRAINING.num_iterations,
            games_per_iteration=TRAINING.games_per_iteration,
            epochs_per_iteration=TRAINING.epochs_per_iteration,
            batch_size=TRAINING.batch_size,
            lr=TRAINING.lr,
            checkpoint_dir=os.path.join(DATA_DIR, tag),
        ),
        arena=ARENA,
        seed=seed,
    )

    game = get_game('tictactoe')
    model = SimpleNetWrapper(
        board_size=game.get_board_size(),
        action_size=game.get_action_size(),
        config=config.network,
        lr=config.training.lr,
    )

    t0 = time.time()
    history = run_pipeline(game, model, config)
    elapsed = time.time() - t0

    history['wall_time'] = elapsed
    history['c_puct'] = c_puct
    return history


def main():
    all_results = {}

    for c_puct in CPUCT_VALUES:
        print(f"\n{'#'*70}")
        print(f"#  c_puct = {c_puct}")
        print(f"{'#'*70}")

        history = run_single(c_puct)
        all_results[c_puct] = history

        # Save per-run data
        tag = f'cpuct_{c_puct}'
        path = os.path.join(DATA_DIR, f'{tag}.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    # Generate comparison plots
    plot_comparison(all_results)

    # Print summary table
    print_summary(all_results)

    # Save config
    config_path = os.path.join(EXPERIMENT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'experiment': 'c_puct sweep',
            'c_puct_values': CPUCT_VALUES,
            'training': {
                'num_iterations': TRAINING.num_iterations,
                'games_per_iteration': TRAINING.games_per_iteration,
                'epochs_per_iteration': TRAINING.epochs_per_iteration,
                'batch_size': TRAINING.batch_size,
                'lr': TRAINING.lr,
            },
            'arena': {
                'arena_games': ARENA.arena_games,
                'update_threshold': ARENA.update_threshold,
            },
            'network': {
                'hidden_size': NETWORK.hidden_size,
                'num_layers': NETWORK.num_layers,
            },
            'mcts_other': {
                'num_simulations': 25,
                'dirichlet_alpha': 0.3,
                'dirichlet_epsilon': 0.25,
                'temp_threshold': 15,
            },
            'seed': 42,
        }, f, indent=2)


def plot_comparison(all_results: dict):
    """Generate comparison plots across all c_puct values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cpuct_sorted = sorted(all_results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cpuct_sorted)))

    # --- vs Random win rate over iterations ---
    ax = axes[0, 0]
    for i, cpuct in enumerate(cpuct_sorted):
        hist = all_results[cpuct]
        iters = hist['iteration']
        wr = [v * 100 for v in hist['vs_random_win_rate']]
        ax.plot(iters, wr, '-o', color=colors[i], markersize=3, linewidth=1.5,
                label=f'c={cpuct}')
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.4)
    ax.set_ylabel('Win Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('vs Random Player')
    ax.set_ylim(40, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Policy entropy over iterations ---
    ax = axes[0, 1]
    for i, cpuct in enumerate(cpuct_sorted):
        hist = all_results[cpuct]
        if hist.get('policy_entropy'):
            ax.plot(hist['iteration'], hist['policy_entropy'], '-o', color=colors[i],
                    markersize=3, linewidth=1.5, label=f'c={cpuct}')
    ax.set_ylabel('Entropy (nats)')
    ax.set_xlabel('Iteration')
    ax.set_title('MCTS Policy Entropy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Total loss over iterations ---
    ax = axes[0, 2]
    for i, cpuct in enumerate(cpuct_sorted):
        hist = all_results[cpuct]
        ax.plot(hist['iteration'], hist['total_loss'], '-', color=colors[i],
                linewidth=1.5, label=f'c={cpuct}')
    ax.set_ylabel('Total Loss')
    ax.set_xlabel('Iteration')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Mean search depth ---
    ax = axes[1, 0]
    for i, cpuct in enumerate(cpuct_sorted):
        hist = all_results[cpuct]
        if hist.get('mean_search_depth'):
            ax.plot(hist['iteration'], hist['mean_search_depth'], '-o', color=colors[i],
                    markersize=3, linewidth=1.5, label=f'c={cpuct}')
    ax.set_ylabel('Mean Depth')
    ax.set_xlabel('Iteration')
    ax.set_title('MCTS Search Depth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Draw rate over iterations ---
    ax = axes[1, 1]
    for i, cpuct in enumerate(cpuct_sorted):
        hist = all_results[cpuct]
        outcomes = hist.get('self_play_outcomes', [])
        if outcomes:
            total_games = [o[0] + o[1] + o[2] for o in outcomes]
            draw_rates = [o[2] / max(1, t) * 100 for o, t in zip(outcomes, total_games)]
            ax.plot(hist['iteration'], draw_rates, '-o', color=colors[i],
                    markersize=3, linewidth=1.5, label=f'c={cpuct}')
    ax.set_ylabel('Draw Rate %')
    ax.set_xlabel('Iteration')
    ax.set_title('Self-Play Draw Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Final metrics bar chart ---
    ax = axes[1, 2]
    final_wr = [all_results[c]['vs_random_win_rate'][-1] * 100 for c in cpuct_sorted]
    wall_times = [all_results[c]['wall_time'] for c in cpuct_sorted]

    x = np.arange(len(cpuct_sorted))
    width = 0.35
    ax.bar(x - width/2, final_wr, width, label='Final vs Random %',
           color='#2ecc71', alpha=0.7)
    ax.set_ylabel('Win Rate %', color='#2ecc71')
    ax.set_ylim(0, 110)

    ax2 = ax.twinx()
    ax2.bar(x + width/2, wall_times, width, label='Wall Time (s)',
            color='#3498db', alpha=0.7)
    ax2.set_ylabel('Time (s)', color='#3498db')

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cpuct_sorted])
    ax.set_xlabel('c_puct')
    ax.set_title('Final Performance vs Cost')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('c_puct Sweep — Tic-Tac-Toe (25 sims, 15 iters × 50 games)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(FIG_DIR, 'cpuct_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlots saved to {path}")


def print_summary(all_results: dict):
    """Print a summary table of all runs."""
    print(f"\n{'='*90}")
    print("SUMMARY: c_puct sweep")
    print(f"{'='*90}")
    print(f"{'c_puct':>6}  {'Final vsRand':>12}  {'Best vsRand':>12}  {'Final Loss':>10}  "
          f"{'Final H(pi)':>11}  {'Final Depth':>11}  {'Time':>8}")
    print("─" * 90)

    for cpuct in sorted(all_results.keys()):
        h = all_results[cpuct]
        final_wr = h['vs_random_win_rate'][-1]
        best_wr = max(h['vs_random_win_rate'])
        final_loss = h['total_loss'][-1]
        final_entropy = h['policy_entropy'][-1] if h.get('policy_entropy') else 0
        final_depth = h['mean_search_depth'][-1] if h.get('mean_search_depth') else 0
        wall = h['wall_time']

        print(f"{cpuct:>6.1f}  {final_wr:>11.0%}  {best_wr:>11.0%}  {final_loss:>10.3f}  "
              f"{final_entropy:>11.2f}  {final_depth:>11.1f}  {wall:>7.1f}s")

    print()


if __name__ == '__main__':
    main()
