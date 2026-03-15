#!/usr/bin/env python3
"""Batch evaluate multiple checkpoints against GnuGo.

Finds all iter_NNNN.pt checkpoints in an experiment directory and evaluates
each against GnuGo, producing a strength-over-iterations curve.

Usage:
    python scripts/eval_checkpoints.py --exp-dir experiments/20260314_go9_fresh_correct/data/checkpoints
    python scripts/eval_checkpoints.py --exp-dir experiments/20260313_go9_scale500/data/checkpoints \
        --gnugo-level 1 --num-games 20 --num-sims 200
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig

# Import eval infrastructure from eval_vs_gnugo
from eval_vs_gnugo import GnuGoGTP, play_game


def find_checkpoints(exp_dir: str) -> list[tuple[int, str]]:
    """Find all iter_NNNN.pt files, return sorted (iter_num, path) pairs."""
    pattern = os.path.join(exp_dir, "iter_*.pt")
    files = glob.glob(pattern)
    results = []
    for f in files:
        match = re.search(r'iter_(\d+)\.pt$', f)
        if match:
            iter_num = int(match.group(1))
            results.append((iter_num, f))
    results.sort()
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate checkpoints vs GnuGo")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Directory containing iter_NNNN.pt checkpoints")
    parser.add_argument("--num-games", type=int, default=20,
                        help="Games per checkpoint (default: 20)")
    parser.add_argument("--num-sims", type=int, default=200,
                        help="MCTS sims per move (default: 200)")
    parser.add_argument("--gnugo-level", type=int, default=1,
                        help="GnuGo level 1-10 (default: 1)")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--use-se", action="store_true")
    parser.add_argument("--global-pool-value", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--also-best", action="store_true",
                        help="Also evaluate best.pt and final.pt")
    args = parser.parse_args()

    # Find checkpoints
    checkpoints = find_checkpoints(args.exp_dir)
    if args.also_best:
        for name in ['best.pt', 'final.pt']:
            path = os.path.join(args.exp_dir, name)
            if os.path.exists(path):
                checkpoints.append((-1 if name == 'best.pt' else -2, path))

    if not checkpoints:
        print(f"No checkpoints found in {args.exp_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    for iter_num, path in checkpoints:
        label = f"iter {iter_num}" if iter_num > 0 else os.path.basename(path)
        print(f"  {label}: {path}")

    # Setup
    game = Go(size=args.board_size)
    net_config = NetworkConfig(
        network_type="cnn",
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
        use_se=args.use_se,
        global_pool_value=args.global_pool_value,
    )
    mcts_config = MCTSConfig(
        num_simulations=args.num_sims,
        c_puct=args.c_puct,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.0,
        nn_batch_size=args.nn_batch_size,
        fpu_reduction=0.2,
        root_fpu_reduction=0.1,
    )

    gnugo = GnuGoGTP(level=args.gnugo_level, size=args.board_size)
    model = create_model(game, net_config, lr=0.001)

    results = []
    t_total = time.time()

    print(f"\nEvaluating vs GnuGo level {args.gnugo_level}, {args.num_games} games each, {args.num_sims} sims")
    print(f"{'Checkpoint':<20} {'W':>3} {'L':>3} {'D':>3} {'WR':>6}  {'Time':>6}")
    print("-" * 50)

    for iter_num, path in checkpoints:
        model.load(path)
        label = f"iter_{iter_num:04d}" if iter_num > 0 else os.path.basename(path).replace('.pt', '')

        wins, losses, draws = 0, 0, 0
        t0 = time.time()

        for i in range(args.num_games):
            model_color = 1 if i % 2 == 0 else -1
            result = play_game(game, model, mcts_config, gnugo, model_color=model_color)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        elapsed = time.time() - t0
        wr = wins / args.num_games
        print(f"{label:<20} {wins:>3} {losses:>3} {draws:>3} {wr:>5.0%}  {elapsed:>5.1f}s")

        results.append({
            'iteration': iter_num,
            'checkpoint': path,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wr,
            'num_games': args.num_games,
            'gnugo_level': args.gnugo_level,
            'num_sims': args.num_sims,
        })

    gnugo.close()

    total_elapsed = time.time() - t_total
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    # Save results
    output = args.output or os.path.join(args.exp_dir, f'gnugo_eval_L{args.gnugo_level}.json')
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
