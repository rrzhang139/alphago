#!/usr/bin/env python3
"""Quick multi-level GnuGo evaluation sweep.

Tests a model against GnuGo at levels 1, 3, 5, 8, 10 with a small
number of games per level. Gives a quick strength profile.

Usage:
    python scripts/eval_sweep.py --weights path/to/best.pt --use-se --global-pool-value --num-res-blocks 6
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from alpha_go.games.go import Go
from alpha_go.mcts.search import MCTS
from alpha_go.neural_net import create_model
from alpha_go.utils.config import MCTSConfig, NetworkConfig


# Reuse functions from eval_vs_gnugo
from eval_vs_gnugo import GnuGoGTP, play_game, action_to_gtp, gtp_to_action


def main():
    parser = argparse.ArgumentParser(description="Quick GnuGo strength sweep")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=10, help="Games per level")
    parser.add_argument("--num-sims", type=int, default=400)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--use-se", action="store_true")
    parser.add_argument("--global-pool-value", action="store_true")
    parser.add_argument("--levels", type=str, default="1,3,5,8,10",
                        help="Comma-separated GnuGo levels")
    args = parser.parse_args()

    game = Go(size=args.board_size)
    net_config = NetworkConfig(
        network_type="cnn",
        num_filters=args.num_filters,
        num_res_blocks=args.num_res_blocks,
        use_se=args.use_se,
        global_pool_value=args.global_pool_value,
    )
    model = create_model(game, net_config, lr=0.001)
    model.load(args.weights)
    print(f"Loaded weights from {args.weights}")

    mcts_config = MCTSConfig(
        num_simulations=args.num_sims,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.0,
        nn_batch_size=args.nn_batch_size,
        fpu_reduction=0.2,
        root_fpu_reduction=0.1,
    )

    levels = [int(x) for x in args.levels.split(",")]

    print(f"\n{'='*60}")
    print(f"GnuGo Strength Sweep — {args.num_sims} sims, {args.num_games} games/level")
    print(f"{'='*60}\n")

    results = {}
    for level in levels:
        gnugo = GnuGoGTP(level=level, size=args.board_size)
        wins = 0
        total_passes = 0
        all_scores = []

        for i in range(args.num_games):
            model_color = 1 if i % 2 == 0 else -1
            game_info = play_game(game, model, mcts_config, gnugo,
                                  model_color=model_color, verbose=False)
            if game_info['result'] == 1:
                wins += 1
            total_passes += game_info['passes']
            all_scores.append(game_info['score'])

        gnugo.close()
        wr = wins / args.num_games
        avg_passes = total_passes / args.num_games
        results[level] = {'wins': wins, 'total': args.num_games, 'wr': wr,
                          'avg_passes': avg_passes, 'scores': all_scores}

        print(f"Level {level:>2}: {wins}/{args.num_games} ({wr:.0%})  "
              f"| passes/game: {avg_passes:.1f}  "
              f"| scores: {all_scores[:5]}{'...' if len(all_scores) > 5 else ''}")

    print(f"\n{'='*60}")
    print("STRENGTH PROFILE:")
    for level, r in results.items():
        bar = "█" * int(r['wr'] * 20) + "░" * int((1 - r['wr']) * 20)
        print(f"  L{level:>2}: {bar} {r['wr']:.0%} ({r['wins']}/{r['total']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
