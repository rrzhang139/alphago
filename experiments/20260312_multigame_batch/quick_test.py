#!/usr/bin/env python3
"""Quick 1-iteration test to find optimal thread count for GPU saturation."""
import json
import os
import sys
import time
import subprocess
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpha_go.games.go import Go
from alpha_go.neural_net import create_model
from alpha_go.utils.config import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig, TrainingConfig,
)
from alpha_go.training.self_play import generate_self_play_data

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_gpu_util(stop_event, samples):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, timeout=2
            )
            samples.append(float(out.strip().split('\n')[0]))
        except Exception:
            pass
        stop_event.wait(0.5)


def test_threads(game, model, mcts_config, num_games, num_threads):
    """Run 1 iteration of self-play, return timing and GPU util."""
    gpu_samples = []
    stop_event = threading.Event()
    gpu_thread = threading.Thread(target=sample_gpu_util, args=(stop_event, gpu_samples), daemon=True)
    gpu_thread.start()

    t0 = time.time()
    examples, stats = generate_self_play_data(
        game=game, model=model, mcts_config=mcts_config,
        num_games=num_games, augment=False,
        num_workers=num_threads, game_name="go9", use_cpp=True,
    )
    elapsed = time.time() - t0

    stop_event.set()
    gpu_thread.join(timeout=2)
    avg_gpu = np.mean(gpu_samples) if gpu_samples else 0.0
    max_gpu = max(gpu_samples) if gpu_samples else 0.0

    return {
        'num_threads': num_threads,
        'time': elapsed,
        'examples': len(examples),
        'avg_gpu': avg_gpu,
        'max_gpu': max_gpu,
        'gpu_samples': gpu_samples,
    }


def main():
    game = Go(size=9)

    mcts_config = MCTSConfig(
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
    )

    config = AlphaZeroConfig(
        game="go9",
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
        ),
        training=TrainingConfig(lr=0.002, weight_decay=1e-4),
    )

    model = create_model(game, config.network, lr=0.002, weight_decay=1e-4)
    device = str(model.net.device)
    print(f"Device: {device}")
    print(f"Config: 200 sims, nn_batch=64, 50 games, 128f CNN")
    print()

    NUM_GAMES = 50
    THREAD_COUNTS = [1, 2, 4, 8, 16]

    results = []
    for nt in THREAD_COUNTS:
        print(f"--- Testing {nt} threads ---")
        r = test_threads(game, model, mcts_config, NUM_GAMES, nt)
        results.append(r)
        print(f"  Time: {r['time']:.1f}s  GPU avg: {r['avg_gpu']:.0f}%  max: {r['max_gpu']:.0f}%  examples: {r['examples']}")
        print(f"  GPU samples: {[f'{s:.0f}' for s in r['gpu_samples']]}")
        print()

    # Summary
    base_time = results[0]['time']
    print("=" * 70)
    print(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<10} {'GPU avg%':<10} {'GPU max%':<10}")
    print("-" * 55)
    for r in results:
        speedup = base_time / max(r['time'], 0.01)
        print(f"{r['num_threads']:<10} {r['time']:<12.1f} {speedup:<10.2f} {r['avg_gpu']:<10.0f} {r['max_gpu']:<10.0f}")

    # Save
    out_path = os.path.join(EXPERIMENT_DIR, 'quick_test_results.json')
    save = {'device': device, 'num_games': NUM_GAMES, 'results': results}
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
