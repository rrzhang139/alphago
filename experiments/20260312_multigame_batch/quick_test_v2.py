#!/usr/bin/env python3
"""Quick test v2: tune coordinator parameters to push GPU utilization higher.

Findings from v1: 8 threads optimal (3.5x, 54% GPU avg, 73% peak).
Hypotheses to test:
1. nn_batch_size=128 (bigger per-game batches → bigger mega-batches)
2. More games (100 instead of 50) for more sustained GPU load
3. 8 vs 12 threads (find the sweet spot between 8 and 16)
"""
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


def test_config(game, model, mcts_config, num_games, num_threads, label):
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

    print(f"  [{label}] Time: {elapsed:.1f}s  GPU avg: {avg_gpu:.0f}%  max: {max_gpu:.0f}%  "
          f"examples: {len(examples)}  samples: {len(gpu_samples)}")
    return {
        'label': label,
        'time': elapsed,
        'examples': len(examples),
        'avg_gpu': avg_gpu,
        'max_gpu': max_gpu,
        'gpu_samples': gpu_samples,
    }


def main():
    game = Go(size=9)
    config = AlphaZeroConfig(
        game="go9",
        network=NetworkConfig(network_type="cnn", num_filters=128, num_res_blocks=4),
        training=TrainingConfig(lr=0.002, weight_decay=1e-4),
    )
    model = create_model(game, config.network, lr=0.002, weight_decay=1e-4)
    device = str(model.net.device)
    print(f"Device: {device}")
    print()

    results = []

    # Test 1: Baseline — 8 threads, nn_batch=64, 50 games (reproduce v1 result)
    print("=== Test 1: Baseline (8t, batch=64, 50 games) ===")
    mcts_base = MCTSConfig(
        num_simulations=200, c_puct=1.0, dirichlet_alpha=0.12, dirichlet_epsilon=0.25,
        temp_threshold=30, temp_decay_halflife=19, nn_batch_size=64,
        playout_cap_prob=0.125, playout_cap_cheap_fraction=0.15,
        fpu_reduction=0.2, root_fpu_reduction=0.1,
    )
    r = test_config(game, model, mcts_base, 50, 8, "8t_b64_50g")
    results.append(r)

    # Test 2: Bigger nn_batch_size=128
    print("\n=== Test 2: Bigger batch (8t, batch=128, 50 games) ===")
    mcts_b128 = MCTSConfig(
        num_simulations=200, c_puct=1.0, dirichlet_alpha=0.12, dirichlet_epsilon=0.25,
        temp_threshold=30, temp_decay_halflife=19, nn_batch_size=128,
        playout_cap_prob=0.125, playout_cap_cheap_fraction=0.15,
        fpu_reduction=0.2, root_fpu_reduction=0.1,
    )
    r = test_config(game, model, mcts_b128, 50, 8, "8t_b128_50g")
    results.append(r)

    # Test 3: More games (100)
    print("\n=== Test 3: More games (8t, batch=64, 100 games) ===")
    r = test_config(game, model, mcts_base, 100, 8, "8t_b64_100g")
    results.append(r)

    # Test 4: Thread sweep around optimal (6, 8, 10, 12)
    print("\n=== Test 4: Thread sweep (batch=64, 50 games) ===")
    for nt in [6, 10, 12]:
        r = test_config(game, model, mcts_base, 50, nt, f"{nt}t_b64_50g")
        results.append(r)

    # Test 5: Best combo — 8 threads, batch=128, 100 games
    print("\n=== Test 5: Best combo (8t, batch=128, 100 games) ===")
    r = test_config(game, model, mcts_b128, 100, 8, "8t_b128_100g")
    results.append(r)

    # Test 6: 10 threads, batch=128, 100 games
    print("\n=== Test 6: 10 threads (10t, batch=128, 100 games) ===")
    r = test_config(game, model, mcts_b128, 100, 10, "10t_b128_100g")
    results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Label':<20} {'Time (s)':<10} {'GPU avg%':<10} {'GPU max%':<10} {'Examples':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<20} {r['time']:<10.1f} {r['avg_gpu']:<10.0f} {r['max_gpu']:<10.0f} {r['examples']:<10}")

    out_path = os.path.join(EXPERIMENT_DIR, 'quick_test_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump({'device': device, 'results': results}, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
