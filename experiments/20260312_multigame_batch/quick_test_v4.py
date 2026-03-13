#!/usr/bin/env python3
"""Quick test v4: Sweep coordinator_wait_us to find optimal batching delay.

v1 (50µs): 8t→2.6s, 54% GPU — best so far
v3 (10µs): 8t→5.6s, 47% GPU — regression, too many small batches

Sweep: 0, 25, 50, 100, 200, 500µs at 8 threads with 50 games.
Then test best wait_us with different thread counts.
"""
import json
import os
import sys
import time
import subprocess
import threading
from dataclasses import replace

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
        stop_event.wait(0.3)


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
    steady = gpu_samples[2:-1] if len(gpu_samples) > 4 else gpu_samples
    steady_gpu = np.mean(steady) if steady else avg_gpu

    games_per_sec = num_games / elapsed
    print(f"  [{label}] {elapsed:.1f}s  {games_per_sec:.1f} games/s  "
          f"GPU avg:{avg_gpu:.0f}% steady:{steady_gpu:.0f}% max:{max_gpu:.0f}%")
    return {
        'label': label,
        'time': elapsed,
        'games_per_sec': games_per_sec,
        'examples': len(examples),
        'avg_gpu': avg_gpu,
        'steady_gpu': steady_gpu,
        'max_gpu': max_gpu,
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

    base_mcts = MCTSConfig(
        num_simulations=200, c_puct=1.0, dirichlet_alpha=0.12, dirichlet_epsilon=0.25,
        temp_threshold=30, temp_decay_halflife=19, nn_batch_size=64,
        playout_cap_prob=0.125, playout_cap_cheap_fraction=0.15,
        fpu_reduction=0.2, root_fpu_reduction=0.1,
    )

    results = []
    NUM_GAMES = 50

    # Phase 1: Baseline (1 thread)
    print("=== Baseline ===")
    r = test_config(game, model, base_mcts, NUM_GAMES, 1, "1t_50us")
    results.append(r)
    base_gps = r['games_per_sec']

    # Phase 2: Sweep coordinator_wait_us at 8 threads
    print("\n=== Wait time sweep (8 threads, 50 games) ===")
    for wait_us in [0, 25, 50, 100, 200, 500]:
        cfg = replace(base_mcts, coordinator_wait_us=wait_us)
        r = test_config(game, model, cfg, NUM_GAMES, 8, f"8t_{wait_us}us")
        results.append(r)

    # Find best wait_us
    sweep_results = [r for r in results if r['label'].startswith('8t_')]
    best = max(sweep_results, key=lambda r: r['games_per_sec'])
    best_wait = int(best['label'].split('_')[1].replace('us', ''))
    print(f"\nBest wait: {best_wait}µs ({best['games_per_sec']:.1f} games/s)")

    # Phase 3: Thread sweep with best wait
    print(f"\n=== Thread sweep (wait={best_wait}µs, 50 games) ===")
    for nt in [4, 6, 8, 10, 12]:
        cfg = replace(base_mcts, coordinator_wait_us=best_wait)
        r = test_config(game, model, cfg, NUM_GAMES, nt, f"{nt}t_{best_wait}us")
        results.append(r)

    # Phase 4: Sustained load with best config
    best_threads = max(
        [r for r in results if r['label'].endswith(f'{best_wait}us') and r['label'] != f'8t_{best_wait}us'],
        key=lambda r: r['games_per_sec'],
        default=best
    )
    best_nt = int(best_threads['label'].split('t_')[0])
    print(f"\n=== Sustained (best: {best_nt}t, {best_wait}µs, 200 games) ===")
    cfg = replace(base_mcts, coordinator_wait_us=best_wait)
    r = test_config(game, model, cfg, 200, best_nt, f"{best_nt}t_{best_wait}us_200g")
    results.append(r)

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Label':<20} {'Time (s)':<10} {'Games/s':<10} {'Speedup':<10} {'GPU avg%':<10} {'Steady%':<10}")
    print("-" * 70)
    for r in results:
        speedup = r['games_per_sec'] / base_gps
        print(f"{r['label']:<20} {r['time']:<10.1f} {r['games_per_sec']:<10.1f} {speedup:<10.2f} "
              f"{r['avg_gpu']:<10.0f} {r['steady_gpu']:<10.0f}")

    out_path = os.path.join(EXPERIMENT_DIR, 'quick_test_v4_results.json')
    with open(out_path, 'w') as f:
        json.dump({'device': device, 'results': results}, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
