#!/usr/bin/env python3
"""Quick test v3: push toward GPU saturation.

v2 findings: 8t+b128 fastest per-game, 100 games sustains 66% GPU.
Now try: larger batches (200), more threads with 100 games, and combined.
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
        stop_event.wait(0.3)  # sample more frequently


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
    # Steady-state: skip first 2 and last 1 samples (ramp up/down)
    steady = gpu_samples[2:-1] if len(gpu_samples) > 4 else gpu_samples
    steady_gpu = np.mean(steady) if steady else 0.0

    print(f"  [{label}] Time: {elapsed:.1f}s  GPU avg: {avg_gpu:.0f}%  steady: {steady_gpu:.0f}%  "
          f"max: {max_gpu:.0f}%  examples: {len(examples)}")
    return {
        'label': label,
        'time': elapsed,
        'examples': len(examples),
        'avg_gpu': avg_gpu,
        'steady_gpu': steady_gpu,
        'max_gpu': max_gpu,
        'gpu_samples': gpu_samples,
    }


def make_config(nn_batch_size=64):
    return MCTSConfig(
        num_simulations=200, c_puct=1.0, dirichlet_alpha=0.12, dirichlet_epsilon=0.25,
        temp_threshold=30, temp_decay_halflife=19, nn_batch_size=nn_batch_size,
        playout_cap_prob=0.125, playout_cap_cheap_fraction=0.15,
        fpu_reduction=0.2, root_fpu_reduction=0.1,
    )


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

    # Reference: 1 thread baseline (v2)
    print("=== 1 thread baseline ===")
    r = test_config(game, model, make_config(64), 50, 1, "1t_b64")
    results.append(r)

    # Batch size sweep at 8 threads, 100 games
    print("\n=== Batch size sweep (8t, 100 games) ===")
    for bs in [64, 128, 200]:
        r = test_config(game, model, make_config(bs), 100, 8, f"8t_b{bs}_100g")
        results.append(r)

    # Thread sweep at batch=128, 100 games
    print("\n=== Thread sweep (b128, 100 games) ===")
    for nt in [4, 8, 12, 16, 24]:
        r = test_config(game, model, make_config(128), 100, nt, f"{nt}t_b128_100g")
        results.append(r)

    # Best combo: highest GPU util config with 200 games for sustained measurement
    print("\n=== Sustained load (200 games) ===")
    r = test_config(game, model, make_config(128), 200, 8, "8t_b128_200g")
    results.append(r)
    r = test_config(game, model, make_config(128), 200, 12, "12t_b128_200g")
    results.append(r)

    # Summary
    base_time = results[0]['time']  # 1 thread baseline
    print("\n" + "=" * 90)
    print(f"{'Label':<20} {'Time (s)':<10} {'Speedup':<10} {'GPU avg%':<10} {'Steady%':<10} {'GPU max%':<10}")
    print("-" * 70)
    for r in results:
        speedup = base_time / max(r['time'], 0.01)
        print(f"{r['label']:<20} {r['time']:<10.1f} {speedup:<10.2f} {r['avg_gpu']:<10.0f} "
              f"{r['steady_gpu']:<10.0f} {r['max_gpu']:<10.0f}")

    out_path = os.path.join(EXPERIMENT_DIR, 'quick_test_v3_results.json')
    with open(out_path, 'w') as f:
        json.dump({'device': device, 'results': results}, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
