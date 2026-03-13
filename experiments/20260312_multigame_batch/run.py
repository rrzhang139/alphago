#!/usr/bin/env python3
"""Benchmark: Multi-game batched inference coordinator.

Compares C++ MCTS with different thread counts:
- 1 thread: direct predict_fn (v2 baseline)
- 2/4/8 threads: BatchInferenceCoordinator (v3)

The coordinator collects NN requests from all workers into one mega-batch,
reducing GIL contention and increasing GPU utilization.
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
from alpha_go.training.self_play import generate_self_play_data, SelfPlayStats
from alpha_go.training.trainer import train_on_examples

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_gpu_util(stop_event, samples):
    """Background thread: sample GPU utilization every 0.5s."""
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


def run_benchmark(game, model, mcts_config, training_config, num_iters,
                  num_games, num_threads, label):
    """Run N training iterations and collect timing + metrics."""
    from collections import deque

    replay_buffer = deque(maxlen=training_config.max_buffer_size)
    results = {
        'label': label,
        'num_threads': num_threads,
        'self_play_times': [],
        'train_times': [],
        'iter_times': [],
        'gpu_utils': [],
        'losses': [],
        'examples_per_iter': [],
    }

    best_model = model

    for i in range(1, num_iters + 1):
        t_iter = time.time()

        # Self-play with GPU sampling
        gpu_samples = []
        stop_event = threading.Event()
        gpu_thread = threading.Thread(target=sample_gpu_util, args=(stop_event, gpu_samples), daemon=True)
        gpu_thread.start()

        t0 = time.time()
        new_examples, sp_stats = generate_self_play_data(
            game=game,
            model=best_model,
            mcts_config=mcts_config,
            num_games=num_games,
            augment=True,
            num_workers=num_threads,
            game_name="go9",
            use_cpp=True,
        )
        t_self_play = time.time() - t0

        stop_event.set()
        gpu_thread.join(timeout=2)
        avg_gpu = np.mean(gpu_samples) if gpu_samples else 0.0

        # Training
        replay_buffer.extend(new_examples)
        training_examples = list(replay_buffer)

        t0 = time.time()
        new_model = best_model.clone()
        losses = train_on_examples(
            model=new_model,
            examples=training_examples,
            batch_size=training_config.batch_size,
            epochs=training_config.epochs_per_iteration,
        )
        t_train = time.time() - t0

        best_model = new_model
        t_total = time.time() - t_iter

        results['self_play_times'].append(t_self_play)
        results['train_times'].append(t_train)
        results['iter_times'].append(t_total)
        results['gpu_utils'].append(avg_gpu)
        results['losses'].append(losses)
        results['examples_per_iter'].append(len(new_examples))

        print(f"  [{label}] iter {i}/{num_iters}: "
              f"self_play={t_self_play:.1f}s, train={t_train:.1f}s, total={t_total:.1f}s, "
              f"GPU={avg_gpu:.0f}%, loss={losses['total_loss']:.4f}, "
              f"examples={len(new_examples)}")

    return results, best_model


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

    training_config = TrainingConfig(
        lr=0.002,
        weight_decay=1e-4,
        batch_size=256,
        epochs_per_iteration=10,
        max_buffer_size=200000,
        buffer_strategy="fifo",
    )

    config = AlphaZeroConfig(
        game="go9",
        network=NetworkConfig(
            network_type="cnn",
            num_filters=128,
            num_res_blocks=4,
        ),
        training=training_config,
    )

    NUM_GAMES = 50
    BENCH_ITERS = 5
    THREAD_COUNTS = [1, 2, 4, 8]

    model = create_model(game, config.network, lr=training_config.lr,
                         weight_decay=training_config.weight_decay)
    device = str(model.net.device)
    print(f"Device: {device}")
    print(f"Benchmark: {BENCH_ITERS} iters × {NUM_GAMES} games, 200 sims")
    print(f"Thread counts to test: {THREAD_COUNTS}")
    print()

    all_results = {}

    for num_threads in THREAD_COUNTS:
        label = f"C++_{num_threads}t"
        print("=" * 50)
        print(f"Testing: {label}")
        print("=" * 50)

        # Fresh model clone for fair comparison
        test_model = model.clone()
        results, _ = run_benchmark(
            game, test_model, mcts_config, training_config,
            num_iters=BENCH_ITERS, num_games=NUM_GAMES,
            num_threads=num_threads, label=label,
        )
        all_results[label] = results
        print()

    # --- Print comparison table ---
    print("\n" + "=" * 80)
    print("MULTI-GAME BATCHING BENCHMARK RESULTS")
    print("=" * 80)

    baseline = all_results[f"C++_1t"]
    base_sp = np.mean(baseline['self_play_times'])

    print(f"\n{'Threads':<10} {'Self-play (s)':<15} {'GPU util %':<12} {'Speedup':>10} {'Examples':>10}")
    print("-" * 60)
    for label, results in all_results.items():
        sp = np.mean(results['self_play_times'])
        gpu = np.mean(results['gpu_utils'])
        exs = np.mean(results['examples_per_iter'])
        speedup = base_sp / max(sp, 0.01)
        print(f"{results['num_threads']:<10} {sp:<15.1f} {gpu:<12.0f} {speedup:>9.2f}x {exs:>10.0f}")

    print("\nPer-iteration self-play times:")
    for label, results in all_results.items():
        times_str = ", ".join(f"{t:.1f}" for t in results['self_play_times'])
        print(f"  {label}: [{times_str}]")

    # Save results
    save_data = {
        'device': device,
        'num_games': NUM_GAMES,
        'bench_iters': BENCH_ITERS,
        'results': {},
    }
    for label, results in all_results.items():
        save_data['results'][label] = {
            'num_threads': results['num_threads'],
            'self_play_times': results['self_play_times'],
            'train_times': results['train_times'],
            'gpu_utils': results['gpu_utils'],
            'losses': [l['total_loss'] for l in results['losses']],
            'examples_per_iter': results['examples_per_iter'],
            'mean_self_play': float(np.mean(results['self_play_times'])),
            'mean_gpu_util': float(np.mean(results['gpu_utils'])),
            'speedup_vs_1t': float(base_sp / max(np.mean(results['self_play_times']), 0.01)),
        }

    out_path = os.path.join(EXPERIMENT_DIR, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
