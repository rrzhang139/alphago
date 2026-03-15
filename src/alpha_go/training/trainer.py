"""PyTorch training: takes (board, policy, value[, ownership]) tuples, updates model.

Standard minibatch SGD over the replay buffer.
"""

from __future__ import annotations

import numpy as np


def train_on_examples(
    model,
    examples: list[tuple],
    batch_size: int,
    epochs: int,
) -> dict[str, float]:
    """Train the model on collected self-play examples.

    Args:
        model: Neural network wrapper with train_step method.
        examples: List of (state, target_policy, target_value) or
                  (state, target_policy, target_value, target_ownership).
        batch_size: Minibatch size.
        epochs: Number of passes over the data.

    Returns:
        Average losses over all training: {'total_loss', 'policy_loss', 'value_loss'}.
    """
    states = np.array([e[0] for e in examples])
    pis = np.array([e[1] for e in examples])
    vs = np.array([e[2] for e in examples], dtype=np.float32)

    # Check if ownership targets are present (4-tuple examples)
    has_ownership = len(examples[0]) > 3
    ownerships = None
    if has_ownership:
        ownerships = np.array([e[3] for e in examples], dtype=np.float32)

    n = len(examples)
    total_losses = {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
    if has_ownership:
        total_losses['ownership_loss'] = 0.0
    num_batches = 0

    for _ in range(epochs):
        # Shuffle
        indices = np.random.permutation(n)
        states = states[indices]
        pis = pis[indices]
        vs = vs[indices]
        if has_ownership:
            ownerships = ownerships[indices]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_states = states[start:end]
            batch_pis = pis[start:end]
            batch_vs = vs[start:end]

            if has_ownership:
                batch_own = ownerships[start:end]
                losses = model.train_step(batch_states, batch_pis, batch_vs,
                                          target_ownership=batch_own)
            else:
                losses = model.train_step(batch_states, batch_pis, batch_vs)

            for k in losses:
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += losses[k]
            num_batches += 1

    # Average
    if num_batches > 0:
        for k in total_losses:
            total_losses[k] /= num_batches

    return total_losses
