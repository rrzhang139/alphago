"""Self-play: generate training data by playing games with MCTS + current model.

Each self-play game produces a list of (state, policy, value) tuples.
The state is the canonical board, the policy is the MCTS visit distribution,
and the value is the final game outcome from that player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..games.base_game import Game
from ..mcts.search import MCTS
from ..utils.config import MCTSConfig


@dataclass
class SelfPlayStats:
    """Aggregated statistics from a batch of self-play games."""
    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0
    mean_game_length: float = 0.0
    mean_root_value: float = 0.0
    mean_policy_entropy: float = 0.0
    mean_search_depth: float = 0.0

    @property
    def outcomes_tuple(self) -> tuple[int, int, int]:
        return (self.p1_wins, self.p2_wins, self.draws)


def self_play_game(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    collect_diagnostics: bool = False,
    score_value_weight: float = 0.0,
    score_value_scale: float = 20.0,
    collect_ownership: bool = False,
) -> tuple[list[tuple], int, dict]:
    """Play one game of self-play, returning training examples, outcome, and diagnostics.

    Supports KataGo-style playout cap randomization when mcts_config.playout_cap_prob < 1.0:
    each move randomly gets 'full' search (recorded for training) or 'cheap' search (not recorded).
    Games finish much faster while training data quality is maintained.

    Returns:
        (examples, outcome, diag): examples is list of (canonical_state, mcts_policy, value).
        outcome: 1 = player 1 won, -1 = player 2 won, 0 = draw.
        diag: dict with diagnostic values (empty if collect_diagnostics=False).
    """
    full_sims = mcts_config.num_simulations
    use_playout_cap = mcts_config.playout_cap_prob < 1.0
    cheap_sims = max(1, int(full_sims * mcts_config.playout_cap_cheap_fraction))

    # Create two MCTS configs if using playout cap
    if use_playout_cap:
        from dataclasses import replace
        cheap_config = replace(mcts_config, num_simulations=cheap_sims)
        mcts_full = MCTS(game, model, mcts_config)
        mcts_cheap = MCTS(game, model, cheap_config)
    else:
        mcts_full = MCTS(game, model, mcts_config)

    state = game.get_initial_state()
    player = 1
    trajectory = []  # (canonical_state, player, mcts_policy, is_full)
    move_count = 0
    root_values = []
    policy_entropies = []
    search_depths = []

    while True:
        canonical = game.get_canonical_state(state, player)

        # Decide full vs cheap search
        is_full = (not use_playout_cap) or (np.random.random() < mcts_config.playout_cap_prob)
        mcts = mcts_full if is_full else mcts_cheap

        # Use temperature: exploratory early, greedy late
        halflife = getattr(mcts_config, 'temp_decay_halflife', 0)
        if halflife > 0:
            # Exponential decay (KataGo-style)
            temp_start = mcts_config.temperature
            temp_end = 0.1
            temp = temp_end + (temp_start - temp_end) * (0.5 ** (move_count / halflife))
        elif move_count < mcts_config.temp_threshold:
            temp = 1.0
        else:
            temp = 0.01  # nearly greedy
        mcts.temperature = temp

        do_diag = collect_diagnostics and is_full
        pi, diag = mcts.search(state, player, collect_diagnostics=do_diag)

        # Policy target pruning (KataGo): remove low-visit moves for cleaner training targets
        prune_thresh = getattr(mcts_config, 'policy_target_pruning', 0.0)
        if prune_thresh > 0 and is_full:
            mask = pi >= prune_thresh
            if mask.any():
                pi = pi * mask
                pi_sum = pi.sum()
                if pi_sum > 0:
                    pi = pi / pi_sum

        trajectory.append((canonical.copy(), player, pi.copy(), is_full))

        if diag is not None:
            root_values.append(diag.root_value)
            policy_entropies.append(diag.policy_entropy)
            search_depths.append(diag.max_depth)

        # Sample action from policy
        action = np.random.choice(len(pi), p=pi)
        state = game.get_next_state(state, action, player)
        move_count += 1

        is_terminal, terminal_value = game.check_terminal(state, action, player)
        if is_terminal:
            # Determine game outcome from P1's perspective
            if terminal_value == 0:
                outcome = 0
            else:
                outcome = player if terminal_value > 0 else -player

            # Compute score-based value target if enabled
            score_target = None
            if score_value_weight > 0:
                raw_score = game.get_terminal_score(state, player)
                if raw_score is not None:
                    score_target = float(np.tanh(raw_score / score_value_scale))

            # Compute ownership map from terminal position (KataGo auxiliary target)
            ownership_map = None
            if collect_ownership and hasattr(game, 'get_ownership_map'):
                # Ownership from P1's perspective (we'll flip for P2 below)
                ownership_map = game.get_ownership_map(state, player=1)

            # Assign per-position values — only full-search positions if using playout cap
            examples = []
            for canonical_state, traj_player, traj_pi, traj_full in trajectory:
                if use_playout_cap and not traj_full:
                    continue  # skip cheap-search positions
                if traj_player == player:
                    v = terminal_value
                else:
                    v = -terminal_value

                # Blend binary win/loss with score-based target
                if score_target is not None:
                    sv = score_target if traj_player == player else -score_target
                    v = (1 - score_value_weight) * v + score_value_weight * sv

                if ownership_map is not None:
                    # Ownership from traj_player's perspective
                    own = ownership_map * traj_player  # flip for player -1
                    examples.append((canonical_state, traj_pi, v, own))
                else:
                    examples.append((canonical_state, traj_pi, v))

            game_diag = {}
            if collect_diagnostics:
                game_diag = {
                    'game_length': move_count,
                    'mean_root_value': float(np.mean(root_values)) if root_values else 0.0,
                    'mean_policy_entropy': float(np.mean(policy_entropies)) if policy_entropies else 0.0,
                    'mean_search_depth': float(np.mean(search_depths)) if search_depths else 0.0,
                }

            return examples, outcome, game_diag

        player = -player


def _augment_examples(game, examples, output_list):
    """Apply symmetry augmentation to examples (handles both 3-tuple and 4-tuple).

    For 4-tuple (state, pi, value, ownership): ownership is spatially augmented
    the same way as the board (rotations + reflections).
    """
    has_ownership = len(examples) > 0 and len(examples[0]) > 3

    if not has_ownership:
        # Standard 3-tuple: use get_symmetries directly
        for state, pi, v in examples:
            for sym_state, sym_pi in game.get_symmetries(state, pi):
                output_list.append((sym_state, sym_pi, v))
    else:
        # 4-tuple with ownership: augment ownership map alongside state/pi
        n = getattr(game, 'size', None)
        if n is None:
            # Fallback: no spatial augmentation for ownership
            for state, pi, v, own in examples:
                for sym_state, sym_pi in game.get_symmetries(state, pi):
                    output_list.append((sym_state, sym_pi, v, own))
            return

        for state, pi, v, own in examples:
            own_2d = own.reshape(n, n)
            sym_list = game.get_symmetries(state, pi)
            for idx, (sym_state, sym_pi) in enumerate(sym_list):
                rotation = idx // 2
                is_flipped = idx % 2 == 1
                sym_own = np.rot90(own_2d, rotation)
                if is_flipped:
                    sym_own = np.fliplr(sym_own)
                output_list.append((sym_state, sym_pi, v, sym_own.flatten().copy()))


def _worker_self_play_game(args):
    """Top-level function for pool.map: play one self-play game using process-global model."""
    from .parallel import _worker_model
    game_name, mcts_config = args

    from ..games import get_game
    game = get_game(game_name)

    examples, outcome, diag = self_play_game(
        game, _worker_model, mcts_config, collect_diagnostics=True
    )
    return examples, outcome, diag


def generate_self_play_data(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool = True,
    num_workers: int = 1,
    game_name: str | None = None,
    use_cpp: bool = False,
    score_value_weight: float = 0.0,
    score_value_scale: float = 20.0,
    collect_ownership: bool = False,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Generate training data from multiple self-play games.

    Args:
        num_workers: Number of parallel workers. 1 = sequential (no overhead).
        game_name: Game name string (required for parallel, used to reconstruct game in workers).
        use_cpp: Use C++ MCTS engine (true multi-threading, bypasses GIL).

    Returns:
        (examples, stats): examples is the training data, stats has outcomes + diagnostics.
    """
    if use_cpp:
        from .parallel import generate_cpp_parallel_self_play
        return generate_cpp_parallel_self_play(
            game, model, mcts_config, num_games,
            num_threads=max(1, num_workers), augment=augment,
            collect_ownership=collect_ownership,
        )

    if num_workers > 1 and game_name is not None:
        from .parallel import generate_gpu_parallel_self_play
        return generate_gpu_parallel_self_play(
            game, model, mcts_config, num_games,
            num_workers=num_workers, game_name=game_name, augment=augment,
            collect_ownership=collect_ownership,
        )

    return _generate_sequential(game, model, mcts_config, num_games, augment,
                                score_value_weight=score_value_weight,
                                score_value_scale=score_value_scale,
                                collect_ownership=collect_ownership)


def _generate_sequential(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool,
    score_value_weight: float = 0.0,
    score_value_scale: float = 20.0,
    collect_ownership: bool = False,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Original sequential implementation."""
    all_examples = []
    stats = SelfPlayStats()
    game_lengths = []
    root_values = []
    policy_entropies = []
    search_depths = []

    for _ in range(num_games):
        examples, outcome, diag = self_play_game(
            game, model, mcts_config, collect_diagnostics=True,
            score_value_weight=score_value_weight,
            score_value_scale=score_value_scale,
            collect_ownership=collect_ownership,
        )

        if outcome == 1:
            stats.p1_wins += 1
        elif outcome == -1:
            stats.p2_wins += 1
        else:
            stats.draws += 1

        if diag:
            game_lengths.append(diag['game_length'])
            root_values.append(diag['mean_root_value'])
            policy_entropies.append(diag['mean_policy_entropy'])
            search_depths.append(diag['mean_search_depth'])

        if augment:
            _augment_examples(game, examples, all_examples)
        else:
            all_examples.extend(examples)

    if game_lengths:
        stats.mean_game_length = float(np.mean(game_lengths))
        stats.mean_root_value = float(np.mean(root_values))
        stats.mean_policy_entropy = float(np.mean(policy_entropies))
        stats.mean_search_depth = float(np.mean(search_depths))

    return all_examples, stats


def _generate_parallel(
    game: Game,
    model,
    mcts_config: MCTSConfig,
    num_games: int,
    augment: bool,
    num_workers: int,
    game_name: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], SelfPlayStats]:
    """Parallel implementation using multiprocessing pool."""
    from .parallel import create_pool

    pool = create_pool(model, num_workers)
    try:
        args_list = [(game_name, mcts_config)] * num_games
        results = pool.map(_worker_self_play_game, args_list)
    finally:
        pool.close()
        pool.join()

    # Aggregate results
    all_examples = []
    stats = SelfPlayStats()
    game_lengths = []
    root_values = []
    policy_entropies = []
    search_depths = []

    for examples, outcome, diag in results:
        if outcome == 1:
            stats.p1_wins += 1
        elif outcome == -1:
            stats.p2_wins += 1
        else:
            stats.draws += 1

        if diag:
            game_lengths.append(diag['game_length'])
            root_values.append(diag['mean_root_value'])
            policy_entropies.append(diag['mean_policy_entropy'])
            search_depths.append(diag['mean_search_depth'])

        if augment:
            for state, pi, v in examples:
                for sym_state, sym_pi in game.get_symmetries(state, pi):
                    all_examples.append((sym_state, sym_pi, v))
        else:
            all_examples.extend(examples)

    if game_lengths:
        stats.mean_game_length = float(np.mean(game_lengths))
        stats.mean_root_value = float(np.mean(root_values))
        stats.mean_policy_entropy = float(np.mean(policy_entropies))
        stats.mean_search_depth = float(np.mean(search_depths))

    return all_examples, stats
