#pragma once
#include "go_game.h"
#include "mcts_node.h"
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <random>
#include <queue>

// MCTS config passed from Python
struct MCTSCppConfig {
    int num_simulations = 200;
    float c_puct = 1.0f;
    float dirichlet_alpha = 0.12f;
    float dirichlet_epsilon = 0.25f;
    float temperature = 1.0f;
    int temp_threshold = 15;
    int temp_decay_halflife = 0;
    int nn_batch_size = 8;
    float playout_cap_prob = 1.0f;
    float playout_cap_cheap_fraction = 0.25f;
    float fpu_reduction = 0.0f;
    float root_fpu_reduction = -1.0f;
    float c_puct_base = 0.0f;
};

// Single training example
struct Example {
    std::vector<float> state;   // canonical state (nn_input_size floats)
    std::vector<float> policy;  // MCTS policy (action_size floats)
    float value;                // game outcome from this player's perspective
};

// Stats for a batch of games
struct GameStats {
    int p1_wins = 0;
    int p2_wins = 0;
    int draws = 0;
    float mean_game_length = 0.0f;
    float mean_root_value = 0.0f;
    float mean_policy_entropy = 0.0f;
    float mean_search_depth = 0.0f;
};

// Type for the Python predict function callback
// Input: (batch_size, nn_input_size) flat float array
// Output: policies (batch_size * action_size), values (batch_size)
using PredictFn = std::function<void(const float* states, int batch_size, int nn_input_size,
                                      float* policies_out, float* values_out, int action_size)>;

// Inference request submitted by a worker to the BatchInferenceCoordinator.
// Created on the worker's stack; the coordinator reads input, writes output,
// then sets done=true. Worker spin-waits on done.
struct InferenceBatchRequest {
    const float* states;       // input: contiguous (num_states × nn_input_size) floats
    int num_states;
    float* policies_out;       // output: (num_states × action_size) floats
    float* values_out;         // output: (num_states) floats
    std::atomic<bool> done{false};
};

// Collects NN eval requests from multiple C++ worker threads and processes
// them in a single mega-batch. Only the coordinator thread calls predict_fn
// (and acquires the GIL), so workers never contend for it.
class BatchInferenceCoordinator {
public:
    BatchInferenceCoordinator(const PredictFn& fn, int nn_input_size, int action_size);

    // Worker calls this — blocks until the request has been processed.
    void submit(InferenceBatchRequest& req);

    // Inference loop — runs on a dedicated thread.
    void run();

    // Signal the loop to exit (called after all workers are done).
    void stop();

private:
    const PredictFn& predict_fn_;
    int nn_input_size_;
    int action_size_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<InferenceBatchRequest*> pending_;
    std::atomic<bool> running_{true};

    // Pre-allocated buffers for the mega-batch (grown as needed)
    std::vector<float> batch_states_;
    std::vector<float> batch_policies_;
    std::vector<float> batch_values_;
};

class SelfPlayWorker {
public:
    SelfPlayWorker(const GoGame& game, const MCTSCppConfig& config, int worker_id,
                   unsigned seed);

    // Play one complete game. Returns (examples, outcome, game_length).
    // outcome: 1 = player 1 won, -1 = player 2 won, 0 = draw.
    struct GameResult {
        std::vector<Example> examples;
        int outcome;
        int game_length;
    };
    GameResult play_game();

    // Run MCTS search from root, using the inference callback
    void run_search(MCTSNode* root, int num_sims);

    // Set the inference function (called from main thread).
    // Stores a pointer — the PredictFn must outlive the worker.
    void set_predict_fn(const PredictFn* fn) { predict_fn_ = fn; }

private:
    const GoGame& game_;
    MCTSCppConfig config_;
    int worker_id_;
    FloodFillScratch scratch_;
    NodeArena arena_;
    const PredictFn* predict_fn_ = nullptr;
    std::mt19937 rng_;

    void add_dirichlet_noise(MCTSNode* root);

    // Single NN evaluation (synchronous, for single-threaded mode)
    void evaluate_leaf(const float* canonical_state, float* policy_out, float& value_out);

    // Extract visit-count policy from root
    void extract_policy(MCTSNode* root, float temperature, float* policy_out);
};

// Main entry point: generate self-play data with multiple threads.
// predict_fn is called from an inference thread with GIL acquired.
std::pair<std::vector<Example>, GameStats>
generate_self_play_data(int board_size, int num_games, const MCTSCppConfig& config,
                        const PredictFn& predict_fn, int num_threads);
