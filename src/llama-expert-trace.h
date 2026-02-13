#ifndef LLAMA_LLAMA_EXPERT_TRACE_H
#define LLAMA_LLAMA_EXPERT_TRACE_H

#include "ggml.h"
#include "llama.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

// Forward declarations
struct llama_context;
struct llama_ubatch;

/**
 * Expert usage tracer for MoE models.
 *
 * Tracks which experts are activated during inference using llama.cpp's
 * existing callback infrastructure. Provides runtime statistics and
 * optional JSON export for analysis.
 *
 * Configuration via environment variables:
 *   LLAMA_EXPERT_TRACE_STATS=1       - Enable activation counting
 *   LLAMA_EXPERT_TRACE_LOGGING=1     - Print expert IDs during execution
 *   LLAMA_EXPERT_TRACE_OUTPUT=<file> - Export statistics to JSON file
 *
 * Usage:
 *   // Initialize (typically in llama_new_context_with_model)
 *   llama_expert_tracer::instance().init(ctx);
 *
 *   // Cleanup (typically in llama_free)
 *   llama_expert_tracer::instance().cleanup(ctx);
 *
 * Performance:
 *   - Zero overhead when disabled (env vars not set)
 *   - <1% overhead when stats enabled
 *   - 5-10% overhead when logging enabled (due to I/O)
 */
struct llama_expert_tracer {
public:
    // Configuration loaded from environment variables
    struct config {
        bool enable_stats = false;
        bool enable_logging = false;
        std::string output_file;

        // Load configuration from environment variables
        void load_from_env();
    };

    // Statistics for a single layer
    struct layer_stats {
        int layer_id;
        std::unordered_map<int, int> expert_activations;  // expert_id -> count
        int total_tokens = 0;
    };

    // Singleton instance
    static llama_expert_tracer & instance();

    // Check if tracing is enabled (call before context creation)
    // Loads configuration from environment variables
    bool is_enabled() const { return m_config.enable_stats || m_config.enable_logging; }

    // Initialize tracer configuration from environment variables
    // Must be called before context creation to set up cb_trace
    void init_config();

    // Initialize tracer for a context (called after context creation)
    // Clears any previous statistics
    void init(llama_context * ctx);

    // Cleanup and print statistics
    // Prints summary to log and exports to JSON if configured
    void cleanup(llama_context * ctx);

    // Graph build callback (optional, for early MoE detection)
    void on_graph_build(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il);

    // Eval callback (primary, for expert ID extraction)
    // Returns true for MUL_MAT_ID operations in ask phase, true always in execute phase
    bool on_eval(struct ggml_tensor * t, bool ask, llama_context * ctx);

    // Get current configuration (for testing)
    const config & get_config() const { return m_config; }

    // Get statistics (for testing/debugging)
    const std::unordered_map<int, layer_stats> & get_stats() const { return m_layer_stats; }

private:
    llama_expert_tracer() = default;
    ~llama_expert_tracer() = default;

    // Non-copyable, non-movable (singleton)
    llama_expert_tracer(const llama_expert_tracer &) = delete;
    llama_expert_tracer & operator=(const llama_expert_tracer &) = delete;

    // Internal helpers
    void record_expert_usage(int layer_id, int expert_id);
    void export_stats_json(const std::string & filename);
    int extract_layer_id(const char * tensor_name);

    // State
    config m_config;
    std::unordered_map<int, layer_stats> m_layer_stats;
    std::mutex m_mutex;
};

// C-style callback wrappers (for ggml callback system)
// These are registered with llama.cpp's callback infrastructure
void llama_expert_trace_graph_cb(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il);
bool llama_expert_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

#endif // LLAMA_LLAMA_EXPERT_TRACE_H
