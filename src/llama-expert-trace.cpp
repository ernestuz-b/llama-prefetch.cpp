#include "llama-expert-trace.h"
#include "llama-impl.h"  // Internal logging: LLAMA_LOG_INFO, LLAMA_LOG_DEBUG, LLAMA_LOG_WARN
#include "llama-context.h"
#include <cstdlib>
#include <cstring>
#include <regex>
#include <fstream>
#include <sstream>

//
// Configuration
//

void llama_expert_tracer::config::load_from_env() {
    const char* env_stats = std::getenv("LLAMA_EXPERT_TRACE_STATS");
    enable_stats = (env_stats && std::strcmp(env_stats, "1") == 0);

    const char* env_logging = std::getenv("LLAMA_EXPERT_TRACE_LOGGING");
    enable_logging = (env_logging && std::strcmp(env_logging, "1") == 0);

    const char* env_output = std::getenv("LLAMA_EXPERT_TRACE_OUTPUT");
    if (env_output) {
        output_file = env_output;
    }
}

//
// Singleton
//

llama_expert_tracer & llama_expert_tracer::instance() {
    static llama_expert_tracer instance;
    return instance;
}

//
// Initialization / Cleanup
//

void llama_expert_tracer::init(llama_context * ctx) {
    // Load configuration from environment variables
    m_config.load_from_env();

    // Early return if tracing disabled (zero overhead)
    if (!m_config.enable_stats && !m_config.enable_logging) {
        return;
    }

    LLAMA_LOG_INFO("Expert tracing enabled (stats=%d, logging=%d)\n",
            m_config.enable_stats, m_config.enable_logging);

    // Clear any previous statistics
    m_layer_stats.clear();

    // Register eval callback, ctx->get_sched() returns the ggml_backend_sched pointer
    ggml_backend_sched_set_eval_callback(
        ctx->get_sched(),
        llama_expert_trace_eval_cb,
        ctx
    );
}

void llama_expert_tracer::cleanup(llama_context * ctx) {
    // Early return if stats disabled
    if (!m_config.enable_stats) {
        return;
    }

    // Print statistics to log
    LLAMA_LOG_INFO("\n Expert Usage Statistics: \n");
    for (const auto & [layer_id, stats] : m_layer_stats) {
        LLAMA_LOG_INFO("Layer %d (%d tokens):\n", layer_id, stats.total_tokens);

        // Sort experts by activation count (descending)
        std::vector<std::pair<int, int>> sorted_experts(
            stats.expert_activations.begin(),
            stats.expert_activations.end()
        );
        std::sort(sorted_experts.begin(), sorted_experts.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });

        for (const auto & [expert_id, count] : sorted_experts) {
            float percentage = 100.0f * count / stats.total_tokens;
            LLAMA_LOG_INFO("  Expert %3d: %5d activations (%.1f%%)\n",
                    expert_id, count, percentage);
        }
    }

    // Export to JSON if configured
    if (!m_config.output_file.empty()) {
        export_stats_json(m_config.output_file);
        LLAMA_LOG_INFO("Statistics exported to: %s\n", m_config.output_file.c_str());
    }
}

//
// Callbacks
//

void llama_expert_tracer::on_graph_build(
    const llama_ubatch & ubatch,
    ggml_tensor * cur,
    const char * name,
    int il
) {
    // Early return if logging disabled
    if (!m_config.enable_logging) {
        return;
    }

    // Detect MoE layers by checking for ffn_moe_topk tensor
    if (std::strcmp(name, "ffn_moe_topk") == 0) {
        LLAMA_LOG_DEBUG("[EXPERT-TRACE] Layer %d: MoE layer detected (tensor '%s')\n",
                il, name);
    }
}

void llama_expert_tracer::on_eval(struct ggml_tensor * t, bool ask, llama_context * ctx) {
    // In "ask" phase, return true for operations we want to intercept
    if (ask) {
        // We want to see MUL_MAT_ID operations (expert computation)
        // Note: This function is called from llama_expert_trace_eval_cb which returns bool
        // The actual return value is handled by the callback wrapper
        return;
    }

    // In "execute" phase, process the operation
    if (t->op != GGML_OP_MUL_MAT_ID) {
        return;
    }

    // Extract expert IDs from the operation
    // For GGML_OP_MUL_MAT_ID, the ids tensor is src[2]
    const ggml_tensor * ids = t->src[2];
    if (!ids) {
        return;
    }

    // Extract layer ID from tensor name
    // Tensor names are like "blk.5.ffn_moe_..."
    int layer_id = extract_layer_id(t->name);
    if (layer_id < 0) {
        // Could not parse layer ID, skip
        return;
    }

    // Copy expert IDs to host (small tensor, safe to copy)
    std::vector<int32_t> expert_ids(ggml_nelements(ids));
    ggml_backend_tensor_get(ids, expert_ids.data(), 0, ggml_nbytes(ids));

    // Update statistics
    for (int32_t expert_id : expert_ids) {
        record_expert_usage(layer_id, expert_id);
    }

    // Logging (if enabled)
    if (m_config.enable_logging) {
        std::stringstream ss;
        ss << "[EXPERT-TRACE] Layer " << layer_id << ": Experts [";
        for (size_t i = 0; i < expert_ids.size(); i++) {
            ss << expert_ids[i];
            if (i + 1 < expert_ids.size()) {
                ss << ", ";
            }
        }
        ss << "]\n";
        LLAMA_LOG_DEBUG("%s", ss.str().c_str());
    }
}

//
// Internal Helpers
//

void llama_expert_tracer::record_expert_usage(int layer_id, int expert_id) {
    // Early return if stats disabled
    if (!m_config.enable_stats) {
        return;
    }

    // Thread-safe update with mutex
    std::lock_guard<std::mutex> lock(m_mutex);

    auto & stats = m_layer_stats[layer_id];
    stats.layer_id = layer_id;
    stats.expert_activations[expert_id]++;
    stats.total_tokens++;
}

int llama_expert_tracer::extract_layer_id(const char * tensor_name) {
    // Extract layer ID from tensor name like "blk.5.ffn_moe_..."
    // Returns -1 if parsing fails

    std::string name(tensor_name);
    std::regex pattern(R"(blk\.(\d+)\.)");
    std::smatch match;

    if (std::regex_search(name, match, pattern)) {
        return std::stoi(match[1].str());
    }

    return -1;
}

void llama_expert_tracer::export_stats_json(const std::string & filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        LLAMA_LOG_WARN("Failed to open output file: %s\n", filename.c_str());
        return;
    }

    file << "{\n";
    file << "  \"layers\": [\n";

    bool first_layer = true;
    for (const auto & [layer_id, stats] : m_layer_stats) {
        if (!first_layer) {
            file << ",\n";
        }
        first_layer = false;

        file << "    {\n";
        file << "      \"layer_id\": " << layer_id << ",\n";
        file << "      \"total_tokens\": " << stats.total_tokens << ",\n";
        file << "      \"experts\": [\n";

        bool first_expert = true;
        for (const auto & [expert_id, count] : stats.expert_activations) {
            if (!first_expert) {
                file << ",\n";
            }
            first_expert = false;

            float percentage = 100.0f * count / stats.total_tokens;
            file << "        {\"expert_id\": " << expert_id
                 << ", \"activations\": " << count
                 << ", \"percentage\": " << percentage << "}";
        }

        file << "\n      ]\n";
        file << "    }";
    }

    file << "\n  ]\n";
    file << "}\n";

    file.close();
}

//
// C-style Callback Wrappers
//

void llama_expert_trace_graph_cb(
    const llama_ubatch & ubatch,
    ggml_tensor * cur,
    const char * name,
    int il
) {
    llama_expert_tracer::instance().on_graph_build(ubatch, cur, name, il);
}

bool llama_expert_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    llama_context * ctx = static_cast<llama_context *>(user_data);
    llama_expert_tracer::instance().on_eval(t, ask, ctx);
    return true;
}
