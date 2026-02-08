# Post-Fetch MoE - Phase 0: Expert Usage Tracer (Object-Oriented Architecture)

**Version:** 3.0 (Object-Oriented Design)  
**Status:** Implementation Specification  
**Target:** llama.cpp MoE Optimization

---

## Purpose

Implement a well-architected, object-oriented diagnostic tool to track MoE expert activation patterns with:
- **Proper encapsulation** and separation of concerns
- **SOLID principles** adherence
- **Dependency injection** for testability
- **No global state** (state managed through context)
- Thread-safe operations with clear ownership

---

## Architecture Overview

### Core Classes

```
┌─────────────────────────────────────────────────┐
│           ExpertTracerConfig                     │
│  (Value Object - Configuration Data)            │
└─────────────────────────────────────────────────┘
                      │
                      │ owns
                      ▼
┌─────────────────────────────────────────────────┐
│           ExpertStatistics                       │
│  (Aggregate - Statistics Container)             │
│  - Thread-safe data access                      │
│  - Encapsulated mutation operations             │
└─────────────────────────────────────────────────┘
                      │
                      │ used by
                      ▼
┌─────────────────────────────────────────────────┐
│           ExpertTracer                           │
│  (Service - Main Orchestrator)                  │
│  - Record expert usage                          │
│  - Coordinate statistics gathering              │
└─────────────────────────────────────────────────┘
                      │
                      │ uses
                      ▼
┌─────────────────────────────────────────────────┐
│         IExpertStatsExporter                     │
│         (Interface)                              │
└─────────────────────────────────────────────────┘
           │                          │
           │                          │
           ▼                          ▼
┌────────────────────┐    ┌────────────────────┐
│  ConsoleExporter   │    │   JsonExporter     │
│  (Implementation)  │    │  (Implementation)  │
└────────────────────┘    └────────────────────┘
```

---

## Detailed Design

### 1. Configuration (Value Object)

**File:** `src/llama-expert-tracer-config.h`

```cpp
#pragma once

#include <string>
#include <memory>

namespace llama {
namespace expert_tracer {

/**
 * @brief Immutable configuration for expert tracing
 * 
 * Value object pattern - once constructed, configuration cannot be modified.
 * This ensures thread-safety and predictable behavior.
 */
class ExpertTracerConfig {
public:
    /**
     * @brief Builder pattern for flexible configuration construction
     */
    class Builder {
    public:
        Builder() = default;
        
        Builder& enable_stats(bool enable) {
            enable_stats_ = enable;
            return *this;
        }
        
        Builder& enable_name_logging(bool enable) {
            enable_name_logging_ = enable;
            return *this;
        }
        
        Builder& enable_per_layer(bool enable) {
            enable_per_layer_ = enable;
            return *this;
        }
        
        Builder& output_file(const std::string& filename) {
            output_file_ = filename;
            return *this;
        }
        
        /**
         * @brief Load configuration from environment variables
         */
        Builder& from_environment();
        
        /**
         * @brief Build the immutable configuration object
         */
        ExpertTracerConfig build() const {
            return ExpertTracerConfig(*this);
        }
        
    private:
        bool enable_stats_ = false;
        bool enable_name_logging_ = false;
        bool enable_per_layer_ = false;
        std::string output_file_;
        
        friend class ExpertTracerConfig;
    };
    
    // Getters (no setters - immutable)
    bool is_stats_enabled() const { return enable_stats_; }
    bool is_name_logging_enabled() const { return enable_name_logging_; }
    bool is_per_layer_enabled() const { return enable_per_layer_; }
    const std::string& get_output_file() const { return output_file_; }
    
    /**
     * @brief Check if any tracing is enabled
     */
    bool is_any_tracing_enabled() const {
        return enable_stats_ || enable_name_logging_;
    }
    
private:
    explicit ExpertTracerConfig(const Builder& builder)
        : enable_stats_(builder.enable_stats_)
        , enable_name_logging_(builder.enable_name_logging_)
        , enable_per_layer_(builder.enable_per_layer_)
        , output_file_(builder.output_file_) {}
    
    const bool enable_stats_;
    const bool enable_name_logging_;
    const bool enable_per_layer_;
    const std::string output_file_;
};

} // namespace expert_tracer
} // namespace llama
```

**Implementation:** `src/llama-expert-tracer-config.cpp`

```cpp
#include "llama-expert-tracer-config.h"
#include "log.h"
#include <cstdlib>

namespace llama {
namespace expert_tracer {

ExpertTracerConfig::Builder& ExpertTracerConfig::Builder::from_environment() {
    const char* env_stats = std::getenv("LLAMA_EXPERT_TRACE_STATS");
    if (env_stats && std::string(env_stats) == "1") {
        enable_stats_ = true;
        LOG_INF("[EXPERT-TRACE] Statistics enabled via environment\n");
    }
    
    const char* env_names = std::getenv("LLAMA_EXPERT_TRACE_NAMES");
    if (env_names && std::string(env_names) == "1") {
        enable_name_logging_ = true;
        LOG_INF("[EXPERT-TRACE] Tensor name logging enabled via environment\n");
    }
    
    const char* env_layer = std::getenv("LLAMA_EXPERT_TRACE_PER_LAYER");
    if (env_layer && std::string(env_layer) == "1") {
        enable_per_layer_ = true;
        LOG_INF("[EXPERT-TRACE] Per-layer tracking enabled via environment\n");
    }
    
    const char* env_output = std::getenv("LLAMA_EXPERT_TRACE_OUTPUT");
    if (env_output) {
        output_file_ = env_output;
        LOG_INF("[EXPERT-TRACE] Output file: %s\n", output_file_.c_str());
    }
    
    return *this;
}

} // namespace expert_tracer
} // namespace llama
```

---

### 2. Statistics Container (Aggregate)

**File:** `src/llama-expert-tracer-stats.h`

```cpp
#pragma once

#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace llama {
namespace expert_tracer {

/**
 * @brief Thread-safe container for expert activation statistics
 * 
 * Encapsulates all statistics data and provides thread-safe access.
 * Uses the Aggregate pattern to maintain consistency of related data.
 */
class ExpertStatistics {
public:
    using ExpertId = int;
    using LayerId = int;
    using ActivationCount = uint64_t;
    
    ExpertStatistics() = default;
    
    // Prevent copying to avoid mutex issues
    ExpertStatistics(const ExpertStatistics&) = delete;
    ExpertStatistics& operator=(const ExpertStatistics&) = delete;
    
    // Allow moving
    ExpertStatistics(ExpertStatistics&&) noexcept = default;
    ExpertStatistics& operator=(ExpertStatistics&&) noexcept = default;
    
    /**
     * @brief Record activation of a single expert
     * @param expert_id Expert identifier
     * Thread-safe operation
     */
    void record_activation(ExpertId expert_id);
    
    /**
     * @brief Record activation of an expert in a specific layer
     * @param layer_id Layer identifier
     * @param expert_id Expert identifier
     * Thread-safe operation
     */
    void record_layer_activation(LayerId layer_id, ExpertId expert_id);
    
    /**
     * @brief Increment token counter
     * Thread-safe operation
     */
    void increment_token_count();
    
    /**
     * @brief Get snapshot of expert activations (thread-safe copy)
     * @return Map of expert_id -> activation_count
     */
    std::unordered_map<ExpertId, ActivationCount> get_expert_activations() const;
    
    /**
     * @brief Get snapshot of per-layer expert activations (thread-safe copy)
     * @return Map of layer_id -> (expert_id -> activation_count)
     */
    std::unordered_map<LayerId, std::unordered_map<ExpertId, ActivationCount>> 
        get_layer_activations() const;
    
    /**
     * @brief Get total token count
     * @return Total number of tokens processed
     */
    uint64_t get_total_tokens() const;
    
    /**
     * @brief Get total activation count across all experts
     * @return Sum of all expert activations
     */
    uint64_t get_total_activations() const;
    
    /**
     * @brief Check if any statistics have been recorded
     * @return true if statistics exist
     */
    bool has_data() const;
    
    /**
     * @brief Clear all statistics
     * Thread-safe operation
     */
    void clear();
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<ExpertId, ActivationCount> expert_activations_;
    std::unordered_map<LayerId, std::unordered_map<ExpertId, ActivationCount>> 
        layer_activations_;
    uint64_t total_tokens_ = 0;
};

} // namespace expert_tracer
} // namespace llama
```

**Implementation:** `src/llama-expert-tracer-stats.cpp`

```cpp
#include "llama-expert-tracer-stats.h"
#include <algorithm>

namespace llama {
namespace expert_tracer {

void ExpertStatistics::record_activation(ExpertId expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_activations_[expert_id]++;
}

void ExpertStatistics::record_layer_activation(LayerId layer_id, ExpertId expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    layer_activations_[layer_id][expert_id]++;
}

void ExpertStatistics::increment_token_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_tokens_++;
}

std::unordered_map<ExpertStatistics::ExpertId, ExpertStatistics::ActivationCount> 
ExpertStatistics::get_expert_activations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return expert_activations_;
}

std::unordered_map<ExpertStatistics::LayerId, 
                   std::unordered_map<ExpertStatistics::ExpertId, 
                                     ExpertStatistics::ActivationCount>>
ExpertStatistics::get_layer_activations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return layer_activations_;
}

uint64_t ExpertStatistics::get_total_tokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_tokens_;
}

uint64_t ExpertStatistics::get_total_activations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& [expert_id, count] : expert_activations_) {
        total += count;
    }
    return total;
}

bool ExpertStatistics::has_data() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !expert_activations_.empty();
}

void ExpertStatistics::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    expert_activations_.clear();
    layer_activations_.clear();
    total_tokens_ = 0;
}

} // namespace expert_tracer
} // namespace llama
```

---

### 3. Exporter Interface (Strategy Pattern)

**File:** `src/llama-expert-tracer-exporter.h`

```cpp
#pragma once

#include "llama-expert-tracer-stats.h"
#include "llama-expert-tracer-config.h"
#include <memory>

namespace llama {
namespace expert_tracer {

/**
 * @brief Interface for exporting expert statistics
 * 
 * Strategy pattern allows different export formats without modifying core logic.
 */
class IExpertStatsExporter {
public:
    virtual ~IExpertStatsExporter() = default;
    
    /**
     * @brief Export statistics to output
     * @param stats Statistics to export
     * @param config Configuration for export details
     * @return true if export succeeded, false otherwise
     */
    virtual bool export_stats(
        const ExpertStatistics& stats,
        const ExpertTracerConfig& config
    ) = 0;
};

/**
 * @brief Export statistics to console/log
 */
class ConsoleStatsExporter : public IExpertStatsExporter {
public:
    bool export_stats(
        const ExpertStatistics& stats,
        const ExpertTracerConfig& config
    ) override;
};

/**
 * @brief Export statistics to JSON file
 */
class JsonStatsExporter : public IExpertStatsExporter {
public:
    bool export_stats(
        const ExpertStatistics& stats,
        const ExpertTracerConfig& config
    ) override;
};

/**
 * @brief Factory for creating appropriate exporters
 */
class ExporterFactory {
public:
    /**
     * @brief Create exporters based on configuration
     * @param config Configuration specifying export requirements
     * @return Vector of exporters to use
     */
    static std::vector<std::unique_ptr<IExpertStatsExporter>> 
        create_exporters(const ExpertTracerConfig& config);
};

} // namespace expert_tracer
} // namespace llama
```

**Implementation:** `src/llama-expert-tracer-exporter.cpp`

```cpp
#include "llama-expert-tracer-exporter.h"
#include "log.h"
#include <fstream>
#include <algorithm>
#include <vector>

namespace llama {
namespace expert_tracer {

// Console Exporter Implementation
bool ConsoleStatsExporter::export_stats(
    const ExpertStatistics& stats,
    const ExpertTracerConfig& config
) {
    if (!stats.has_data()) {
        return true; // Nothing to export
    }
    
    LOG_INF("\n=== Expert Usage Statistics ===\n");
    LOG_INF("Total tokens: %lu\n", stats.get_total_tokens());
    
    const uint64_t total_activations = stats.get_total_activations();
    LOG_INF("Total expert activations: %lu\n\n", total_activations);
    
    // Get and sort expert activations
    auto expert_acts = stats.get_expert_activations();
    std::vector<std::pair<int, uint64_t>> sorted_experts(
        expert_acts.begin(), expert_acts.end()
    );
    std::sort(sorted_experts.begin(), sorted_experts.end());
    
    for (const auto& [expert_id, count] : sorted_experts) {
        const double percentage = (100.0 * count) / total_activations;
        LOG_INF("  Expert %3d: %8lu activations (%.2f%%)\n",
                expert_id, count, percentage);
    }
    
    // Per-layer breakdown if enabled
    if (config.is_per_layer_enabled()) {
        LOG_INF("\nPer-layer breakdown:\n");
        auto layer_acts = stats.get_layer_activations();
        
        std::vector<int> layer_ids;
        for (const auto& [layer_id, _] : layer_acts) {
            layer_ids.push_back(layer_id);
        }
        std::sort(layer_ids.begin(), layer_ids.end());
        
        for (int layer_id : layer_ids) {
            LOG_INF("  Layer %d:\n", layer_id);
            const auto& expert_counts = layer_acts[layer_id];
            
            std::vector<std::pair<int, uint64_t>> sorted_layer_experts(
                expert_counts.begin(), expert_counts.end()
            );
            std::sort(sorted_layer_experts.begin(), sorted_layer_experts.end());
            
            for (const auto& [expert_id, count] : sorted_layer_experts) {
                LOG_INF("    Expert %3d: %8lu\n", expert_id, count);
            }
        }
    }
    
    LOG_INF("\n");
    return true;
}

// JSON Exporter Implementation
bool JsonStatsExporter::export_stats(
    const ExpertStatistics& stats,
    const ExpertTracerConfig& config
) {
    const std::string filename = config.get_output_file();
    if (filename.empty()) {
        LOG_ERR("[EXPERT-TRACE] No output file specified for JSON export\n");
        return false;
    }
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        LOG_ERR("[EXPERT-TRACE] Failed to open %s for writing\n", filename.c_str());
        return false;
    }
    
    out << "{\n";
    out << "  \"total_tokens\": " << stats.get_total_tokens() << ",\n";
    out << "  \"total_activations\": " << stats.get_total_activations() << ",\n";
    out << "  \"expert_activations\": {\n";
    
    auto expert_acts = stats.get_expert_activations();
    std::vector<std::pair<int, uint64_t>> sorted_experts(
        expert_acts.begin(), expert_acts.end()
    );
    std::sort(sorted_experts.begin(), sorted_experts.end());
    
    for (size_t i = 0; i < sorted_experts.size(); ++i) {
        const auto& [expert_id, count] = sorted_experts[i];
        out << "    \"" << expert_id << "\": " << count;
        if (i < sorted_experts.size() - 1) {
            out << ",";
        }
        out << "\n";
    }
    out << "  }";
    
    // Per-layer data if enabled
    if (config.is_per_layer_enabled()) {
        out << ",\n  \"per_layer\": {\n";
        
        auto layer_acts = stats.get_layer_activations();
        std::vector<int> layer_ids;
        for (const auto& [layer_id, _] : layer_acts) {
            layer_ids.push_back(layer_id);
        }
        std::sort(layer_ids.begin(), layer_ids.end());
        
        for (size_t i = 0; i < layer_ids.size(); ++i) {
            const int layer_id = layer_ids[i];
            out << "    \"" << layer_id << "\": {\n";
            
            const auto& expert_counts = layer_acts[layer_id];
            std::vector<std::pair<int, uint64_t>> sorted_layer_experts(
                expert_counts.begin(), expert_counts.end()
            );
            std::sort(sorted_layer_experts.begin(), sorted_layer_experts.end());
            
            for (size_t j = 0; j < sorted_layer_experts.size(); ++j) {
                const auto& [expert_id, count] = sorted_layer_experts[j];
                out << "      \"" << expert_id << "\": " << count;
                if (j < sorted_layer_experts.size() - 1) {
                    out << ",";
                }
                out << "\n";
            }
            
            out << "    }";
            if (i < layer_ids.size() - 1) {
                out << ",";
            }
            out << "\n";
        }
        out << "  }";
    }
    
    out << "\n}\n";
    out.close();
    
    LOG_INF("[EXPERT-TRACE] Statistics exported to %s\n", filename.c_str());
    return true;
}

// Factory Implementation
std::vector<std::unique_ptr<IExpertStatsExporter>> 
ExporterFactory::create_exporters(const ExpertTracerConfig& config) {
    std::vector<std::unique_ptr<IExpertStatsExporter>> exporters;
    
    // Always add console exporter if stats are enabled
    if (config.is_stats_enabled()) {
        exporters.push_back(std::make_unique<ConsoleStatsExporter>());
    }
    
    // Add JSON exporter if output file is specified
    if (!config.get_output_file().empty()) {
        exporters.push_back(std::make_unique<JsonStatsExporter>());
    }
    
    return exporters;
}

} // namespace expert_tracer
} // namespace llama
```

---

### 4. Main Tracer Service (Facade/Service Pattern)

**File:** `src/llama-expert-tracer.h`

```cpp
#pragma once

#include "llama.h"
#include "ggml.h"
#include "llama-expert-tracer-config.h"
#include "llama-expert-tracer-stats.h"
#include "llama-expert-tracer-exporter.h"
#include <memory>
#include <vector>

namespace llama {
namespace expert_tracer {

/**
 * @brief Main expert tracing service
 * 
 * Facade pattern that coordinates configuration, statistics gathering,
 * and export operations. This is the primary interface for the rest of
 * the codebase to interact with expert tracing.
 */
class ExpertTracer {
public:
    /**
     * @brief Construct tracer with custom configuration
     * @param config Immutable configuration object
     */
    explicit ExpertTracer(ExpertTracerConfig config);
    
    /**
     * @brief Construct tracer with configuration from environment
     */
    static std::unique_ptr<ExpertTracer> create_from_environment();
    
    /**
     * @brief Record usage of an expert
     * @param model Model containing the expert
     * @param tensor Expert tensor being used
     * @param layer_id Layer identifier
     * @param expert_id Expert identifier
     * 
     * This is a lightweight operation with early return if tracing is disabled.
     */
    void record_expert_usage(
        const llama_model* model,
        const ggml_tensor* tensor,
        int layer_id,
        int expert_id
    );
    
    /**
     * @brief Export all statistics using configured exporters
     * @return true if all exports succeeded
     */
    bool export_statistics();
    
    /**
     * @brief Get read-only access to statistics
     * @return Reference to statistics object
     */
    const ExpertStatistics& get_statistics() const { return statistics_; }
    
    /**
     * @brief Get configuration
     * @return Reference to configuration object
     */
    const ExpertTracerConfig& get_config() const { return config_; }
    
    /**
     * @brief Check if tracer is active
     * @return true if any tracing is enabled
     */
    bool is_active() const {
        return config_.is_any_tracing_enabled();
    }
    
    /**
     * @brief Clear all accumulated statistics
     */
    void clear_statistics();
    
private:
    const ExpertTracerConfig config_;
    ExpertStatistics statistics_;
    std::vector<std::unique_ptr<IExpertStatsExporter>> exporters_;
    
    void log_tensor_usage(const ggml_tensor* tensor, int layer_id, int expert_id) const;
};

} // namespace expert_tracer
} // namespace llama
```

**Implementation:** `src/llama-expert-tracer.cpp`

```cpp
#include "llama-expert-tracer.h"
#include "log.h"

namespace llama {
namespace expert_tracer {

ExpertTracer::ExpertTracer(ExpertTracerConfig config)
    : config_(std::move(config))
    , statistics_()
    , exporters_(ExporterFactory::create_exporters(config_))
{
    if (config_.is_any_tracing_enabled()) {
        LOG_INF("[EXPERT-TRACE] Tracer initialized\n");
    }
}

std::unique_ptr<ExpertTracer> ExpertTracer::create_from_environment() {
    auto config = ExpertTracerConfig::Builder()
        .from_environment()
        .build();
    
    return std::make_unique<ExpertTracer>(std::move(config));
}

void ExpertTracer::record_expert_usage(
    const llama_model* model,
    const ggml_tensor* tensor,
    int layer_id,
    int expert_id
) {
    // Early return for performance when tracing is disabled
    if (!config_.is_any_tracing_enabled()) {
        return;
    }
    
    // Log tensor name if enabled
    if (config_.is_name_logging_enabled() && tensor) {
        log_tensor_usage(tensor, layer_id, expert_id);
    }
    
    // Update statistics if enabled
    if (config_.is_stats_enabled()) {
        statistics_.record_activation(expert_id);
        
        if (config_.is_per_layer_enabled()) {
            statistics_.record_layer_activation(layer_id, expert_id);
        }
    }
}

bool ExpertTracer::export_statistics() {
    if (!statistics_.has_data()) {
        return true; // No data to export is success
    }
    
    bool all_succeeded = true;
    for (auto& exporter : exporters_) {
        if (!exporter->export_stats(statistics_, config_)) {
            all_succeeded = false;
            LOG_WRN("[EXPERT-TRACE] Export failed for one exporter\n");
        }
    }
    
    return all_succeeded;
}

void ExpertTracer::clear_statistics() {
    statistics_.clear();
    LOG_INF("[EXPERT-TRACE] Statistics cleared\n");
}

void ExpertTracer::log_tensor_usage(
    const ggml_tensor* tensor,
    int layer_id,
    int expert_id
) const {
    if (tensor && tensor->name) {
        LOG_DBG("[EXPERT-TRACE] Layer %d, Expert %d: %s\n",
                layer_id, expert_id, tensor->name);
    }
}

} // namespace expert_tracer
} // namespace llama
```

---

## Integration with llama.cpp

### Context Integration

**File:** `src/llama.cpp` (modifications)

```cpp
#include "llama-expert-tracer.h"

// Add to llama_context struct
struct llama_context {
    // ... existing fields ...
    
    // Expert tracer (optional, nullptr if disabled)
    std::unique_ptr<llama::expert_tracer::ExpertTracer> expert_tracer;
    
    // ... rest of fields ...
};

// In llama_new_context_with_model()
struct llama_context * llama_new_context_with_model(
    llama_model * model,
    struct llama_context_params params
) {
    // ... existing initialization ...
    
    // Initialize expert tracer from environment
    ctx->expert_tracer = llama::expert_tracer::ExpertTracer::create_from_environment();
    
    if (ctx->expert_tracer && ctx->expert_tracer->is_active()) {
        LOG_INF("Expert tracing enabled for this context\n");
    }
    
    // ... rest of initialization ...
    
    return ctx;
}

// In llama_free_context()
void llama_free_context(struct llama_context * ctx) {
    if (!ctx) {
        return;
    }
    
    // Export statistics before cleanup
    if (ctx->expert_tracer && ctx->expert_tracer->is_active()) {
        ctx->expert_tracer->export_statistics();
    }
    
    // ... rest of cleanup ...
    
    delete ctx;
}
```

### Graph Integration

**File:** `src/llama-graph.cpp` (modifications)

```cpp
#include "llama-expert-tracer.h"

// In build_moe_ffn(), after expert selection
static ggml_tensor * build_moe_ffn(
    /* ... parameters ... */
) {
    // ... existing routing code ...
    
    // Record expert usage if tracer is active
    if (ctx->expert_tracer && ctx->expert_tracer->is_active()) {
        for (int i = 0; i < n_active_experts; i++) {
            const int expert_id = selected_experts[i];
            
            // Record for each expert tensor
            ctx->expert_tracer->record_expert_usage(
                &model, layer.ffn_gate_exps[expert_id], il, expert_id);
            ctx->expert_tracer->record_expert_usage(
                &model, layer.ffn_down_exps[expert_id], il, expert_id);
            ctx->expert_tracer->record_expert_usage(
                &model, layer.ffn_up_exps[expert_id], il, expert_id);
        }
    }
    
    // ... rest of function ...
}
```

---

## CMake Configuration

**File:** `src/CMakeLists.txt`

```cmake
# Add expert tracer sources
set(LLAMA_EXPERT_TRACER_SOURCES
    llama-expert-tracer-config.cpp
    llama-expert-tracer-stats.cpp
    llama-expert-tracer-exporter.cpp
    llama-expert-tracer.cpp
)

set(SOURCES
    # ... existing sources ...
    ${LLAMA_EXPERT_TRACER_SOURCES}
    # ... other sources ...
)
```

---

## Testing Strategy

### Unit Tests

```cpp
// test/test-expert-tracer.cpp

#include "llama-expert-tracer-config.h"
#include "llama-expert-tracer-stats.h"
#include "llama-expert-tracer.h"
#include <cassert>
#include <thread>
#include <vector>

namespace test {

void test_config_builder() {
    auto config = llama::expert_tracer::ExpertTracerConfig::Builder()
        .enable_stats(true)
        .enable_per_layer(true)
        .output_file("test.json")
        .build();
    
    assert(config.is_stats_enabled());
    assert(config.is_per_layer_enabled());
    assert(config.get_output_file() == "test.json");
}

void test_statistics_thread_safety() {
    llama::expert_tracer::ExpertStatistics stats;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&stats, i]() {
            for (int j = 0; j < 1000; ++j) {
                stats.record_activation(i);
                stats.record_layer_activation(0, i);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    assert(stats.get_total_activations() == 10000);
}

void test_tracer_disabled() {
    auto config = llama::expert_tracer::ExpertTracerConfig::Builder().build();
    llama::expert_tracer::ExpertTracer tracer(std::move(config));
    
    assert(!tracer.is_active());
    
    // Should be no-op
    tracer.record_expert_usage(nullptr, nullptr, 0, 0);
    assert(!tracer.get_statistics().has_data());
}

void test_tracer_enabled() {
    auto config = llama::expert_tracer::ExpertTracerConfig::Builder()
        .enable_stats(true)
        .build();
    llama::expert_tracer::ExpertTracer tracer(std::move(config));
    
    assert(tracer.is_active());
    
    tracer.record_expert_usage(nullptr, nullptr, 0, 5);
    tracer.record_expert_usage(nullptr, nullptr, 0, 7);
    tracer.record_expert_usage(nullptr, nullptr, 1, 5);
    
    assert(tracer.get_statistics().has_data());
    assert(tracer.get_statistics().get_total_activations() == 3);
}

} // namespace test

int main() {
    test::test_config_builder();
    test::test_statistics_thread_safety();
    test::test_tracer_disabled();
    test::test_tracer_enabled();
    
    return 0;
}
```

---

## Design Principles Applied

### 1. **Single Responsibility Principle (SRP)**
- `ExpertTracerConfig`: Only handles configuration
- `ExpertStatistics`: Only manages statistics data
- `ExpertTracer`: Only coordinates tracing operations
- Exporters: Each handles one export format

### 2. **Open/Closed Principle (OCP)**
- New export formats can be added without modifying existing code
- Just implement `IExpertStatsExporter` interface

### 3. **Liskov Substitution Principle (LSP)**
- All exporters are interchangeable through the interface
- Factory can create any combination of exporters

### 4. **Interface Segregation Principle (ISP)**
- Small, focused interfaces (`IExpertStatsExporter`)
- Clients depend only on what they use

### 5. **Dependency Inversion Principle (DIP)**
- High-level `ExpertTracer` depends on `IExpertStatsExporter` abstraction
- Concrete exporters depend on the same abstraction

### Additional Patterns

- **Builder Pattern**: For flexible configuration construction
- **Strategy Pattern**: For interchangeable export strategies
- **Factory Pattern**: For creating appropriate exporters
- **Facade Pattern**: `ExpertTracer` simplifies complex subsystem
- **Value Object**: `ExpertTracerConfig` is immutable

---

## Benefits Over Original Design

| Aspect | Original | Improved OO Design |
|--------|----------|-------------------|
| **Global State** | `g_expert_stats` global | Owned by `llama_context` |
| **Testability** | Hard to test (globals) | Easy to test (DI) |
| **Thread Safety** | Manual mutex management | Encapsulated in classes |
| **Extensibility** | Hard to add exporters | Just implement interface |
| **Coupling** | Tight coupling | Loose coupling via interfaces |
| **Initialization** | Scattered | Centralized in constructors |
| **Memory Safety** | Manual management | RAII via smart pointers |
| **Code Organization** | Single file | Logical separation |

---

## Performance Considerations

1. **Early Return**: `is_active()` check before any work
2. **Move Semantics**: Configuration passed by move
3. **Lock Granularity**: Minimal critical sections
4. **Lazy Initialization**: Exporters created only when needed
5. **Zero Overhead When Disabled**: No virtual calls when inactive

---

## Future Extensions

### Pluggable Exporters
```cpp
ctx->expert_tracer->add_exporter(
    std::make_unique<PrometheusExporter>("localhost:9090")
);
```

### Real-time Monitoring
```cpp
class RealtimeExporter : public IExpertStatsExporter {
    void export_stats(...) override {
        // Stream to monitoring system
    }
};
```

### Custom Statistics
```cpp
class ExtendedStatistics : public ExpertStatistics {
    void record_latency(ExpertId id, std::chrono::nanoseconds latency);
};
```

---

## File Checklist

- [ ] `src/llama-expert-tracer-config.h`
- [ ] `src/llama-expert-tracer-config.cpp`
- [ ] `src/llama-expert-tracer-stats.h`
- [ ] `src/llama-expert-tracer-stats.cpp`
- [ ] `src/llama-expert-tracer-exporter.h`
- [ ] `src/llama-expert-tracer-exporter.cpp`
- [ ] `src/llama-expert-tracer.h`
- [ ] `src/llama-expert-tracer.cpp`
- [ ] Modify `src/llama.cpp` (context integration)
- [ ] Modify `src/llama-graph.cpp` (usage recording)
- [ ] Modify `src/CMakeLists.txt` (build config)
- [ ] Create `test/test-expert-tracer.cpp` (unit tests)

---

*Version 3.0 - Object-Oriented Architecture | Last Updated: 2026-02-08*
