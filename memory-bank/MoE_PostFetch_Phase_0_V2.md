# Post-Fetch MoE - Phase 0: Expert Usage Tracer

**Version:** 2.0 (Condensed & Grounded)  
**Status:** Implementation Specification  
**Target:** llama.cpp MoE Optimization

---

## Purpose

Implement a lightweight diagnostic tool to track MoE expert activation patterns:
- Which experts activate during inference
- Activation frequency per expert/layer
- Optional tensor name logging for debugging
-
**Key Design:** Use llama.cpp's existing logging infrastructure (`common/log.h`) for thread-safe, minimal-overhead instrumentation.

## Key Constraints

1. **Thread Safety:** All stats updates use `std::mutex`
2. **Zero Impact When Disabled:** Early returns if no tracing enabled
3. **Use Existing Infrastructure:** Leverage `common/log.h` macros
4. **Minimal Allocation:** Reuse existing tensor names (no copies)
5. **C++17 Standard:** Per llama.cpp conventions
6. **Use good Object Oriented design practices and patterns:** Code should be contained into classes as much as possible.

---

## Configuration

Control via environment variables:

```bash
LLAMA_EXPERT_TRACE_STATS=1        # Enable activation counting
LLAMA_EXPERT_TRACE_NAMES=1        # Print tensor names
LLAMA_EXPERT_TRACE_PER_LAYER=1    # Per-layer statistics
LLAMA_EXPERT_TRACE_OUTPUT=file    # Export to JSON
```

---

## Integration Points

| Hook Location | File | Function | Purpose |
|---------------|------|----------|---------|
| **Recording** | `src/llama-graph.cpp` | `build_moe_ffn()` | Track expert usage after routing |
| **Initialization** | `src/llama.cpp` | `llama_new_context_with_model()` | Read config at startup |
| **Cleanup** | `src/llama.cpp` | `llama_free_context()` | Print/export statistics |

---

## Implementation

### File Structure

```
src/llama-expert-tracer.h      # Interface (NEW)
src/llama-expert-tracer.cpp    # Implementation (NEW)
src/llama-graph.cpp            # Add tracer hook (MODIFY)
src/llama.cpp                  # Init/cleanup (MODIFY)
src/CMakeLists.txt             # Add sources (MODIFY)
```

### Step 1: Header (`src/llama-expert-tracer.h`)

```cpp
#pragma once

#include "llama.h"
#include <unordered_map>
#include <mutex>
#include <string>

struct expert_tracer_config {
    bool enable_stats = false;
    bool enable_name_logging = false;
    bool enable_per_layer = false;
    std::string output_file;
};

struct expert_tracer_stats {
    std::mutex stats_mutex;
    std::unordered_map<int, uint64_t> expert_activations;
    std::unordered_map<int, std::unordered_map<int, uint64_t>> layer_expert_activations;
    uint64_t total_tokens = 0;
    expert_tracer_config config;
};

// Global instance
extern expert_tracer_stats g_expert_stats;

// API
void expert_tracer_init();
void expert_tracer_record_usage(
    const llama_model * model,
    const struct ggml_tensor * tensor,
    int layer_id,
    int expert_id
);
void expert_tracer_print_stats();
void expert_tracer_export_json(const std::string & filename);
```

### Step 2: Implementation (`src/llama-expert-tracer.cpp`)

```cpp
#include "llama-expert-tracer.h"
#include "log.h"
#include <cstdlib>
#include <fstream>

expert_tracer_stats g_expert_stats;

void expert_tracer_init() {
    const char* env_stats = std::getenv("LLAMA_EXPERT_TRACE_STATS");
    g_expert_stats.config.enable_stats = 
        (env_stats != nullptr && std::string(env_stats) == "1");
    
    const char* env_names = std::getenv("LLAMA_EXPERT_TRACE_NAMES");
    g_expert_stats.config.enable_name_logging = 
        (env_names != nullptr && std::string(env_names) == "1");
    
    const char* env_layer = std::getenv("LLAMA_EXPERT_TRACE_PER_LAYER");
    g_expert_stats.config.enable_per_layer = 
        (env_layer != nullptr && std::string(env_layer) == "1");
    
    const char* env_output = std::getenv("LLAMA_EXPERT_TRACE_OUTPUT");
    if (env_output != nullptr) {
        g_expert_stats.config.output_file = env_output;
    }
    
    if (g_expert_stats.config.enable_stats) {
        LOG_INF("[EXPERT-TRACE] Statistics enabled\n");
    }
    if (g_expert_stats.config.enable_name_logging) {
        LOG_INF("[EXPERT-TRACE] Tensor name logging enabled\n");
    }
}

void expert_tracer_record_usage(
    const llama_model * model,
    const struct ggml_tensor * tensor,
    int layer_id,
    int expert_id
) {
    if (!g_expert_stats.config.enable_stats && 
        !g_expert_stats.config.enable_name_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_expert_stats.stats_mutex);
    
    // Log tensor name if enabled
    if (g_expert_stats.config.enable_name_logging && tensor) {
        LOG_DBG("[EXPERT-TRACE] Layer %d, Expert %d: %s\n", 
                layer_id, expert_id, tensor->name);
    }
    
    // Update statistics if enabled
    if (g_expert_stats.config.enable_stats) {
        g_expert_stats.expert_activations[expert_id]++;
        
        if (g_expert_stats.config.enable_per_layer) {
            g_expert_stats.layer_expert_activations[layer_id][expert_id]++;
        }
    }
}

void expert_tracer_print_stats() {
    std::lock_guard<std::mutex> lock(g_expert_stats.stats_mutex);
    
    if (g_expert_stats.expert_activations.empty()) {
        return;
    }
    
    LOG_INF("\n=== Expert Usage Statistics ===\n");
    LOG_INF("Total tokens: %lu\n", g_expert_stats.total_tokens);
    
    uint64_t total_activations = 0;
    for (const auto & [expert_id, count] : g_expert_stats.expert_activations) {
        total_activations += count;
    }
    
    LOG_INF("Total expert activations: %lu\n\n", total_activations);
    
    for (const auto & [expert_id, count] : g_expert_stats.expert_activations) {
        double percentage = (100.0 * count) / total_activations;
        LOG_INF("  Expert %3d: %8lu activations (%.2f%%)\n",
                expert_id, count, percentage);
    }
    
    if (g_expert_stats.config.enable_per_layer) {
        LOG_INF("\nPer-layer breakdown:\n");
        for (const auto & [layer, expert_counts] : 
             g_expert_stats.layer_expert_activations) {
            LOG_INF("  Layer %d:\n", layer);
            for (const auto & [expert_id, count] : expert_counts) {
                LOG_INF("    Expert %3d: %8lu\n", expert_id, count);
            }
        }
    }
}

void expert_tracer_export_json(const std::string & filename) {
    std::lock_guard<std::mutex> lock(g_expert_stats.stats_mutex);
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        LOG_ERR("[EXPERT-TRACE] Failed to open %s\n", filename.c_str());
        return;
    }
    
    out << "{\n";
    out << "  \"total_tokens\": " << g_expert_stats.total_tokens << ",\n";
    out << "  \"expert_activations\": {\n";
    
    bool first = true;
    for (const auto & [expert_id, count] : g_expert_stats.expert_activations) {
        if (!first) out << ",\n";
        out << "    \"" << expert_id << "\": " << count;
        first = false;
    }
    out << "\n  }";
    
    if (g_expert_stats.config.enable_per_layer) {
        out << ",\n  \"per_layer\": {\n";
        first = true;
        for (const auto & [layer, expert_counts] : 
             g_expert_stats.layer_expert_activations) {
            if (!first) out << ",\n";
            out << "    \"" << layer << "\": {\n";
            bool first_expert = true;
            for (const auto & [expert_id, count] : expert_counts) {
                if (!first_expert) out << ",\n";
                out << "      \"" << expert_id << "\": " << count;
                first_expert = false;
            }
            out << "\n    }";
            first = false;
        }
        out << "\n  }";
    }
    
    out << "\n}\n";
    out.close();
    
    LOG_INF("[EXPERT-TRACE] Statistics exported to %s\n", filename.c_str());
}
```

### Step 3: Hook in `src/llama-graph.cpp`

In `build_moe_ffn()`, after routing selects experts:

```cpp
// Add at top of file
#include "llama-expert-tracer.h"

// In build_moe_ffn(), after routing code:
if (g_expert_stats.config.enable_stats || 
    g_expert_stats.config.enable_name_logging) {
    for (int i = 0; i < n_active_experts; i++) {
        int expert_id = selected_experts[i];
        
        // Record usage for each expert tensor
        expert_tracer_record_usage(&model, 
            layer.ffn_gate_exps[expert_id], il, expert_id);
        expert_tracer_record_usage(&model, 
            layer.ffn_down_exps[expert_id], il, expert_id);
        expert_tracer_record_usage(&model, 
            layer.ffn_up_exps[expert_id], il, expert_id);
    }
}
```

### Step 4: Initialize/Cleanup in `src/llama.cpp`

```cpp
// Add at top of file
#include "llama-expert-tracer.h"

// In llama_new_context_with_model(), after model loading:
expert_tracer_init();

// In llama_free_context(), before cleanup:
if (g_expert_stats.config.enable_stats) {
    expert_tracer_print_stats();
    if (!g_expert_stats.config.output_file.empty()) {
        expert_tracer_export_json(g_expert_stats.config.output_file);
    }
}
```

### Step 5: Update `src/CMakeLists.txt`

```cmake
set(SOURCES
    # ... existing sources ...
    llama-expert-tracer.cpp
    # ... other sources ...
)
```

---

## Testing

### Unit Tests

```bash
# 1. Basic statistics tracking
LLAMA_EXPERT_TRACE_STATS=1 \
./llama-cli -m mixtral-8x7b.gguf -p "test"

# 2. Tensor name logging
LLAMA_EXPERT_TRACE_NAMES=1 \
./llama-cli -m mixtral-8x7b.gguf -p "test"

# 3. Per-layer tracking
LLAMA_EXPERT_TRACE_STATS=1 \
LLAMA_EXPERT_TRACE_PER_LAYER=1 \
./llama-cli -m mixtral-8x7b.gguf -p "test"

# 4. JSON export
LLAMA_EXPERT_TRACE_STATS=1 \
LLAMA_EXPERT_TRACE_OUTPUT=stats.json \
./llama-cli -m mixtral-8x7b.gguf -p "test"

# 5. Validate JSON
python3 -c "import json; json.load(open('stats.json'))"

# 6. Thread safety test
LLAMA_EXPERT_TRACE_STATS=1 \
./llama-cli -m mixtral-8x7b.gguf -p "test" -t 8
```

### Integration Tests

Test with target models:
- Mixtral-8x7B (32 experts, top-4)
- GPT-OSS 120B (128 experts, top-4)
- Qwen3-Next 80B (512 experts, top-10 + 1 shared)

---

## Performance

| Feature | Overhead | Notes |
|---------|----------|-------|
| Statistics | < 1% | Fast hash map updates |
| Name logging | 5-10% | String lookups + I/O |
| Per-layer | < 2% | Additional map level |
| JSON export | Negligible | Only at end |

**Recommendation:** Enable stats by default, name logging only for debugging.

---

## Reference: llama.cpp Codebase

### Logging Infrastructure

**GGML Layer:** `ggml/include/ggml.h:622-629`
```cpp
enum ggml_log_level {
    GGML_LOG_LEVEL_NONE  = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO  = 2,
    GGML_LOG_LEVEL_WARN  = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT  = 5,
};
```

**Common Layer (RECOMMENDED):** `common/log.h`
```cpp
#define LOG_DBG(...)  // Debug logging
#define LOG_INF(...)  // Info logging
#define LOG_WRN(...)  // Warning logging
#define LOG_ERR(...)  // Error logging
```

### MoE Tensor Naming

Pattern: `blk.<layer>.ffn_<proj>_exps.<expert>.weight`

Example (Layer 5, Expert 3):
```
blk.5.ffn_gate_exps.3.weight
blk.5.ffn_down_exps.3.weight
blk.5.ffn_up_exps.3.weight
```

Located in:
- `src/llama.cpp`: `llm_load_tensors()`
- `src/llama.h`: `llama_layer` struct
- `src/llama.cpp`: `llama_model::tensors_by_name`

---

## Usage Examples

```bash
# Example 1: Track activation frequencies
LLAMA_EXPERT_TRACE_STATS=1 \
LLAMA_EXPERT_TRACE_OUTPUT=stats.json \
./llama-cli -m mixtral-8x7b.gguf -p "Explain quantum computing"

# Example 2: Debug tensor names
LLAMA_EXPERT_TRACE_NAMES=1 \
./llama-cli -m mixtral-8x7b.gguf -p "Hello"

# Example 3: Per-layer analysis
LLAMA_EXPERT_TRACE_STATS=1 \
LLAMA_EXPERT_TRACE_PER_LAYER=1 \
./llama-cli -m qwen3-next-80b.gguf -p "Write a poem"

# Example 4: Combined with Post-Fetch (future)
LLAMA_POSTFETCH_ENABLE=1 \
LLAMA_EXPERT_TRACE_STATS=1 \
./llama-cli -m mixtral-8x7b.gguf -p "Test"
```

---

## Implementation Checklist

- [ ] Create `src/llama-expert-tracer.h`
- [ ] Create `src/llama-expert-tracer.cpp`
- [ ] Modify `src/llama-graph.cpp` (add hook)
- [ ] Modify `src/llama.cpp` (init/cleanup)
- [ ] Modify `src/CMakeLists.txt`
- [ ] Test with Mixtral-8x7B
- [ ] Test with Qwen3-Next 80B
- [ ] Verify JSON output
- [ ] Measure overhead
- [ ] Test thread safety

---



---

*Version 2.0 | Last Updated: 2026-02-08 | Based on llama.cpp codebase*
