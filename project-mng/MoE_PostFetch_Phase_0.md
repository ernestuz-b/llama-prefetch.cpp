# Phase 0 Implementation Guide: Expert Usage Tracer

**Version:** 1.0  
**Status:** Implementation Guide  
**Target:** llama.cpp MoE Optimization - Phase 0  
**Date:** 2026-02-09

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [File Structure](#file-structure)
4. [Implementation Steps](#implementation-steps)
5. [Code Style Guidelines](#code-style-guidelines)
6. [Testing Strategy](#testing-strategy)
7. [Integration Checklist](#integration-checklist)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

Phase 0 implements a **single-file expert usage tracer** that tracks which experts are activated during MoE model inference. This tracer:

- Uses llama.cpp's existing callback infrastructure (non-invasive)
- Provides runtime statistics on expert activation patterns
- Exports data to JSON for analysis
- Has zero overhead when disabled (controlled by environment variables)

### Key Features

| Feature | Description |
|---------|-------------|
| **Callback-Based** | Leverages `llm_graph_cb` and `ggml_backend_sched_eval_callback` |
| **Zero Overhead** | No cost when environment variables not set |
| **Optional** | Controlled via `LLAMA_EXPERT_TRACE_*` environment variables |
| **Single File** | All implementation in `common/expert-trace.cpp` |
| **Well-Documented** | Clear header comments and inline documentation |

### Target Models

- Qwen3-Next-80B (512 experts, 10 active per token)
- Qwen3-Coder-Next-80B (512 experts, 10 active per token)
- GPT-OSS-120B (128 experts, 4 active per token)
- GPT-OSS-20B (32 experts, 4 active per token)

---

## Design Principles

### 1. Simplicity Over Sophistication

Follow llama.cpp's core philosophy:

- **Avoid templates** - Use concrete types
- **Avoid modern STL** - Use basic for loops, simple containers
- **C-style APIs** - Plain interfaces with C++ implementation
- **Minimal dependencies** - Only use existing llama.cpp/ggml facilities

### 2. Cross-Platform Compatibility

- Test on Windows, Linux, macOS
- Support x86, ARM, RISC-V architectures
- Use platform-agnostic code (no OS-specific APIs)

### 3. Zero Overhead When Disabled

- Early return in `init()` if environment variables not set
- No callback registration when disabled
- No memory allocation when disabled
- No performance impact on normal inference

### 4. Upstream Acceptance

Design for easy PR acceptance:

- **Single compilation unit** - All code in one `.cpp` file
- **Minimal API surface** - Only essential public functions
- **Clean encapsulation** - Singleton pattern with private state
- **Well-documented** - Header comments explain usage
- **No dependencies** - Uses only existing facilities

---

## File Structure

### Files to Create

```
src/
├── llama-expert-trace.h      # Header file with class declaration
└── llama-expert-trace.cpp    # Implementation file
```

**Note:** The expert trace implementation is placed in `src/` (not `common/`) because:
1. It uses the internal logging system (`llama-impl.h`) which is only available in `src/`
2. It's tightly integrated with the core llama library
3. The `common/` directory contains application utilities, not core library code

### Files to Modify

```
src/
└── llama.cpp           # Add init/cleanup calls

src/CMakeLists.txt      # Add llama-expert-trace.cpp to build
```

---

## Implementation Steps

### Step 1: Create Header File (`src/llama-expert-trace.h`)

**Location:** `src/llama-expert-trace.h`

**Requirements:**
- Traditional header guards (not `#pragma once`)
- Include necessary headers
- Forward declare llama types
- Define `llama_expert_tracer` struct (not in a namespace - follows llama.cpp style)
- Provide C-style callback wrappers

**Implementation:**

```cpp
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

namespace llama {

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
 *   llama::expert_tracer::instance().init(ctx);
 *
 *   // Cleanup (typically in llama_free)
 *   llama::expert_tracer::instance().cleanup(ctx);
 *
 * Performance:
 *   - Zero overhead when disabled (env vars not set)
 *   - <1% overhead when stats enabled
 *   - 5-10% overhead when logging enabled (due to I/O)
 */
class expert_tracer {
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
    static expert_tracer & instance();

    // Initialize tracer for a context
    // Loads configuration from environment variables and registers callbacks
    void init(llama_context * ctx);

    // Cleanup and print statistics
    // Prints summary to log and exports to JSON if configured
    void cleanup(llama_context * ctx);

    // Graph build callback (optional, for early MoE detection)
    void on_graph_build(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il);

    // Eval callback (primary, for expert ID extraction)
    void on_eval(struct ggml_tensor * t, bool ask, llama_context * ctx);

    // Get current configuration (for testing)
    const config & get_config() const { return m_config; }

    // Get statistics (for testing/debugging)
    const std::unordered_map<int, layer_stats> & get_stats() const { return m_layer_stats; }

private:
    expert_tracer() = default;
    ~expert_tracer() = default;

    // Non-copyable, non-movable (singleton)
    expert_tracer(const expert_tracer &) = delete;
    expert_tracer & operator=(const expert_tracer &) = delete;

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
void expert_trace_graph_cb(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il);
bool expert_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

} // namespace llama

#endif // LLAMA_EXPERT_TRACE_H
```

**Style Notes:**
- Traditional header guards: `#ifndef LLAMA_LLAMA_EXPERT_TRACE_H`
- 4-space indentation
- Opening brace on same line
- Clear documentation comments
- Private constructors for singleton pattern

---

### Step 2: Create Implementation File (`src/llama-expert-trace.cpp`)

**Location:** `src/llama-expert-trace.cpp`

**Requirements:**
- Include header and necessary dependencies
- Implement all struct methods
- Follow llama.cpp naming conventions
- Use `LLAMA_LOG_*` macros from `llama-impl.h` for logging (internal logging system)
- Handle errors gracefully

**Implementation:**

```cpp
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

    // Register eval callback
    // Note: ctx->get_sched() returns the ggml_backend_sched pointer
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
    LLAMA_LOG_INFO("\n=== Expert Usage Statistics ===\n");
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
        return (t->op == GGML_OP_MUL_MAT_ID);
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
```

**Style Notes:**
- Include order: project headers first, then system headers
- Use `LLAMA_LOG_INFO`, `LLAMA_LOG_WARN`, `LLAMA_LOG_DEBUG` macros from `llama-impl.h`
- Include `__func__` in log messages (implicit in LLAMA_LOG_* macros)
- Use `std::lock_guard` for mutex locking
- Simple for loops, no range-based for loops
- Explicit `std::` prefix for STL types

---

### Step 3: Modify `src/llama.cpp`

**Location:** `src/llama.cpp`

**Changes Required:**

1. Add include at top of file:

```cpp
#include "common/expert-trace.h"
```

2. In `llama_new_context_with_model()` function, add initialization:

```cpp
struct llama_context * llama_new_context_with_model(
        struct llama_model * model,
        const llama_context_params & params) {
    // ... existing initialization code ...

    // Initialize expert tracer (Phase 0)
    llama::expert_tracer::instance().init(ctx);

    return ctx;
}
```

3. In `llama_free()` function, add cleanup:

```cpp
void llama_free(struct llama_context * ctx) {
    // Cleanup expert tracer (Phase 0)
    llama::expert_tracer::instance().cleanup(ctx);

    // ... existing cleanup code ...
}
```

**Important Notes:**
- Place initialization after context is fully constructed
- Place cleanup before context is destroyed
- Use `llama::` namespace prefix (not `using namespace`)
- Keep changes minimal and localized

---

### Step 4: Update Build System

**Location:** `CMakeLists.txt` (or `common/CMakeLists.txt`)

**Changes Required:**

Add `expert-trace.cpp` to the common library sources:

```cmake
# In common/CMakeLists.txt or similar
set(COMMON_SOURCES
    # ... existing sources ...
    common/expert-trace.cpp
)
```

**Alternative (if using Makefile):**

```makefile
# In Makefile
COMMON_OBJS += common/expert-trace.o
```

---

### Step 5: Verify Compilation

**Commands:**

```bash
# Clean build
make clean

# Build with expert tracing
cmake -B build -DLLAMA_CURL=ON
cmake --build build -j$(nproc)

# Verify no compilation errors
```

**Expected Output:**
- No warnings or errors
- `expert-trace.cpp` compiled successfully
- `llama-cli` binary created

---

## Code Style Guidelines

### Naming Conventions

| Category | Convention | Examples |
|----------|------------|----------|
| **Functions** | `snake_case` | `init()`, `cleanup()`, `record_expert_usage()` |
| **Variables** | `snake_case` | `layer_id`, `expert_id`, `total_tokens` |
| **Constants** | `UPPER_CASE` | (none in this implementation) |
| **Types/Structs** | `snake_case` | `config`, `layer_stats`, `expert_tracer` |
| **Class Methods** | `snake_case` | `load_from_env()`, `export_stats_json()` |

### Formatting

- **Indentation:** 4 spaces
- **Line length:** Pragmatic (readability over strict limits)
- **Braces:** Opening brace on same line
- **Spaces:** Consistent spacing around operators

**Example:**

```cpp
// Good - follows style guide
void expert_tracer::init(llama_context * ctx) {
    m_config.load_from_env();

    if (!m_config.enable_stats && !m_config.enable_logging) {
        return;
    }

    LLAMA_LOG_INFO("Expert tracing enabled (stats=%d, logging=%d)\n",
                   m_config.enable_stats, m_config.enable_logging);
}
```

### Language Features

**Use:**
- Basic for loops
- Simple STL containers (`std::vector`, `std::unordered_map`, `std::string`)
- `std::mutex` and `std::lock_guard` for thread safety
- C++17 features only when necessary

**Avoid:**
- Templates
- Range-based for loops (use traditional for loops)
- Modern C++ idioms (lambda expressions, etc.)
- Complex STL algorithms
- Smart pointers (use straightforward memory management)

### Error Handling

**Logging:**
```cpp
// Use LLAMA_LOG_* macros from llama-impl.h
LLAMA_LOG_INFO("Expert tracing enabled\n");
LLAMA_LOG_WARN("Failed to open file: %s\n", filename.c_str());
LLAMA_LOG_DEBUG("Layer %d: Experts [...]\n", layer_id);
```

**Exceptions:**
```cpp
// Use std::runtime_error for exceptions
throw std::runtime_error("Failed to initialize expert tracer");
```

**Graceful Degradation:**
```cpp
// Return early on errors
if (!file.is_open()) {
    LLAMA_LOG_WARN("Failed to open output file: %s\n", filename.c_str());
    return;
}
```

### Comments

**Function Documentation:**
```cpp
/**
 * Initialize tracer for a context.
 *
 * Loads configuration from environment variables and registers callbacks.
 * Zero overhead when disabled (env vars not set).
 *
 * @param ctx The llama context to trace
 */
void init(llama_context * ctx);
```

**Inline Comments:**
```cpp
// Extract layer ID from tensor name like "blk.5.ffn_moe_..."
int layer_id = extract_layer_id(t->name);
```

**TODO Comments:**
```cpp
// TODO: Add support for multi-context tracing
```

---

## Testing Strategy

### Unit Tests

**Test 1: Configuration Loading**

```bash
# Test stats enable
export LLAMA_EXPERT_TRACE_STATS=1
./llama-cli -m model.gguf -p "test"
# Expected: "Expert tracing enabled (stats=1, logging=0)"

# Test logging enable
export LLAMA_EXPERT_TRACE_LOGGING=1
./llama-cli -m model.gguf -p "test"
# Expected: "Expert tracing enabled (stats=0, logging=1)"

# Test both enabled
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_LOGGING=1
./llama-cli -m model.gguf -p "test"
# Expected: "Expert tracing enabled (stats=1, logging=1)"

# Test disabled (no env vars)
unset LLAMA_EXPERT_TRACE_STATS
unset LLAMA_EXPERT_TRACE_LOGGING
./llama-cli -m model.gguf -p "test"
# Expected: No tracing output
```

**Test 2: Statistics Collection**

```bash
export LLAMA_EXPERT_TRACE_STATS=1
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "Hello world"

# Expected output:
# === Expert Usage Statistics ===
# Layer 0 (10 tokens):
#   Expert  23:    3 activations (30.0%)
#   Expert  45:    2 activations (20.0%)
#   ...
```

**Test 3: JSON Export**

```bash
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_OUTPUT=expert_stats.json
./llama-cli -m model.gguf -p "test"

# Verify file exists and is valid JSON
cat expert_stats.json | python -m json.tool
```

### Integration Tests

**Test 4: MoE Model Compatibility**

```bash
# Test with Qwen3-Next-80B
export LLAMA_EXPERT_TRACE_STATS=1
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "test"

# Test with GPT-OSS-120B
./llama-cli -m gpt-oss-120b-Q4_K_M.gguf -p "test"

# Test with GPT-OSS-20B
./llama-cli -m gpt-oss-20b-Q4_K_M.gguf -p "test"
```

**Test 5: Zero Overhead When Disabled**

```bash
# Measure performance without tracing
time ./llama-cli -m model.gguf -p "test" > /dev/null

# Measure performance with tracing disabled (no env vars)
unset LLAMA_EXPERT_TRACE_STATS
time ./llama-cli -m model.gguf -p "test" > /dev/null

# Expected: No significant difference (<1%)
```

### Performance Tests

**Test 6: Overhead Measurement**

```bash
# Baseline (no tracing)
time ./llama-cli -m model.gguf -p "test" > /dev/null

# Stats enabled
export LLAMA_EXPERT_TRACE_STATS=1
time ./llama-cli -m model.gguf -p "test" > /dev/null

# Logging enabled
export LLAMA_EXPERT_TRACE_LOGGING=1
time ./llama-cli -m model.gguf -p "test" > /dev/null

# Expected:
# - Stats: <1% overhead
# - Logging: 5-10% overhead (due to I/O)
```

#### Execution

The commands to setup  and compile the software are:
cmake -B ../llama.cpp-build-cuda -DGGML_CUDA=ON
cmake --build ../llama.cpp-build-cuda --config Release -j 32

---

## Integration Checklist

### Pre-Integration

- [ ] Read and understand [`CPP_STYLE_GUIDE.md`](CPP_STYLE_GUIDE.md)
- [ ] Read and understand [`MoE_PostFetch.md`](MoE_PostFetch.md)
- [ ] Review llama.cpp contribution guidelines
- [ ] Set up development environment

### Implementation

- [ ] Create `common/expert-trace.h` with complete class declaration
- [ ] Create `common/expert-trace.cpp` with complete implementation
- [ ] Modify `src/llama.cpp` to add init/cleanup calls
- [ ] Update build system (`CMakeLists.txt` or `Makefile`)
- [ ] Verify compilation succeeds

### Testing

- [ ] Test configuration loading (all env var combinations)
- [ ] Test statistics collection on MoE models
- [ ] Test JSON export functionality
- [ ] Test zero overhead when disabled
- [ ] Test on multiple MoE models (Qwen3, GPT-OSS)
- [ ] Verify no memory leaks (valgrind, if available)

### Documentation

- [ ] Add header comments to `expert-trace.h`
- [ ] Add inline comments to `expert-trace.cpp`
- [ ] Update README with usage examples
- [ ] Document environment variables

### Code Review

- [ ] Self-review against style guide
- [ ] Check for potential race conditions
- [ ] Verify error handling
- [ ] Ensure no memory leaks
- [ ] Test on multiple platforms (if possible)

---

## Troubleshooting

### Common Issues

**Issue 1: Compilation Error - Missing Headers**

```
error: 'ggml_backend_sched_set_eval_callback' was not declared
```

**Solution:**
- Ensure `ggml.h` and `llama.h` are included
- Check that llama.cpp version supports eval callbacks
- Verify `ctx->sched` is accessible

**Issue 2: Callback Not Invoked**

```
No expert statistics printed
```

**Solution:**
- Verify environment variables are set: `export LLAMA_EXPERT_TRACE_STATS=1`
- Check that model is actually MoE (has `ffn_moe_topk` tensor)
- Enable logging: `export LLAMA_EXPERT_TRACE_LOGGING=1`
- Check that callback is registered in `init()`

**Issue 3: Layer ID Extraction Fails**

```
Layer -1: Experts [...]
```

**Solution:**
- Verify tensor name format matches regex pattern `blk\.(\d+)\.`
- Check that tensor names follow llama.cpp conventions
- Add debug logging to see actual tensor names

**Issue 4: JSON Export Fails**

```
Failed to open output file: expert_stats.json
```

**Solution:**
- Check file path is valid and writable
- Verify directory exists
- Check file permissions

**Issue 5: Performance Regression**

```
Significant slowdown with tracing enabled
```

**Solution:**
- Disable logging (use only stats): `unset LLAMA_EXPERT_TRACE_LOGGING`
- Verify early return in `init()` when disabled
- Check for unnecessary mutex locks
- Profile with performance tools

### Debug Mode

Enable verbose logging for debugging:

```bash
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_LOGGING=1
export LLAMA_LOG_LEVEL=2  # Debug level
./llama-cli -m model.gguf -p "test"
```

### Validation

Validate expert ID extraction:

```bash
# Enable logging to see expert IDs
export LLAMA_EXPERT_TRACE_LOGGING=1
./llama-cli -m model.gguf -p "test"

# Compare with manual inspection using GGML_SCHED_DEBUG
export GGML_SCHED_DEBUG=1
./llama-cli -m model.gguf -p "test"
```

---

## Usage Examples

### Basic Usage

```bash
# Enable expert statistics
export LLAMA_EXPERT_TRACE_STATS=1

# Run inference
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "Hello world"

# Output:
# Expert tracing enabled (stats=1, logging=0)
# ... inference ...
# === Expert Usage Statistics ===
# Layer 0 (10 tokens):
#   Expert  23:    3 activations (30.0%)
#   Expert  45:    2 activations (20.0%)
#   ...
```

### Export to JSON

```bash
# Enable statistics and JSON export
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_OUTPUT=expert_stats.json

# Run inference
./llama-cli -m model.gguf -p "test"

# View JSON
cat expert_stats.json | python -m json.tool
```

### Debug Mode

```bash
# Enable verbose logging
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_LOGGING=1

# Run inference
./llama-cli -m model.gguf -p "test"

# Output includes:
# [EXPERT-TRACE] Layer 0: MoE layer detected (tensor 'ffn_moe_topk')
# [EXPERT-TRACE] Layer 0: Experts [23, 45, 67, 89]
```

### Disabled (Zero Overhead)

```bash
# No environment variables set
./llama-cli -m model.gguf -p "test"

# No tracing output, zero overhead
```

---

## Next Steps

After completing Phase 0:

1. **Submit PR for Phase 0**
   - Create pull request with expert tracer
   - Include usage examples in PR description
   - Reference this implementation guide

2. **Prepare for Phase 1**
   - Review [`MoE_PostFetch.md`](MoE_PostFetch.md) Phase 1 requirements
   - Plan Post-Fetch hook integration
   - Design CUDA async transfer implementation

3. **Gather Feedback**
   - Monitor PR review comments
   - Address any style or implementation issues
   - Refine based on upstream feedback

---

## Summary

Phase 0 implements a **minimal, non-invasive expert usage tracer** that:

- ✅ Uses llama.cpp's existing callback infrastructure
- ✅ Has zero overhead when disabled
- ✅ Provides runtime statistics on expert activation
- ✅ Exports data to JSON for analysis
- ✅ Follows llama.cpp C++ style guide
- ✅ Is ready for upstream PR submission

**Key Design Decisions:**

- **Single file implementation** - Easy to review and integrate
- **Singleton pattern** - Clean state management
- **Environment variable configuration** - No code changes needed
- **Callback-based** - Non-invasive, leverages existing infrastructure
- **Zero overhead when disabled** - Early return in `init()`

**Files Created:**
- `common/expert-trace.h` - Header file with class declaration
- `common/expert-trace.cpp` - Implementation file

**Files Modified:**
- `src/llama.cpp` - Add init/cleanup calls
- `CMakeLists.txt` - Add expert-trace.cpp to build

**Testing:**
- Configuration loading
- Statistics collection
- JSON export
- Zero overhead verification
- MoE model compatibility

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-09  
**References:**
- [`CPP_STYLE_GUIDE.md`](CPP_STYLE_GUIDE.md) - C++ coding standards
- [`MoE_PostFetch.md`](MoE_PostFetch.md) - Full Post-Fetch specification
- [`Debugging_MoE_Experts.md`](Debugging_MoE_Experts.md) - MoE debugging guide

# To run

# Enable expert statistics
export LLAMA_EXPERT_TRACE_STATS=1
# Enable verbose logging
export LLAMA_EXPERT_TRACE_LOGGING=1
# Export to JSON file
export LLAMA_EXPERT_TRACE_OUTPUT=expert_stats.json
# Run inference
./llama-cli -m model.gguf -p "Hello world"

