# Post-Fetch MoE Execution

**Version:** 0.0.7 (Implemented Callback Infrastructure) **Status:** Design Document **Target:** llama.cpp MoE Optimization for Low-VRAM Consumer GPUs

**IMPORTANT UPDATE (2026-02-13):** Phase 0 (Expert Usage Tracer) has been **IMPLEMENTED** with the following key findings:

1. **Tensor naming convention** - Actual tensor names are `ffn_moe_gate-N`, `ffn_moe_up-N`, `ffn_moe_down-N` (not `blk.N.ffn_moe_...`)
2. **Only process gate operations** - Must filter for `ffn_moe_gate` to avoid triple-counting (gate, up, down all have same expert IDs)
3. **Tensor data padding** - The ids tensor may have padding between rows, requiring row-by-row reading using `nb[1]` stride
4. **Callback return type** - The `on_eval` function must return `bool`, not `void`
5. **Layer ID extraction** - Must handle both `blk.N.` and `-N` suffix patterns

**Previous UPDATE (2026-02-13):** This document has been updated with a dedicated callback infrastructure to avoid collisions with user callbacks:

1. **Added `cb_trace` callback** - Dedicated callback slot for tracing, separate from user's `cb_eval`
2. **Fixed callback overwrite issue** - The original approach was overwritten by the graph build loop
3. **Added prefetch callback design** - Future extension for prefetching functionality
4. **Documented nullptr initialization** - All callback pointers must be initialized to `nullptr`

**Previous UPDATE (2026-02-09):** 
1. **Tensor naming convention corrected** - Now reflects actual 3D tensor structure (`blk.5.ffn_gate_exps` with dimensions `{n_embd, n_ff, n_expert}`)
2. **Tensor lookup code corrected** - Now uses direct pointer access to layer structure instead of `ggml_get_tensor()` lookup
3. **Context access corrected** - Now uses `ctx->get_sched()` accessor method consistently
4. **Removed use of non-exported function** - `common_ggml_ne_string()` replaced with manual formatting
5. **Added 3D tensor offset calculations** - Shows how to access individual expert weights from merged tensor structure

See [`tools/imatrix/imatrix.cpp:253-254`](../tools/imatrix/imatrix.cpp:253-254) for reference to the tensor structure change.


# Coding conventions

C++ Standard:  C++17 and using more modern STL. CUDA files will have extensions cuh/cu.

Cross-Platform Compatibility

Always consider cross-compatibility with other operating systems (Windows, Linux, macOS) and architectures (x86, ARM, RISC-V). Test your code on multiple platforms before submitting.

Minimal Dependencies

Avoid adding third-party dependencies, extra files, or extra headers. Each new dependency increases the maintenance burden and potential for compatibility issues.

Pragmatic Over Dogmatic

Use good Object Oriented patterns, and SOLID principles, but in order to be little intrusive, it's allowed to keep very related classes in the same file. It's important to do little impact in the existing codebase.

Minimize the points of insertion in the current code base.



## Model Architecture Comparison

The Post-Fetch mechanism is designed for modern MoE models with varying expert configurations:

| Feature | GPT-OSS 120B | GPT-OSS 20B | Qwen3-Next (80B) | Qwen3-Coder-Next |
| - | - | - | - | - |
| **Total Parameters** | ~117B - 120B | ~21B | 80B | 80B |
| **Active Parameters** | 5.1B | 3.6B | 3.0B | 3.0B |
| **Total Experts** | 128 | 32 | 512 | 512 |
| **Experts Activated** | Top-4 | Top-4 | 10 Routed + 1 Shared | 10 Routed + 1 Shared |
| **Non-Expert Params** | ~12B | ~3.5B | ~6.5B | ~6.5B |
| **Transformer Layers** | 36 | 24 | 48 | 48 |
| **Hidden Dimension** | 2,880 | 2,880 | 2,048 | 2,048 |
| **Attention Type** | Dense (Multi-Head) | Dense (Multi-Head) | Hybrid (Gated DeltaNet) | Hybrid (Gated DeltaNet) |


### Key Observations

- **Expert diversity varies significantly**: Qwen3 models use 512 experts vs. GPT-OSS's 128

- **Active parameters remain low**: Despite large total sizes, only ~3-5B parameters are active per token

- **Attention evolution**: Qwen3 models use hybrid Gated DeltaNet instead of traditional attention

- **Post-Fetch relevance**: Larger expert counts benefit more from efficient prefetching


## Table of Contents

1. [Task 0: Expert Usage Tracer (Callback-Based)](#task-0-expert-usage-tracer-callback-based)

2. [Overview](#overview)

3. [Key Principles](#key-principles)

4. [Target Use Case](#target-use-case)

5. [Technical Background](#technical-background)

6. [How Post-Fetch Works](#how-post-fetch-works)

7. [Architecture](#architecture)

8. [Implementation Details](#implementation-details)

9. [Configuration](#configuration)

10. [Performance Characteristics](#performance-characteristics)

11. [Future Extensions](#future-extensions)

12. [Implementation Roadmap](#implementation-roadmap)


## Task 0: Expert Usage Tracer (Callback-Based)

**Purpose:** Provide optional runtime statistics and logging of expert activation patterns using llama.cpp's existing callback infrastructure.

### Overview

The Expert Usage Tracer uses a **dedicated callback infrastructure** (`cb_trace`) to avoid collisions with user callbacks. This ensures reliable operation regardless of whether the user sets their own `cb_eval` callback.

The tracer tracks:

1. **Which experts are activated** during inference (via `ffn_moe_topk` tensor)

2. **How frequently each expert is used** across tokens/layers

3. **Expert selection pipeline** (logits → probabilities → weights → activations)

### Dedicated Callback Architecture

Instead of reusing the user's `cb_eval` callback slot, we add a dedicated `cb_trace` callback:

```cpp
// In llama_context_params (include/llama.h)
struct llama_context_params {
    // ... existing members ...
    
    // User's callback (existing)
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    
    // Tracing callback (internal, for expert tracing, prefetching, etc.)
    // IMPORTANT: Must be initialized to nullptr
    ggml_backend_sched_eval_callback cb_trace = nullptr;
    void * cb_trace_user_data = nullptr;
    
    // Future: Prefetch callback
    // ggml_backend_sched_eval_callback cb_prefetch = nullptr;
    // void * cb_prefetch_user_data = nullptr;
};
```

### Why Not Use `cb_eval` Directly?

The original approach of calling `ggml_backend_sched_set_eval_callback()` directly had a critical flaw:

1. Expert tracer sets callback in `init()`
2. Graph build loop at `llama-context.cpp:1141` calls `ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, ...)` 
3. This **overwrites** the expert tracer's callback with `nullptr` (the default)

The dedicated `cb_trace` slot avoids this issue entirely.

### Why Not Functors?

While C++ functors would provide better type safety, llama.cpp uses C-style function pointers for:
- C API compatibility
- Simplicity and portability
- Historical consistency with the codebase

### Callback Hook Points

The MoE pathway provides **23 callback points** (see [`Debugging_MoE_Experts.md`](file:///mnt/AI/AiProgs/llama-prefetch.cpp/memory-bank/Debugging_MoE_Experts.md#8-callback-points-before-and-after-expert-selection)):

#### Critical Callbacks for Post-Fetch

| Callback | Tensor Name | Purpose for Post-Fetch |
| - | - | - |
| **`ffn_moe_topk`** (line 1197) | Selected expert indices | **PRIMARY**: Identifies which experts to prefetch |
| **`ffn_moe_weights_norm`** (line 1230) | Normalized expert weights | Optional: Weight-based prefetch prioritization |
| `ffn_moe_logits` (line 1121) | Raw gating scores | Analysis: Understanding routing patterns |
| `ffn_moe_probs` (line 1148) | Expert probabilities | Analysis: Router confidence metrics |


### Implementation Strategy

#### 1. Add Dedicated Callback Infrastructure

Add `cb_trace` to the core structures:

```cpp
// In include/llama.h - llama_context_params
ggml_backend_sched_eval_callback cb_trace = nullptr;  // MUST be nullptr
void * cb_trace_user_data = nullptr;

// In src/llama-cparams.h - llama_cparams  
ggml_backend_sched_eval_callback cb_trace = nullptr;
void * cb_trace_user_data = nullptr;

// In ggml/src/ggml-backend.cpp - scheduler struct
ggml_backend_sched_eval_callback callback_trace = nullptr;
void * callback_trace_user_data = nullptr;
```

#### 2. Set Trace Callback in Context Initialization

```cpp
// In llama_init_from_model() - BEFORE context creation
llama_expert_tracer::instance().init_config();
if (llama_expert_tracer::instance().is_enabled()) {
    params.cb_trace = llama_expert_trace_eval_cb;
    params.cb_trace_user_data = nullptr;  // Will be set to ctx after creation
}

// Create context
auto * ctx = new llama_context(*model, params);

// After context creation
llama_expert_tracer::instance().init(ctx);
```

#### 3. Invoke Trace Callback in Evaluation Loop

```cpp
// In ggml-backend.cpp evaluation loop (around line 1612)
if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
    break;
}

// Also invoke trace callback (if set)
if (sched->callback_trace) {
    sched->callback_trace(t, false, sched->callback_trace_user_data);
}
```

#### 4. Eval Callback for Expert Data Extraction

Use the `ggml_backend_sched_eval_callback` to extract expert indices:

```cpp
// Eval callback function
bool llama_expert_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (!g_expert_trace.enable_stats) {
        return true;
    }

    if (ask) {
        // We only want to see MUL_MAT_ID operations for the gate (expert selection)
        // The gate tensor is named "ffn_moe_gate-N" - this is where expert IDs are determined
        // The up/down tensors use the same expert IDs but for different computations
        return (t->op == GGML_OP_MUL_MAT_ID) && 
               (strstr(t->name, "ffn_moe_gate") != nullptr);
    }

    // Extract expert IDs from the operation
    if (t->op == GGML_OP_MUL_MAT_ID && strstr(t->name, "ffn_moe_gate") != nullptr) {
        // ids tensor is src[2]
        const ggml_tensor * ids = t->src[2];

        // Extract layer number from tensor name (e.g., "ffn_moe_gate-0")
        int layer_id = extract_layer_id(t->name);

        // Copy expert IDs to host
        // The tensor has shape [ne[0], ne[1]] = [experts_per_token, n_tokens]
        // There might be padding between rows, so we read row by row using nb[1] stride
        std::vector<int32_t> expert_ids;
        expert_ids.reserve(ggml_nelements(ids));
        
        for (int64_t i = 0; i < ids->ne[1]; i++) {
            size_t offset = i * ids->nb[1];
            size_t row_size = ids->ne[0] * sizeof(int32_t);
            std::vector<int32_t> row(ids->ne[0]);
            ggml_backend_tensor_get(ids, row.data(), offset, row_size);
            expert_ids.insert(expert_ids.end(), row.begin(), row.end());
        }

        // Update statistics
        std::lock_guard<std::mutex> lock(g_expert_trace.trace_mutex);
        for (int32_t expert_id : expert_ids) {
            if (expert_id >= 0) {
                g_expert_trace.layer_expert_counts[layer_id][expert_id]++;
            }
        }

        if (g_expert_trace.enable_logging) {
            LLAMA_LOG_DEBUG("[EXPERT-TRACE] Layer %d: Expert IDs = [", layer_id);
            for (size_t i = 0; i < expert_ids.size(); i++) {
                LLAMA_LOG_DEBUG("%d%s", expert_ids[i], i+1 < expert_ids.size() ? ", " : "");
            }
            LLAMA_LOG_DEBUG("]\n");
        }
    }

    return true;
}
```

**Key Implementation Notes:**

1. **Filter for gate operations only** - The MoE layer has three MUL_MAT_ID operations (gate, up, down). Only the gate operation should be processed to avoid triple-counting experts.

2. **Tensor naming** - Actual tensor names are `ffn_moe_gate-N`, `ffn_moe_up-N`, `ffn_moe_down-N` where N is the layer number.

3. **Handle tensor padding** - The ids tensor may have padding between rows. Use `nb[1]` stride to read row by row.

4. **Return bool** - The callback must return `true` to continue execution.

#### 3. Integration with Post-Fetch

**IMPORTANT: Tensor Structure Note**

The actual llama.cpp codebase uses a **single 3D tensor** for all experts, not per-expert tensors. This is a critical architectural difference that affects Post-Fetch implementation.

**Actual tensor structure:**
```cpp
// From src/llama-model.cpp:2826
layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), 
                                  {n_embd, n_ff, n_expert}, ...);
// Tensor name: "blk.5.ffn_gate_exps.weight"
// Dimensions: {n_embd, n_ff, n_expert} - contains ALL experts in one tensor
```

**Reference:** See [`tools/imatrix/imatrix.cpp:253-254`](../tools/imatrix/imatrix.cpp:253-254) for comment about this change.

**Post-Fetch with 3D Tensors:**

```cpp
// In build_moe_ffn() or similar high-level hook
void postfetch_initiate_transfer(
    const llama_model * model,
    const llama_context * ctx,
    int layer_id,
    const std::vector<int32_t> & selected_experts
) {
    if (!is_postfetch_enabled()) {
        return;
    }

    // These expert IDs come from the callback
    for (int32_t expert_id : selected_experts) {
        // Access expert tensors directly from layer structure
        // (NOT via ggml_get_tensor lookup - tensors are direct pointers)
        const llama_layer & layer = model->layers[layer_id];

        // Calculate offset into 3D tensor for this expert
        // Tensor dimensions: {n_embd, n_ff, n_expert}
        // Expert weights are at offset: expert_id * (n_embd * n_ff)
        size_t expert_offset = expert_id * (layer.ffn_gate_exps->ne[0] * layer.ffn_gate_exps->ne[1]);

        // Calculate sizes
        size_t gate_size = layer.ffn_gate_exps->ne[0] * layer.ffn_gate_exps->ne[1];
        size_t up_size = layer.ffn_up_exps->ne[0] * layer.ffn_up_exps->ne[1];
        size_t down_size = layer.ffn_down_exps->ne[0] * layer.ffn_down_exps->ne[1];

        // Pointers to expert weight slices within the 3D tensors
        ggml_tensor * gate_slice = (ggml_tensor *)((char *)layer.ffn_gate_exps->data + expert_offset);
        ggml_tensor * up_slice = (ggml_tensor *)((char *)layer.ffn_up_exps->data + expert_offset);
        ggml_tensor * down_slice = (ggml_tensor *)((char *)layer.ffn_down_exps->data + expert_offset);

        // Initiate async transfer to GPU
        postfetch_transfer_async(gate_slice, up_slice, down_slice);
    }
}
```

### Configuration via Environment Variables

Instead of command-line flags, use environment variables (llama.cpp convention):

| Variable | Values | Description |
| - | - | - |
| `LLAMA_EXPERT_TRACE_STATS` | `0` or `1` | Enable activation counting via callbacks |
| `LLAMA_EXPERT_TRACE_LOGGING` | `0` or `1` | Print expert IDs during execution |
| `LLAMA_EXPERT_TRACE_OUTPUT` | File path | Export statistics to JSON file |


### Initialization Code

```cpp
// In llama_init_from_model() - BEFORE context creation
// Initialize expert tracer config and set trace callback if enabled
llama_expert_tracer::instance().init_config();
if (llama_expert_tracer::instance().is_enabled()) {
    params.cb_trace = llama_expert_trace_eval_cb;
    params.cb_trace_user_data = nullptr;  // Will be set to ctx after creation
}

// Create context
auto * ctx = new llama_context(*model, params);

// After context creation - clear any previous statistics
llama_expert_tracer::instance().init(ctx);

// In llama_free() or similar
void cleanup_expert_trace(llama_context * ctx) {
    if (!llama_expert_tracer::instance().is_enabled()) {
        return;
    }

    // Print statistics
    LLAMA_LOG_INFO("\n=== Expert Usage Statistics ===\n");
    for (const auto & [layer_id, expert_counts] : g_expert_trace.layer_expert_counts) {
        LLAMA_LOG_INFO("Layer %d:\n", layer_id);
        for (const auto & [expert_id, count] : expert_counts) {
            LLAMA_LOG_INFO("  Expert %d: %d activations\n", expert_id, count);
        }
    }

    // Export to JSON if configured
    const char* output_file = std::getenv("LLAMA_EXPERT_TRACE_OUTPUT");
    if (output_file) {
        export_expert_trace_json(output_file, g_expert_trace);
    }
}
```

### Advantages of Callback-Based Approach

| Aspect | Custom Instrumentation | Callback-Based (New) |
| - | - | - |
| **Code Complexity** | High (modify MoE internals) | Low (register callbacks) |
| **Integration Risk** | High (invasive changes) | Low (existing infrastructure) |
| **Maintenance** | High (track MoE changes) | Low (callbacks are stable API) |
| **Debug Tools** | None | Works with existing `GGML_SCHED_DEBUG` |
| **Performance** | Unknown (custom code) | Known (callback overhead <1%) |
| **Compatibility** | Fragile (model-specific) | Robust (works with all MoE models) |


### Performance Overhead

Based on llama.cpp's existing callback usage:

| Feature | Overhead | Notes |
| - | - | - |
| Graph callback only | <0.1% | Called during graph building (once per batch) |
| Eval callback (stats) | <1% | Minimal logic, infrequent tensor copies |
| Eval callback (logging) | 5-10% | String formatting, I/O overhead |
| Full pipeline (23 callbacks) | 2-3% | Only if `LLAMA_EXPERT_TRACE_VERBOSE=1` |


### Integration with Post-Fetch

The callback-based tracer feeds directly into Post-Fetch:

```
┌─────────────────────────────────────────────────────────────┐
│                     INFERENCE LOOP                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Callback: Discover MoE layers                        │
│  - Filter for "ffn_moe_topk" tensors                        │
│  - Mark layers with MoE                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Eval Callback: Extract expert indices                      │
│  - Monitor GGML_OP_MUL_MAT_ID operations                    │
│  - Copy expert IDs from 'ids' tensor                        │
│  - Update statistics (optional)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Post-Fetch Hook: Initiate async transfers                  │
│  - Use expert IDs from callback                             │
│  - Lookup expert tensors in model                           │
│  - Start cudaMemcpyAsync() to GPU                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Computation: Use prefetched experts                        │
│  - Check if transfer complete (non-blocking)                │
│  - Fall back to CPU if not ready                            │
└─────────────────────────────────────────────────────────────┘
```

### Minimal Example Usage

```bash

# Enable expert tracing with statistics
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_OUTPUT=expert_stats.json

# Run inference
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "Hello world"

# Output:
# Expert tracing enabled (stats=1, logging=0)
# ...inference...
# === Expert Usage Statistics ===
# Layer 0:
#   Expert 3: 12 activations
#   Expert 7: 8 activations
# ...
# Statistics exported to: expert_stats.json
```

### JSON Export Format

```bash

{
  "model": "qwen3-next-80b",
  "total_tokens": 100,
  "layers": [
    {
      "layer_id": 0,
      "experts": [
        {"expert_id": 3, "activations": 12, "percentage": 12.0},
        {"expert_id": 7, "activations": 8, "percentage": 8.0}
      ]
    }
  ]
}
```

### Summary: What Changed from v0.0.4

| Aspect | v0.0.4 (Custom) | v0.0.5 (Callback-Based) |
| - | - | - |
| **Implementation** | Custom instrumentation in MoE code | Use existing callback infrastructure |
| **Integration** | Invasive (modify build_moe_ffn) | Non-invasive (register callbacks) |
| **Data Access** | Direct tensor access | Via callback parameters |
| **Configuration** | Command-line flags | Environment variables |
| **Overhead** | Unknown | <1% (stats), 5-10% (logging) |
| **Compatibility** | Model-specific | All MoE models |
| **Maintenance** | High (track MoE changes) | Low (stable callback API) |



## Overview

[Rest of document continues with Post-Fetch implementation details...]

## Key Principles

Post-Fetch MoE execution is built on three core principles:

1. **Simplicity Over Sophistication**

   - No caching, no prediction, no state tracking

   - Atomic transfers: fetch exactly what's needed, when it's needed

   - Clean fallback: if transfer isn't ready, use CPU (correct, just slower)

2. **Target the Weakest Hardware First**

   - Optimize for the **lowest common denominator**: low-VRAM consumer GPUs

   - If it works on weak hardware, it works everywhere

   - Trade peak performance for robustness and correctness

3. **Hide Latency, Don't Fight It**

   - PCIe transfer latency is unavoidable

   - But CPU computation during expert routing is also unavoidable

   - Overlap them: fetch weights during routing computation


## Target Use Case

**Primary:** Running large MoE models (80B+ parameters, 512+ experts) on consumer GPUs with limited VRAM (8-16GB).

**Example Hardware:**

- GPU: RTX 4060 Ti 16GB, RTX 3090 24GB

- System RAM: 64GB+

- PCIe: Gen 3 x16 (typical consumer motherboard)

**Example Workload:**

- Model: Qwen3-Next-80B (512 experts, 10 active per token)
- Quantization: Q4_K_M (expert weights ~4 bits per parameter)

- Inference: Interactive chat, batch size 1-8

**Key Constraint:** Expert weights cannot all fit in GPU VRAM, but non-expert parameters (attention, embeddings) can.


## Technical Background

### Why MoE Models Don't Fit in VRAM

For **Qwen3-Next-80B** with **Q4_K_M** quantization:

```
Total model size:     ~45 GB (quantized)
├─ Non-expert params: ~6.5 GB (attention, embeddings, norms)
└─ Expert params:     ~38.5 GB (512 experts × ~75 MB each)

Available VRAM:       16 GB (RTX 4060 Ti)

Problem: 38.5 GB experts > 16 GB VRAM
```

### Traditional Approaches and Their Limits

| Approach | Idea | Limitation |
| - | - | - |
| **Offloading** | Keep all weights in CPU RAM, transfer on demand | Too slow (100+ ms latency) |
| **Expert Caching** | Cache frequently-used experts in GPU | Complex (eviction policy, prediction) |
| **Pre-Fetching** | Predict next experts, fetch early | Fragile (routing is non-deterministic) |
| **Layer-by-Layer** | Process one layer at a time | Memory thrashing, poor GPU utilization |


### Post-Fetch Insight

**Key Observation:** Expert routing has **unavoidable CPU computation**:

1. Compute gating logits (small dense matrix multiply)

2. Apply activation function (softmax/sigmoid)

3. Select top-k experts (argsort operation)

This takes **5-15 ms** on typical CPUs — approximately the same as PCIe transfer time for expert weights!

**Post-Fetch Strategy:**

- Start weight transfer **after** experts are selected

- Let CPU computation and PCIe transfer run in parallel

- By the time experts are needed, weights are (mostly) ready


## How Post-Fetch Works

### Execution Timeline

```
Traditional Approach (blocking):
├─ [5ms]  CPU: Compute routing
├─ [10ms] PCIe: Transfer expert weights  ← BLOCKING
└─ [8ms]  GPU: Compute expert outputs
Total: 23ms

Post-Fetch Approach (overlapped):
├─ [5ms]  CPU: Compute routing
│  └─ Trigger async PCIe transfer (non-blocking)
├─ [8ms]  CPU: Other work (next layer prep, scheduling)
│         PCIe: Transfer expert weights (in parallel)
└─ [8ms]  GPU: Compute expert outputs
Total: 13-16ms (depends on overlap)
```

**Overlap Efficiency:**

- If transfer completes during CPU work: 13ms total (optimal)

- If transfer partially overlaps: 14-16ms total (typical)

- If transfer isn't ready: Fall back to CPU (safe, slower)

### Critical Implementation Details

1. **Async Transfer Initiation**

```cpp
// After expert selection (in build_moe_ffn or callback)
std::vector<int32_t> selected_experts = extract_from_ffn_moe_topk();

for (int expert_id : selected_experts) {
    ggml_tensor * expert_weight = lookup_expert_tensor(layer, expert_id);

    // Non-blocking transfer to GPU scratchpad
    cudaMemcpyAsync(
        gpu_scratchpad + offset,
        expert_weight->data,
        ggml_nbytes(expert_weight),
        cudaMemcpyHostToDevice,
        fetch_stream
    );
}
```

2. **Readiness Check (Non-Blocking)**

```cpp
// Before expert computation
bool weights_ready = (cudaStreamQuery(fetch_stream) == cudaSuccess);

if (weights_ready) {
    // Use GPU scratchpad weights
    ggml_cuda_mul_mat_id(..., gpu_scratchpad, ...);
} else {
    // Fallback: use CPU (slower but correct)
    ggml_cpu_mul_mat_id(..., cpu_weights, ...);
}
```

3. **No State, No Prediction**

   - Each token's experts are fetched independently

   - No attempt to predict future experts

   - No eviction policy (scratchpad is overwritten each layer)


## Architecture

### Component Overview

```
┌────────────────────────────────────────────────────────────┐
│  llama.cpp Core (MoE Graph Building)                       │
│  ├─ build_moe_ffn() - Expert selection callback point      │
│  └─ Callback: expert_trace_graph_cb()                      │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  Post-Fetch Hook (High-Level Integration)                  │
│  ├─ Extract selected expert IDs from callback              │
│  ├─ Lookup expert tensors in model                         │
│  └─ Initiate async transfers                               │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  CUDA Backend (Transfer Management)                        │
│  ├─ Dedicated fetch_stream (separate from compute)         │
│  ├─ GPU scratchpad buffer (persistent allocation)          │
│  └─ cudaMemcpyAsync() for non-blocking transfers           │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  Execution (Readiness Check)                               │
│  ├─ cudaStreamQuery() - Check if transfer complete         │
│  ├─ If ready: Use GPU scratchpad                           │
│  └─ If not ready: Fall back to CPU                         │
└────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Graph Building Phase:
   ┌─────────────────────────────────────────┐
   │ build_moe_ffn()                         │
   │ ├─ Create ffn_moe_topk tensor           │
   │ └─ Register graph callback              │
   └─────────────────────────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │ expert_trace_graph_cb()                 │
   │ └─ Note: MoE layer detected             │
   └─────────────────────────────────────────┘

2. Execution Phase:
   ┌─────────────────────────────────────────┐
   │ Eval Callback (GGML_OP_MUL_MAT_ID)      │  
   │ ├─ Extract expert IDs from ids tensor   │  
   │ └─ Trigger Post-Fetch hook              │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ Post-Fetch Transfer                     │  
   │ ├─ Lookup expert tensors                │  
   │ ├─ cudaMemcpyAsync() to scratchpad      │  
   │ └─ Return immediately (non-blocking)    │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ CPU Work (other computations)           │  
   │ └─ Overlap with PCIe transfer           │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ Expert Computation                      │  
   │ ├─ cudaStreamQuery(fetch_stream)        │
   │ ├─ If ready: Use GPU scratchpad         │
   │ └─ Else: Fall back to CPU               │  
   └─────────────────────────────────────────┘
```


## Implementation Details

### 1. Integration Points (Callback-Based)

#### Primary Integration: Eval Callback

```cpp
// Register eval callback during context initialization
void llama_postfetch_init(llama_context * ctx) {
    if (!is_postfetch_enabled()) {
        return;
    }
    
    // Set eval callback for expert tracking
    ggml_backend_sched_set_eval_callback(
        ctx->get_sched(),  // Use accessor method
        postfetch_eval_callback,
        ctx  // Pass context as user data
    );
    
    // Initialize CUDA resources
    postfetch_cuda_init(ctx);
}

// Eval callback implementation
bool postfetch_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    llama_context * ctx = (llama_context *) user_data;

    if (ask) {
        // We want to intercept MUL_MAT_ID operations
        return t->op == GGML_OP_MUL_MAT_ID;
    }

    // Extract expert IDs and initiate transfers
    if (t->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * ids = t->src[2];
        int layer_id = extract_layer_id(t->name);

        // Copy expert IDs to host (small tensor)
        std::vector<int32_t> expert_ids(ggml_nelements(ids));
        ggml_backend_tensor_get(ids, expert_ids.data(), 0, ggml_nbytes(ids));

        // Initiate async transfers
        postfetch_transfer_experts(ctx, layer_id, expert_ids);
    }

    return true;
}
```

#### Secondary Integration: Graph Callback (Optional)

```cpp
// Graph callback for early detection (optimization)
void postfetch_graph_callback(
    const llama_ubatch & ubatch,
    ggml_tensor * cur,
    const char * name,
    int il
) {
    // Pre-allocate resources when MoE layer is detected
    if (std::string(name) == "ffn_moe_topk") {
        postfetch_prepare_layer(il);
    }
}
```

### 2. Expert Tensor Lookup

**IMPORTANT: Tensor Structure Note**

The actual llama.cpp codebase uses a **single 3D tensor** for all experts, not per-expert tensors. This is a critical architectural difference that affects Post-Fetch implementation.

**Actual tensor structure:**
```cpp
// From src/llama-model.cpp:2826
layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), 
                                  {n_embd, n_ff, n_expert}, ...);
// Tensor name: "blk.5.ffn_gate_exps.weight"
// Dimensions: {n_embd, n_ff, n_expert} - contains ALL experts in one tensor
```

**Reference:** See [`tools/imatrix/imatrix.cpp:253-254`](../tools/imatrix/imatrix.cpp:253-254) for comment about this change.

**Post-Fetch with 3D Tensors:**

```cpp
struct postfetch_expert_tensors {
    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * down;
};

// Lookup expert tensors by layer and expert ID
// NOTE: This function calculates offsets into the 3D tensor structure
postfetch_expert_tensors lookup_expert_tensors(
    const llama_model * model,
    int layer_id,
    int expert_id
) {
    postfetch_expert_tensors result;
    
    // Access layer structure directly (tensors are pointers, not looked up by name)
    const llama_layer & layer = model->layers[layer_id];
    
    // Calculate offset into 3D tensor for this expert
    // Tensor dimensions: {n_embd, n_ff, n_expert}
    // Expert weights are at offset: expert_id * (n_embd * n_ff)
    size_t expert_offset = expert_id * (layer.ffn_gate_exps->ne[0] * layer.ffn_gate_exps->ne[1]);
    
    // Calculate sizes
    size_t gate_size = layer.ffn_gate_exps->ne[0] * layer.ffn_gate_exps->ne[1];
    size_t up_size = layer.ffn_up_exps->ne[0] * layer.ffn_up_exps->ne[1];
    size_t down_size = layer.ffn_down_exps->ne[0] * layer.ffn_down_exps->ne[1];
    
    // Create tensor views for individual expert slices
    // These are lightweight views into the 3D tensors
    result.gate = ggml_view_2d(layer.ctx_data, layer.ffn_gate_exps,
                                  expert_offset, gate_size);
    result.up = ggml_view_2d(layer.ctx_data, layer.ffn_up_exps,
                              expert_offset, up_size);
    result.down = ggml_view_2d(layer.ctx_data, layer.ffn_down_exps,
                                expert_offset, down_size);
    
    return result;
}
```

### 3. Async Transfer Implementation

```cpp
// CUDA resources
struct postfetch_cuda_state {
    void * scratchpad_ptr;       // GPU memory for expert weights
    size_t scratchpad_size;      // Total scratchpad size
    cudaStream_t fetch_stream;   // Dedicated stream for transfers
    cudaEvent_t sync_event;      // For cross-stream synchronization
};

static postfetch_cuda_state g_pf_cuda;

// Initialize CUDA resources
void postfetch_cuda_init(llama_context * ctx) {
    // Allocate scratchpad (size from env var or auto-calculate)
    const char* env_size = std::getenv("LLAMA_POSTFETCH_SCRATCHPAD_MB");
    size_t size_mb = env_size ? std::atoi(env_size) : 256;  // Default 256MB

    g_pf_cuda.scratchpad_size = size_mb * 1024 * 1024;
    cudaMalloc(&g_pf_cuda.scratchpad_ptr, g_pf_cuda.scratchpad_size);

    // Create dedicated stream
    cudaStreamCreate(&g_pf_cuda.fetch_stream);
    cudaEventCreate(&g_pf_cuda.sync_event);

    LLAMA_LOG_INFO("Post-Fetch initialized: scratchpad=%zu MB\n", size_mb);
}  
  
// Transfer experts to GPU asynchronously
void postfetch_transfer_experts(
    llama_context * ctx,
    int layer_id,
    const std::vector<int32_t> & expert_ids
) {
    size_t offset = 0;
    
    for (int32_t expert_id : expert_ids) {
        // Lookup expert tensors (using updated 3D tensor structure)
        auto tensors = lookup_expert_tensors(ctx->get_model(), layer_id, expert_id);
        
        // Transfer each tensor component
        for (ggml_tensor * t : {tensors.gate, tensors.up, tensors.down}) {
            if (!t) continue;
            
            size_t nbytes = ggml_nbytes(t);
            
            // Check scratchpad space
            if (offset + nbytes > g_pf_cuda.scratchpad_size) {
                LLAMA_LOG_WARN("Post-Fetch scratchpad full, skipping expert %d\n", expert_id);
                continue;
            }
            
            // Async copy to GPU
            cudaMemcpyAsync(
                (char*)g_pf_cuda.scratchpad_ptr + offset,
                t->data,
                nbytes,
                cudaMemcpyHostToDevice,
                g_pf_cuda.fetch_stream
            );
            
            // Store mapping (offset -> tensor) for later use
            postfetch_record_mapping(t, offset);
            
            offset += nbytes;
        }
    }
    
    // Record event for synchronization
    cudaEventRecord(g_pf_cuda.sync_event, g_pf_cuda.fetch_stream);
}
```

### 4. Readiness Check and Fallback

```cpp
// Check if transfers are complete (non-blocking)
bool postfetch_weights_ready() {
    cudaError_t status = cudaStreamQuery(g_pf_cuda.fetch_stream);
    return (status == cudaSuccess);
}

// Usage in CUDA backend (ggml_cuda_mul_mat_id or similar)
void ggml_cuda_mul_mat_id_postfetch(/* ... */) {
    if (postfetch_weights_ready()) {
        // Use GPU scratchpad (fast path)
        void * weight_ptr = postfetch_get_scratchpad_ptr(tensor);
        cuda_mul_mat_id_kernel<<<...>>>(input, weight_ptr, output);
    } else {
        // Fall back to CPU execution (slow but correct)
        LLAMA_LOG_DEBUG("Post-Fetch not ready, using CPU fallback\n");
        ggml_cpu_mul_mat_id(/* ... */);
    }
}
```

### 5. LoRA Compatibility

Post-Fetch works transparently with LoRA adapters when using the high-level callback integration:

```cpp
// In eval callback
bool postfetch_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    // The GGML_OP_MUL_MAT_ID operation already includes LoRA adapters
    // We just transfer the base weights; LoRA is applied separately

    // No special handling needed - LoRA is transparent at this level
    if (t->op == GGML_OP_MUL_MAT_ID) {
        // This works for both base model and LoRA-adapted models
        const ggml_tensor * ids = t->src[2];
        // ... standard processing
    }

    return true;
}
```

**Note:** If implementing at backend level, check tensor flags to ensure you're transferring base weights, not LoRA adapters (which are typically small and already on GPU).


## Configuration

### Environment Variables

| Variable | Type | Default | Description |
| - | - | - | - |
| `LLAMA_POSTFETCH_ENABLE` | `0` or `1` | `1` | Enable/disable Post-Fetch |
| `LLAMA_POSTFETCH_SCRATCHPAD_MB` | Integer | `256` | GPU scratchpad size in MB |
| `LLAMA_POSTFETCH_FORCE_CPU` | `0` or `1` | `0` | Force CPU fallback (testing) |
| `LLAMA_POSTFETCH_DEBUG` | `0` or `1` | `0` | Enable verbose logging |
| `LLAMA_EXPERT_TRACE_STATS` | `0` or `1` | `0` | Enable expert usage statistics |
| `LLAMA_EXPERT_TRACE_LOGGING` | `0` or `1` | `0` | Log expert IDs during execution |
| `LLAMA_EXPERT_TRACE_OUTPUT` | File path | (none) | Export statistics to JSON file |


### Usage Examples

```bash
# Standard usage (Post-Fetch enabled, minimal logging)
export LLAMA_POSTFETCH_ENABLE=1
export LLAMA_POSTFETCH_SCRATCHPAD_MB=512
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "Hello"

# Debugging (verbose logging, expert tracing)
export LLAMA_POSTFETCH_ENABLE=1
export LLAMA_POSTFETCH_DEBUG=1
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_LOGGING=1
export LLAMA_EXPERT_TRACE_OUTPUT=trace.json
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "Debug test"

# Testing fallback behavior
export LLAMA_POSTFETCH_FORCE_CPU=1  # Force CPU path
./llama-cli -m qwen3-next-80b-Q4_K_M.gguf -p "CPU fallback test"
```

### Auto-Configuration

The scratchpad size can be auto-calculated based on model architecture:

```cpp
size_t postfetch_calculate_scratchpad_size(const llama_model * model) {
    // Find largest MoE layer
    size_t max_layer_size = 0;

    for (int i = 0; i < model->n_layers; i++) {
        if (!model->layers[i].is_moe) continue;

        // Calculate size for max active experts
        int n_experts_active = model->layers[i].n_experts_active;
        size_t expert_size = /* calculate from tensor dimensions */;

        size_t layer_size = n_experts_active * expert_size * 3;  // gate + up + down
        max_layer_size = std::max(max_layer_size, layer_size);
    }

    // Add 20% safety margin
    return max_layer_size * 1.2;
}
```


## Performance Characteristics

### Expected Latency Reduction

| Hardware | Model | Without Post-Fetch | With Post-Fetch | Improvement |
| - | - | - | - | - |
| RTX 4060 Ti 16GB | Qwen3-Next-80B Q4_K_M | 23ms/token | 13-16ms/token | 30-43% |
| RTX 3090 24GB | GPT-OSS-120B Q4_K_M | 28ms/token | 16-20ms/token | 29-43% |
| RTX 4090 24GB | Qwen3-Coder-Next-80B Q4_K_M | 18ms/token | 11-14ms/token | 22-39% |


**Note:** Performance depends on overlap efficiency. Maximum gain occurs when transfer completes during CPU work.

### Overhead Analysis

| Component | Overhead | Impact |
| - | - | - |
| **Callback registration** | One-time (init) | Negligible |
| **Expert ID extraction** | ~0.1ms per layer | Negligible (small tensor copy) |
| **Tensor lookup** | ~0.01ms per expert | Negligible (hashtable lookup) |
| **Transfer initiation** | ~0.05ms per expert | Negligible (async call) |
| **Readiness check** | ~0.001ms | Negligible (single CUDA query) |
| **Total overhead** | <0.5ms per layer | <3% of layer time |


### Memory Usage

| Component | Size | Notes |
| - | - | - |
| **GPU scratchpad** | 256-512 MB | Configurable via env var |
| **Callback state** | <1 MB | Global or context-local |
| **Expert trace data** | <10 MB | Only if tracing enabled |
| **Total VRAM overhead** | 256-512 MB | One-time allocation |


### Scalability

Post-Fetch scales well with model size:

| Model Size | Expert Count | Active Experts | Scratchpad Size | Performance Gain |
| - | - | - | - | - |
| 20B (GPT-OSS) | 32 | 4 | 128 MB | 20-30% |
| 80B (Qwen3) | 512 | 10 | 512 MB | 30-43% |
| 120B (GPT-OSS) | 128 | 4 | 256 MB | 29-43% |


**Key Insight:** Larger models benefit more because:

1. More expert computation time → more overlap opportunity

2. Larger expert weights → PCIe transfer time dominates CPU work


## Future Extensions

### 1. Multi-GPU Support

Extend Post-Fetch to distribute expert transfers across multiple GPUs:

```cpp
// Round-robin expert assignment  
int gpu_id = expert_id % n_gpus;  
cudaSetDevice(gpu_id);  
cudaMemcpyAsync(/* ... */, fetch_streams[gpu_id]);
```

### 2. Predictive Prefetching (Optional)

Add lightweight expert prediction for sequential token generation:

```cpp
// Track expert co-occurrence patterns
std::unordered_map<int, std::vector<int>> expert_affinity;

// Prefetch likely next-token experts (low priority stream)
for (int likely_expert : expert_affinity[current_expert]) {
    cudaMemcpyAsync(/* ... */, prefetch_stream);  // Lower priority
}
```

**Note:** Keep this optional and disabled by default to maintain simplicity.

### 3. Dynamic Scratchpad Sizing

Adjust scratchpad size based on runtime VRAM availability:

```cpp
size_t free_vram, total_vram;
cudaMemGetInfo(&free_vram, &total_vram);

// Use up to 50% of free VRAM for scratchpad
size_t scratchpad_size = std::min(free_vram / 2, calculated_max_size);
```

### 4. Expert Compression

Compress expert weights in CPU RAM, decompress during transfer:

```cpp
// GPU decompression kernel (e.g., LZ4, Zstd)
decompress_kernel<<<...>>>(
    compressed_data,
    scratchpad_ptr,
    compression_metadata
);
```

**Trade-off:** Reduced PCIe transfer time vs. decompression overhead.


## Implementation Roadmap

### Phase 0: Single-File Expert Tracing Prototype (Week 1-2)

**Goal:** Implement expert usage tracking in a single, self-contained file for easy upstream proposal.

**File Structure:** All functionality in `common/expert-trace.cpp` + `common/expert-trace.h`

#### Design Principles for Upstream Acceptance

1. **Single compilation unit:** All implementation in one `.cpp` file

2. **Minimal API surface:** Clean header with only essential functions

3. **Zero dependencies on Post-Fetch:** Tracing works standalone

4. **OO encapsulation:** Use class/struct for state management

5. **Optional feature:** Controlled by environment variables, zero overhead when disabled

#### File: `common/expert-trace.h`

```cpp
// forgot the guards :)

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
 * Tracks which experts are activated during inference using callbacks.
 *
 * Configuration via environment variables:
 *   LLAMA_EXPERT_TRACE_STATS=1       - Enable activation counting
 *   LLAMA_EXPERT_TRACE_LOGGING=1     - Print expert IDs during execution
 *   LLAMA_EXPERT_TRACE_OUTPUT=<file> - Export statistics to JSON
 */
class expert_tracer {
public:
    // Configuration loaded from environment variables
    struct config {
        bool enable_stats = false;
        bool enable_logging = false;
        std::string output_file;

        // Load from environment variables
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
    void init(llama_context * ctx);

    // Cleanup and print statistics
    void cleanup(llama_context * ctx);

    // Callback functions (called by llama.cpp)
    void on_graph_build(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il);
    void on_eval(struct ggml_tensor * t, bool ask, llama_context * ctx);

    // Get current configuration
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
void llama_expert_trace_graph_cb(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il);
bool llama_expert_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

```

#### File: `src/llama-expert-trace.cpp`

**Note:** The implementation is placed in `src/` (not `common/`) because it uses the internal logging system from `llama-impl.h`.

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
    // Load configuration
    m_config.load_from_env();

    if (!m_config.enable_stats && !m_config.enable_logging) {
        return; // Tracing disabled, zero overhead
    }

    LLAMA_LOG_INFO("Expert tracing enabled (stats=%d, logging=%d)\n",
            m_config.enable_stats, m_config.enable_logging);

    // Clear any previous statistics
    m_layer_stats.clear();

    // Register eval callback
    // Note: Graph callback registration depends on llama.cpp API
    // This is a placeholder - actual registration may differ
    ggml_backend_sched_set_eval_callback(
        ctx->get_sched(),
        llama_expert_trace_eval_cb,
        ctx
    );
}

void llama_expert_tracer::cleanup(llama_context * ctx) {
    if (!m_config.enable_stats) {
        return;
    }

    // Print statistics
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
            LLAMA_LOG_INFO("  Expert %3d: %5d activations (%.1f%%)\n", expert_id, count, percentage);
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
    if (!m_config.enable_logging) {
        return;
    }

    // Detect MoE layers
    if (std::strcmp(name, "ffn_moe_topk") == 0) {
        LLAMA_LOG_DEBUG("[EXPERT-TRACE] Layer %d: MoE layer detected (tensor '%s')\n", il, name);
    }
}

void llama_expert_tracer::on_eval(struct ggml_tensor * t, bool ask, llama_context * ctx) {
    if (ask) {
        // We want to see MUL_MAT_ID operations (expert computation)
        return (t->op == GGML_OP_MUL_MAT_ID);
    }

    if (t->op != GGML_OP_MUL_MAT_ID) {
        return;
    }

    // Extract expert IDs from the operation
    // ids tensor is src[2]
    const ggml_tensor * ids = t->src[2];
    if (!ids) {
        return;
    }

    // Extract layer ID from tensor name
    int layer_id = extract_layer_id(t->name);
    if (layer_id < 0) {
        return; // Could not parse layer ID
    }

    // Copy expert IDs to host (small tensor, safe to copy)
    std::vector<int32_t> expert_ids(ggml_nelements(ids));
    ggml_backend_tensor_get(ids, expert_ids.data(), 0, ggml_nbytes(ids));

    // Update statistics
    for (int32_t expert_id : expert_ids) {
        record_expert_usage(layer_id, expert_id);
    }

    // Logging
    if (m_config.enable_logging) {
        std::stringstream ss;
        ss << "[EXPERT-TRACE] Layer " << layer_id << ": Experts [";
        for (size_t i = 0; i < expert_ids.size(); i++) {
            ss << expert_ids[i];
            if (i + 1 < expert_ids.size()) ss << ", ";
        }
        ss << "]\n";
        LLAMA_LOG_DEBUG("%s", ss.str().c_str());
    }
}  
  
//
// Internal Helpers
//

void llama_expert_tracer::record_expert_usage(int layer_id, int expert_id) {
    if (!m_config.enable_stats) {
        return;
    }

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
        if (!first_layer) file << ",\n";
        first_layer = false;

        file << "    {\n";
        file << "      \"layer_id\": " << layer_id << ",\n";
        file << "      \"total_tokens\": " << stats.total_tokens << ",\n";
        file << "      \"experts\": [\n";

        bool first_expert = true;
        for (const auto & [expert_id, count] : stats.expert_activations) {
            if (!first_expert) file << ",\n";
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
  
// ============================================================================  
// C-style callback wrappers  
// ============================================================================
void expert_trace_graph_cb(
    const llama_ubatch & ubatch,
    ggml_tensor * cur,
    const char * name,
    int il
) {
    expert_tracer::instance().on_graph_build(ubatch, cur, name, il);
}

bool expert_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    llama_context * ctx = static_cast<llama_context *>(user_data);
    expert_tracer::instance().on_eval(t, ask, ctx);
    return true;
}

} // namespace llama
```

#### Integration Points

**1. In `src/llama.cpp` (context initialization):**

```cpp
#include "common/expert-trace.h"

struct llama_context * llama_new_context_with_model(/* ... */) {
    // ... existing initialization ...

    // Initialize expert tracer
    llama::expert_tracer::instance().init(ctx);

    return ctx;
}
```

**2. In `src/llama.cpp` (context cleanup):**

```cpp
void llama_free(struct llama_context * ctx) {
    // Cleanup expert tracer
    llama::expert_tracer::instance().cleanup(ctx);

    // ... existing cleanup ...
}
```

**3. Build system (`CMakeLists.txt` or similar):**

```cpp
# Add expert-trace.cpp to common library
set(COMMON_SOURCES
    # ... existing sources ...
    common/expert-trace.cpp
)
```

#### Compilation

The commands to setup  and compile the software are:
cmake -B ../llama.cpp-build-cuda -DGGML_CUDA=ON
cmake --build ../llama.cpp-build-cuda --config Release -j 32


#### Testing Checklist

- [ ] Compile with expert-trace enabled

- [ ] Verify zero overhead when tracing disabled (env vars not set)

- [ ] Test with Qwen3-Next-80B: `LLAMA_EXPERT_TRACE_STATS=1 ./llama-cli -m ...`

- [ ] Test with GPT-OSS-120B: `LLAMA_EXPERT_TRACE_LOGGING=1 ./llama-cli -m ...`

- [ ] Verify JSON export: `LLAMA_EXPERT_TRACE_OUTPUT=stats.json ./llama-cli -m ...`

- [ ] Validate expert ID extraction accuracy (compare with manual inspection)

#### Advantages for Upstream Acceptance

| Aspect | Benefit |
| - | - |
| **Single file** | Easy to review, self-contained |
| **OO design** | Clean encapsulation, singleton pattern |
| **Zero overhead** | No cost when disabled (env vars not set) |
| **Optional feature** | Doesn't affect existing functionality |
| **Minimal API** | Only 2 public functions for integration |
| **No dependencies** | Only uses existing llama.cpp/ggml facilities |
| **Well documented** | Header comments explain usage |


**Deliverable:** Working expert tracer in a single file (`common/expert-trace.cpp`) ready for upstream PR.

### Phase 1: Post-Fetch Hook Integration (Week 3-4)

**Goal:** Connect expert tracing to Post-Fetch transfers (builds on Phase 0).

**Prerequisites:** Phase 0 completed (expert tracer working)

- [ ] Create `common/postfetch.cpp` + `common/postfetch.h` (following Phase 0 pattern)

- [ ] Implement `postfetch_transfer_experts()` with async CUDA

- [ ] Add expert tensor lookup (`lookup_expert_tensors`)

- [ ] Implement scratchpad allocation and management

- [ ] Hook into expert_tracer callbacks to trigger transfers

- [ ] Add readiness check and CPU fallback

- [ ] Test overlap efficiency (measure transfer vs computation time)

**Deliverable:** Basic Post-Fetch working, integrated with expert tracer from Phase 0.

### Phase 2: Production Hardening (Week 5-6)

**Goal:** Make Post-Fetch robust and production-ready.

- [ ] Add comprehensive error handling (CUDA errors, OOM, etc.)

- [ ] Implement auto-configuration (scratchpad sizing)

- [ ] Add performance monitoring (overlap metrics, fallback rate)

- [ ] Test with LoRA adapters

- [ ] Validate correctness (bit-exact output vs. baseline)

- [ ] Write documentation and usage guide

**Deliverable:** Production-ready Post-Fetch with documentation.

### Phase 3: Optimization (Week 7-8, Optional)

**Goal:** Squeeze out additional performance gains.

- [ ] Multi-stream optimization (overlap multiple transfers)

- [ ] Multi-GPU support (distribute experts)

- [ ] Dynamic scratchpad sizing based on VRAM availability

- [ ] Benchmark on diverse hardware (RTX 3090, 4060 Ti, 4090)

**Deliverable:** Optimized Post-Fetch with multi-GPU support.


## Common Error Messages and Solutions

| Error | Cause | Solution |
| - | - | - |
| `cudaErrorInvalidValue` | Incorrect scratchpad offset | Use cumulative offsets, not index × max_size |
| `cudaErrorMemoryAllocation` | Scratchpad too large | Reduce `LLAMA_POSTFETCH_SCRATCHPAD_MB` |
| `cudaErrorLaunchFailure` | Stream synchronization issue | Add `cudaEventRecord/Wait` for cross-stream sync |
| `Segmentation fault` | Tensor lookup failed | Check tensor name format, verify model structure |
| `cudaErrorNotReady` | Missing fallback logic | Implement CPU fallback when `cudaStreamQuery` fails |
| `Callback not invoked` | Wrong callback type | Verify `ggml_backend_sched_set_eval_callback` usage |



## Testing Strategy

### Unit Tests

- [ ] Verify callback registration and invocation

- [ ] Test expert ID extraction from `GGML_OP_MUL_MAT_ID`

- [ ] Validate tensor lookup by name

- [ ] Test scratchpad allocation and offset calculation

- [ ] Verify stream synchronization with timing measurements

### Integration Tests

- [ ] Run with Qwen3-Next-80B Q4_K_M

- [ ] Run with GPT-OSS-120B Q4_K_M

- [ ] Test with LoRA adapters (ensure compatibility)

- [ ] Verify identical outputs with/without Post-Fetch

### Performance Tests

- [ ] Measure transfer overlap percentage

- [ ] Track CPU fallback rate (should be <5%)

- [ ] Verify no regression on high-VRAM systems

- [ ] Test multi-threaded execution for race conditions


## Summary

**Post-Fetch with Callback-Based Expert Tracing** is a minimal, robust optimization:

- **Callback-based tracing:** Leverages existing `llm_graph_cb` and `ggml_backend_sched_eval_callback`

- **Non-invasive integration:** No modifications to MoE internals

- **Simple async transfers:** Fetch experts after selection, overlap with CPU work

- **Safe fallback:** Use CPU if GPU transfer isn't ready

- **Low overhead:** <1% for stats, <3% total including transfers

- **Production-ready:** Compatible with LoRA, multi-model, existing debug tools

It trades peak performance for **robustness, simplicity, and correctness** — exactly what low-VRAM llama.cpp users need.

### Key Advantages Over v0.0.4

| Aspect | v0.0.4 | v0.0.5 (Callback-Based) |
| - | - | - |
| **Integration** | Invasive (modify MoE code) | Non-invasive (register callbacks) |
| **Complexity** | High (custom instrumentation) | Low (use existing infrastructure) |
| **Maintenance** | High (track MoE changes) | Low (stable callback API) |
| **Compatibility** | Model-specific | All MoE models |
| **Debug Tools** | None | Works with `GGML_SCHED_DEBUG` |
| **Overhead** | Unknown | <1% (stats), <3% (total) |



*Document Version: 0.0.5 (Callback-Based Expert Tracing)* *Last Updated: 2026-02-09* *Leverages existing llama.cpp callback infrastructure for expert tracking*

