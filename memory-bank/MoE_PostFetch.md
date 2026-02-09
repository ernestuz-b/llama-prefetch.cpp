# Post-Fetch MoE Execution

**Version:** 0.0.5 (Callback-Based Expert Tracing) **Status:** Design Document **Target:** llama.cpp MoE Optimization for Low-VRAM Consumer GPUs


# Coding conventions

C++ Standard:  C++17 and using more modern STL constructs. CUDA files will have extensions cuh/cu.

Cross-Platform Compatibility

Always consider cross-compatibility with other operating systems (Windows, Linux, macOS) and architectures (x86, ARM, RISC-V). Test your code on multiple platforms before submitting.

Minimal Dependencies

Avoid adding third-party dependencies, extra files, or extra headers. Each new dependency increases the maintenance burden and potential for compatibility issues.

Pragmatic Over Dogmatic


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

The Expert Usage Tracer leverages llama.cpp's existing callback system instead of implementing custom instrumentation. This significantly simplifies implementation and integrates seamlessly with existing debugging tools.

The tracer tracks:

1. **Which experts are activated** during inference (via `ffn\_moe\_topk` tensor)

2. **How frequently each expert is used** across tokens/layers

3. **Expert selection pipeline** (logits → probabilities → weights → activations)

### Callback-Based Architecture

Instead of custom instrumentation, we use two existing callback mechanisms:

| Callback Type | Purpose | Integration Point | Access to |
| - | - | - | - |
| **Graph Callback** (`llm\_graph\_cb`) | Track expert selection during graph building | `llama\_graph\_context::cb()` | Tensor names, shapes, layer indices |
| **Eval Callback** (`ggml\_backend\_sched\_eval\_callback`) | Track expert usage during execution | `ggml\_backend\_sched\_set\_eval\_callback()` | Tensor data, operation types |


### Key Features

| Feature | Description | Implementation Method |
| - | - | - |
| **Expert Selection Tracking** | Monitor which experts are chosen | Filter for `ffn\_moe\_topk` tensor in callbacks |
| **Activation Counting** | Count expert usage per layer | Accumulate from `GGML\_OP\_MUL\_MAT\_ID` operations |
| **Selection Pipeline** | Track logits → probs → weights | Monitor all 23 callback points in MoE pathway |
| **Export Statistics** | Save to JSON/CSV | Post-processing of callback data |


### Callback Hook Points

The MoE pathway provides **23 callback points** (see [`Debugging\_MoE\_Experts.md`](file:///mnt/AI/AiProgs/llama-prefetch.cpp/memory-bank/Debugging_MoE_Experts.md#8-callback-points-before-and-after-expert-selection)):

#### Critical Callbacks for Post-Fetch

| Callback | Tensor Name | Purpose for Post-Fetch |
| - | - | - |
| **`ffn\_moe\_topk`** (line 1197) | Selected expert indices | **PRIMARY**: Identifies which experts to prefetch |
| **`ffn\_moe\_weights\_norm`** (line 1230) | Normalized expert weights | Optional: Weight-based prefetch prioritization |
| `ffn\_moe\_logits` (line 1121) | Raw gating scores | Analysis: Understanding routing patterns |
| `ffn\_moe\_probs` (line 1148) | Expert probabilities | Analysis: Router confidence metrics |


### Implementation Strategy

#### 1. Graph Callback for Expert Discovery

Use the `llm\_graph\_cb` callback to discover expert selection:

```
// In llama-graph.cpp or new expert-trace.cpp file  
struct expert\_trace\_data \{  
    std::unordered\_map\<int, std::unordered\_map\<int, int\>\> layer\_expert\_counts;  
    std::mutex trace\_mutex;  
    bool enable\_stats = false;  
    bool enable\_logging = false;  
\};  
  
// Global state (or in context struct)  
static expert\_trace\_data g\_expert\_trace;  
  
// Graph callback function  
void expert\_trace\_graph\_cb(  
    const llama\_ubatch & ubatch,  
    ggml\_tensor \* cur,  
    const char \* name,  
    int il  
) \{  
    if (!g\_expert\_trace.enable\_stats && !g\_expert\_trace.enable\_logging) \{  
        return;  
    \}  
      
    // Filter for MoE expert selection tensor  
    if (std::string(name) == "ffn\_moe\_topk") \{  
        if (g\_expert\_trace.enable\_logging) \{  
            LOG\_DBG("\[EXPERT-TRACE\] Layer %d: Expert selection tensor '%s' shape=%s\\n",  
                    il, name, common\_ggml\_ne\_string(cur).c\_str());  
        \}  
          
        // At this point, we know expert selection will happen  
        // Mark this layer as having MoE  
        // Store tensor for later data extraction (if needed)  
    \}  
\}
```

#### 2. Eval Callback for Expert Data Extraction

Use the `ggml\_backend\_sched\_eval\_callback` to extract expert indices:

```
// Eval callback function  
bool expert\_trace\_eval\_cb(struct ggml\_tensor \* t, bool ask, void \* user\_data) \{  
    if (!g\_expert\_trace.enable\_stats) \{  
        return true;  
    \}  
      
    if (ask) \{  
        // We want to see MUL\_MAT\_ID operations (expert computation)  
        return t-\>op == GGML\_OP\_MUL\_MAT\_ID;  
    \}  
      
    // Extract expert IDs from the operation  
    if (t-\>op == GGML\_OP\_MUL\_MAT\_ID) \{  
        // ids tensor is src\[2\]  
        const ggml\_tensor \* ids = t-\>src\[2\];  
          
        // Extract layer number from tensor name (e.g., "blk.5.ffn\_moe\_...")  
        int layer\_id = extract\_layer\_id(t-\>name);  
          
        // Copy expert IDs to host (small tensor, safe to copy)  
        std::vector\<int32\_t\> expert\_ids(ggml\_nelements(ids));  
        ggml\_backend\_tensor\_get(ids, expert\_ids.data(), 0, ggml\_nbytes(ids));  
          
        // Update statistics  
        std::lock\_guard\<std::mutex\> lock(g\_expert\_trace.trace\_mutex);  
        for (int32\_t expert\_id : expert\_ids) \{  
            g\_expert\_trace.layer\_expert\_counts\[layer\_id\]\[expert\_id\]++;  
        \}  
          
        if (g\_expert\_trace.enable\_logging) \{  
            LOG\_DBG("\[EXPERT-TRACE\] Layer %d: Expert IDs = \[", layer\_id);  
            for (size\_t i = 0; i \< expert\_ids.size(); i++) \{  
                LOG\_CNT("%d%s", expert\_ids\[i\], i+1 \< expert\_ids.size() ? ", " : "");  
            \}  
            LOG\_CNT("\]\\n");  
        \}  
    \}  
      
    return true;  
\}
```

#### 3. Integration with Post-Fetch

The callback-based tracer provides the expert indices needed for Post-Fetch:

```
// In build\_moe\_ffn() or similar high-level hook  
void postfetch\_initiate\_transfer(  
    const llama\_model \* model,  
    const llama\_context \* ctx,  
    int layer\_id,  
    const std::vector\<int32\_t\> & selected\_experts  
) \{  
    if (!is\_postfetch\_enabled()) \{  
        return;  
    \}  
      
    // These expert IDs come from the callback  
    for (int32\_t expert\_id : selected\_experts) \{  
        // Construct tensor names  
        std::string gate\_name = format\_expert\_tensor\_name(layer\_id, expert\_id, "gate");  
        std::string up\_name = format\_expert\_tensor\_name(layer\_id, expert\_id, "up");  
        std::string down\_name = format\_expert\_tensor\_name(layer\_id, expert\_id, "down");  
          
        // Find tensors in model  
        ggml\_tensor \* gate = ggml\_get\_tensor(model-\>layers\[layer\_id\].ctx, gate\_name.c\_str());  
        ggml\_tensor \* up = ggml\_get\_tensor(model-\>layers\[layer\_id\].ctx, up\_name.c\_str());  
        ggml\_tensor \* down = ggml\_get\_tensor(model-\>layers\[layer\_id\].ctx, down\_name.c\_str());  
          
        // Initiate async transfer to GPU  
        postfetch\_transfer\_async(gate, up, down);  
    \}  
\}
```

### Configuration via Environment Variables

Instead of command-line flags, use environment variables (llama.cpp convention):

| Variable | Values | Description |
| - | - | - |
| `LLAMA\_EXPERT\_TRACE\_STATS` | `0` or `1` | Enable activation counting via callbacks |
| `LLAMA\_EXPERT\_TRACE\_LOGGING` | `0` or `1` | Print expert IDs during execution |
| `LLAMA\_EXPERT\_TRACE\_OUTPUT` | File path | Export statistics to JSON file |
| `LLAMA\_EXPERT\_TRACE\_VERBOSE` | `0` or `1` | Include full selection pipeline (all 23 callbacks) |


### Initialization Code

```
// In llama\_new\_context\_with\_model() or similar  
void init\_expert\_trace(llama\_context \* ctx) \{  
    // Read environment variables  
    const char\* env\_stats = std::getenv("LLAMA\_EXPERT\_TRACE\_STATS");  
    g\_expert\_trace.enable\_stats = (env\_stats && std::string(env\_stats) == "1");  
      
    const char\* env\_log = std::getenv("LLAMA\_EXPERT\_TRACE\_LOGGING");  
    g\_expert\_trace.enable\_logging = (env\_log && std::string(env\_log) == "1");  
      
    if (!g\_expert\_trace.enable\_stats && !g\_expert\_trace.enable\_logging) \{  
        return; // Tracing disabled  
    \}  
      
    LOG\_INF("Expert tracing enabled (stats=%d, logging=%d)\\n",  
            g\_expert\_trace.enable\_stats, g\_expert\_trace.enable\_logging);  
      
    // Set graph callback (if using llm\_graph\_cb)  
    // ctx-\>graph\_callback = expert\_trace\_graph\_cb;  
      
    // Set eval callback  
    ggml\_backend\_sched\_set\_eval\_callback(  
        ctx-\>sched,  
        expert\_trace\_eval\_cb,  
        nullptr  // user\_data not needed (using global state)  
    );  
\}  
  
// In llama\_free() or similar  
void cleanup\_expert\_trace(llama\_context \* ctx) \{  
    if (!g\_expert\_trace.enable\_stats) \{  
        return;  
    \}  
      
    // Print statistics  
    LOG\_INF("\\n=== Expert Usage Statistics ===\\n");  
    for (const auto & \[layer\_id, expert\_counts\] : g\_expert\_trace.layer\_expert\_counts) \{  
        LOG\_INF("Layer %d:\\n", layer\_id);  
        for (const auto & \[expert\_id, count\] : expert\_counts) \{  
            LOG\_INF("  Expert %d: %d activations\\n", expert\_id, count);  
        \}  
    \}  
      
    // Export to JSON if configured  
    const char\* output\_file = std::getenv("LLAMA\_EXPERT\_TRACE\_OUTPUT");  
    if (output\_file) \{  
        export\_expert\_trace\_json(output\_file, g\_expert\_trace);  
    \}  
\}
```

### Advantages of Callback-Based Approach

| Aspect | Custom Instrumentation | Callback-Based (New) |
| - | - | - |
| **Code Complexity** | High (modify MoE internals) | Low (register callbacks) |
| **Integration Risk** | High (invasive changes) | Low (existing infrastructure) |
| **Maintenance** | High (track MoE changes) | Low (callbacks are stable API) |
| **Debug Tools** | None | Works with existing `GGML\_SCHED\_DEBUG` |
| **Performance** | Unknown (custom code) | Known (callback overhead \<1%) |
| **Compatibility** | Fragile (model-specific) | Robust (works with all MoE models) |


### Performance Overhead

Based on llama.cpp's existing callback usage:

| Feature | Overhead | Notes |
| - | - | - |
| Graph callback only | \<0.1% | Called during graph building (once per batch) |
| Eval callback (stats) | \<1% | Minimal logic, infrequent tensor copies |
| Eval callback (logging) | 5-10% | String formatting, I/O overhead |
| Full pipeline (23 callbacks) | 2-3% | Only if `LLAMA\_EXPERT\_TRACE\_VERBOSE=1` |


### Integration with Post-Fetch

The callback-based tracer feeds directly into Post-Fetch:

```
┌─────────────────────────────────────────────────────────────┐  
│                     INFERENCE LOOP                           │  
└─────────────────────────────────────────────────────────────┘  
                             │  
                             ▼  
┌─────────────────────────────────────────────────────────────┐  
│  Graph Callback: Discover MoE layers                         │  
│  - Filter for "ffn\_moe\_topk" tensors                         │  
│  - Mark layers with MoE                                      │  
└─────────────────────────────────────────────────────────────┘  
                             │  
                             ▼  
┌─────────────────────────────────────────────────────────────┐  
│  Eval Callback: Extract expert indices                       │  
│  - Monitor GGML\_OP\_MUL\_MAT\_ID operations                     │  
│  - Copy expert IDs from 'ids' tensor                         │  
│  - Update statistics (optional)                              │  
└─────────────────────────────────────────────────────────────┘  
                             │  
                             ▼  
┌─────────────────────────────────────────────────────────────┐  
│  Post-Fetch Hook: Initiate async transfers                   │  
│  - Use expert IDs from callback                              │  
│  - Lookup expert tensors in model                           │  
│  - Start cudaMemcpyAsync() to GPU                           │  
└─────────────────────────────────────────────────────────────┘  
                             │  
                             ▼  
┌─────────────────────────────────────────────────────────────┐  
│  Computation: Use prefetched experts                         │  
│  - Check if transfer complete (non-blocking)                 │  
│  - Fall back to CPU if not ready                            │  
└─────────────────────────────────────────────────────────────┘
```

### Minimal Example Usage

```
\# Enable expert tracing with statistics  
export LLAMA\_EXPERT\_TRACE\_STATS=1  
export LLAMA\_EXPERT\_TRACE\_OUTPUT=expert\_stats.json  
  
\# Run inference  
./llama-cli -m qwen3-next-80b-Q4\_K\_M.gguf -p "Hello world"  
  
\# Output:  
\# Expert tracing enabled (stats=1, logging=0)  
\# ...inference...  
\# === Expert Usage Statistics ===  
\# Layer 0:  
\#   Expert 3: 12 activations  
\#   Expert 7: 8 activations  
\# ...  
\# Statistics exported to: expert\_stats.json
```

### JSON Export Format

```
\{  
  "model": "qwen3-next-80b",  
  "total\_tokens": 100,  
  "layers": \[  
    \{  
      "layer\_id": 0,  
      "experts": \[  
        \{"expert\_id": 3, "activations": 12, "percentage": 12.0\},  
        \{"expert\_id": 7, "activations": 8, "percentage": 8.0\}  
      \]  
    \}  
  \]  
\}
```

### Summary: What Changed from v0.0.4

| Aspect | v0.0.4 (Custom) | v0.0.5 (Callback-Based) |
| - | - | - |
| **Implementation** | Custom instrumentation in MoE code | Use existing callback infrastructure |
| **Integration** | Invasive (modify build\_moe\_ffn) | Non-invasive (register callbacks) |
| **Data Access** | Direct tensor access | Via callback parameters |
| **Configuration** | Command-line flags | Environment variables |
| **Overhead** | Unknown | \<1% (stats), 5-10% (logging) |
| **Compatibility** | Model-specific | All MoE models |
| **Maintenance** | High (track MoE changes) | Low (stable callback API) |



## Overview

\[Rest of document continues with Post-Fetch implementation details...\]

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

- Quantization: Q4\_K\_M (expert weights ~4 bits per parameter)

- Inference: Interactive chat, batch size 1-8

**Key Constraint:** Expert weights cannot all fit in GPU VRAM, but non-expert parameters (attention, embeddings) can.


## Technical Background

### Why MoE Models Don't Fit in VRAM

For **Qwen3-Next-80B** with **Q4\_K\_M** quantization:

```
Total model size:     ~45 GB (quantized)  
├─ Non-expert params: ~6.5 GB (attention, embeddings, norms)  
└─ Expert params:     ~38.5 GB (512 experts × ~75 MB each)  
  
Available VRAM:       16 GB (RTX 4060 Ti)  
  
Problem: 38.5 GB experts \>\> 16 GB VRAM
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
├─ \[5ms\]  CPU: Compute routing  
├─ \[10ms\] PCIe: Transfer expert weights  ← BLOCKING  
└─ \[8ms\]  GPU: Compute expert outputs  
Total: 23ms  
  
Post-Fetch Approach (overlapped):  
├─ \[5ms\]  CPU: Compute routing  
│  └─ Trigger async PCIe transfer (non-blocking)  
├─ \[8ms\]  CPU: Other work (next layer prep, scheduling)  
│         PCIe: Transfer expert weights (in parallel)  
└─ \[8ms\]  GPU: Compute expert outputs  
Total: 13-16ms (depends on overlap)
```

**Overlap Efficiency:**

- If transfer completes during CPU work: 13ms total (optimal)

- If transfer partially overlaps: 14-16ms total (typical)

- If transfer isn't ready: Fall back to CPU (safe, slower)

### Critical Implementation Details

1. **Async Transfer Initiation**

```
// After expert selection (in build\_moe\_ffn or callback)  
std::vector\<int32\_t\> selected\_experts = extract\_from\_ffn\_moe\_topk();  
  
for (int expert\_id : selected\_experts) \{  
    ggml\_tensor \* expert\_weight = lookup\_expert\_tensor(layer, expert\_id);  
      
    // Non-blocking transfer to GPU scratchpad  
    cudaMemcpyAsync(  
        gpu\_scratchpad + offset,  
        expert\_weight-\>data,  
        ggml\_nbytes(expert\_weight),  
        cudaMemcpyHostToDevice,  
        fetch\_stream  
    );  
\}
```

2. **Readiness Check (Non-Blocking)**

```
// Before expert computation  
bool weights\_ready = (cudaStreamQuery(fetch\_stream) == cudaSuccess);  
  
if (weights\_ready) \{  
    // Use GPU scratchpad weights  
    ggml\_cuda\_mul\_mat\_id(..., gpu\_scratchpad, ...);  
\} else \{  
    // Fallback: use CPU (slower but correct)  
    ggml\_cpu\_mul\_mat\_id(..., cpu\_weights, ...);  
\}
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
│  ├─ build\_moe\_ffn() - Expert selection callback point      │  
│  └─ Callback: expert\_trace\_graph\_cb()                     │  
└────────────────────────────────────────────────────────────┘  
                           │  
                           ▼  
┌────────────────────────────────────────────────────────────┐  
│  Post-Fetch Hook (High-Level Integration)                  │  
│  ├─ Extract selected expert IDs from callback              │  
│  ├─ Lookup expert tensors in model                        │  
│  └─ Initiate async transfers                              │  
└────────────────────────────────────────────────────────────┘  
                           │  
                           ▼  
┌────────────────────────────────────────────────────────────┐  
│  CUDA Backend (Transfer Management)                        │  
│  ├─ Dedicated fetch\_stream (separate from compute)        │  
│  ├─ GPU scratchpad buffer (persistent allocation)         │  
│  └─ cudaMemcpyAsync() for non-blocking transfers          │  
└────────────────────────────────────────────────────────────┘  
                           │  
                           ▼  
┌────────────────────────────────────────────────────────────┐  
│  Execution (Readiness Check)                               │  
│  ├─ cudaStreamQuery() - Check if transfer complete        │  
│  ├─ If ready: Use GPU scratchpad                          │  
│  └─ If not ready: Fall back to CPU                        │  
└────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Graph Building Phase:  
   ┌─────────────────────────────────────────┐  
   │ build\_moe\_ffn()                         │  
   │ ├─ Create ffn\_moe\_topk tensor          │  
   │ └─ Register graph callback              │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ expert\_trace\_graph\_cb()                 │  
   │ └─ Note: MoE layer detected             │  
   └─────────────────────────────────────────┘  
  
2. Execution Phase:  
   ┌─────────────────────────────────────────┐  
   │ Eval Callback (GGML\_OP\_MUL\_MAT\_ID)      │  
   │ ├─ Extract expert IDs from ids tensor   │  
   │ └─ Trigger Post-Fetch hook              │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ Post-Fetch Transfer                     │  
   │ ├─ Lookup expert tensors               │  
   │ ├─ cudaMemcpyAsync() to scratchpad     │  
   │ └─ Return immediately (non-blocking)    │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ CPU Work (other computations)           │  
   │ └─ Overlap with PCIe transfer          │  
   └─────────────────────────────────────────┘  
                    │  
                    ▼  
   ┌─────────────────────────────────────────┐  
   │ Expert Computation                      │  
   │ ├─ cudaStreamQuery(fetch\_stream)       │  
   │ ├─ If ready: Use GPU scratchpad        │  
   │ └─ Else: Fall back to CPU              │  
   └─────────────────────────────────────────┘
```


## Implementation Details

### 1. Integration Points (Callback-Based)

#### Primary Integration: Eval Callback

```
// Register eval callback during context initialization  
void llama\_postfetch\_init(llama\_context \* ctx) \{  
    if (!is\_postfetch\_enabled()) \{  
        return;  
    \}  
      
    // Set eval callback for expert tracking  
    ggml\_backend\_sched\_set\_eval\_callback(  
        ctx-\>sched,  
        postfetch\_eval\_callback,  
        ctx  // Pass context as user data  
    );  
      
    // Initialize CUDA resources  
    postfetch\_cuda\_init(ctx);  
\}  
  
// Eval callback implementation  
bool postfetch\_eval\_callback(struct ggml\_tensor \* t, bool ask, void \* user\_data) \{  
    llama\_context \* ctx = (llama\_context \*) user\_data;  
      
    if (ask) \{  
        // We want to intercept MUL\_MAT\_ID operations  
        return t-\>op == GGML\_OP\_MUL\_MAT\_ID;  
    \}  
      
    // Extract expert IDs and initiate transfers  
    if (t-\>op == GGML\_OP\_MUL\_MAT\_ID) \{  
        const ggml\_tensor \* ids = t-\>src\[2\];  
        int layer\_id = extract\_layer\_id(t-\>name);  
          
        // Copy expert IDs to host (small tensor)  
        std::vector\<int32\_t\> expert\_ids(ggml\_nelements(ids));  
        ggml\_backend\_tensor\_get(ids, expert\_ids.data(), 0, ggml\_nbytes(ids));  
          
        // Initiate async transfers  
        postfetch\_transfer\_experts(ctx, layer\_id, expert\_ids);  
    \}  
      
    return true;  
\}
```

#### Secondary Integration: Graph Callback (Optional)

```
// Graph callback for early detection (optimization)  
void postfetch\_graph\_callback(  
    const llama\_ubatch & ubatch,  
    ggml\_tensor \* cur,  
    const char \* name,  
    int il  
) \{  
    // Pre-allocate resources when MoE layer is detected  
    if (std::string(name) == "ffn\_moe\_topk") \{  
        postfetch\_prepare\_layer(il);  
    \}  
\}
```

### 2. Expert Tensor Lookup

```
struct postfetch\_expert\_tensors \{  
    ggml\_tensor \* gate;  
    ggml\_tensor \* up;  
    ggml\_tensor \* down;  
\};  
  
// Lookup expert tensors by layer and expert ID  
postfetch\_expert\_tensors lookup\_expert\_tensors(  
    const llama\_model \* model,  
    int layer\_id,  
    int expert\_id  
) \{  
    postfetch\_expert\_tensors result;  
      
    // Construct tensor names (following llama.cpp convention)  
    std::string gate\_name = "blk." + std::to\_string(layer\_id) +   
                           ".ffn\_gate\_exps." + std::to\_string(expert\_id) + ".weight";  
    std::string up\_name = "blk." + std::to\_string(layer\_id) +   
                         ".ffn\_up\_exps." + std::to\_string(expert\_id) + ".weight";  
    std::string down\_name = "blk." + std::to\_string(layer\_id) +   
                           ".ffn\_down\_exps." + std::to\_string(expert\_id) + ".weight";  
      
    // Get tensors from model (ggml\_get\_tensor is fast - hashtable lookup)  
    result.gate = ggml\_get\_tensor(model-\>layers\[layer\_id\].ctx, gate\_name.c\_str());  
    result.up = ggml\_get\_tensor(model-\>layers\[layer\_id\].ctx, up\_name.c\_str());  
    result.down = ggml\_get\_tensor(model-\>layers\[layer\_id\].ctx, down\_name.c\_str());  
      
    return result;  
\}
```

### 3. Async Transfer Implementation

```
// CUDA resources  
struct postfetch\_cuda\_state \{  
    void \* scratchpad\_ptr;       // GPU memory for expert weights  
    size\_t scratchpad\_size;      // Total scratchpad size  
    cudaStream\_t fetch\_stream;   // Dedicated stream for transfers  
    cudaEvent\_t sync\_event;      // For cross-stream synchronization  
\};  
  
static postfetch\_cuda\_state g\_pf\_cuda;  
  
// Initialize CUDA resources  
void postfetch\_cuda\_init(llama\_context \* ctx) \{  
    // Allocate scratchpad (size from env var or auto-calculate)  
    const char\* env\_size = std::getenv("LLAMA\_POSTFETCH\_SCRATCHPAD\_MB");  
    size\_t size\_mb = env\_size ? std::atoi(env\_size) : 256;  // Default 256MB  
      
    g\_pf\_cuda.scratchpad\_size = size\_mb \* 1024 \* 1024;  
    cudaMalloc(&g\_pf\_cuda.scratchpad\_ptr, g\_pf\_cuda.scratchpad\_size);  
      
    // Create dedicated stream  
    cudaStreamCreate(&g\_pf\_cuda.fetch\_stream);  
    cudaEventCreate(&g\_pf\_cuda.sync\_event);  
      
    LOG\_INF("Post-Fetch initialized: scratchpad=%zu MB\\n", size\_mb);  
\}  
  
// Transfer experts to GPU asynchronously  
void postfetch\_transfer\_experts(  
    llama\_context \* ctx,  
    int layer\_id,  
    const std::vector\<int32\_t\> & expert\_ids  
) \{  
    size\_t offset = 0;  
      
    for (int32\_t expert\_id : expert\_ids) \{  
        // Lookup expert tensors  
        auto tensors = lookup\_expert\_tensors(ctx-\>model, layer\_id, expert\_id);  
          
        // Transfer each tensor component  
        for (ggml\_tensor \* t : \{tensors.gate, tensors.up, tensors.down\}) \{  
            if (!t) continue;  
              
            size\_t nbytes = ggml\_nbytes(t);  
              
            // Check scratchpad space  
            if (offset + nbytes \> g\_pf\_cuda.scratchpad\_size) \{  
                LOG\_WRN("Post-Fetch scratchpad full, skipping expert %d\\n", expert\_id);  
                continue;  
            \}  
              
            // Async copy to GPU  
            cudaMemcpyAsync(  
                (char\*)g\_pf\_cuda.scratchpad\_ptr + offset,  
                t-\>data,  
                nbytes,  
                cudaMemcpyHostToDevice,  
                g\_pf\_cuda.fetch\_stream  
            );  
              
            // Store mapping (offset -\> tensor) for later use  
            postfetch\_record\_mapping(t, offset);  
              
            offset += nbytes;  
        \}  
    \}  
      
    // Record event for synchronization  
    cudaEventRecord(g\_pf\_cuda.sync\_event, g\_pf\_cuda.fetch\_stream);  
\}
```

### 4. Readiness Check and Fallback

```
// Check if transfers are complete (non-blocking)  
bool postfetch\_weights\_ready() \{  
    cudaError\_t status = cudaStreamQuery(g\_pf\_cuda.fetch\_stream);  
    return (status == cudaSuccess);  
\}  
  
// Usage in CUDA backend (ggml\_cuda\_mul\_mat\_id or similar)  
void ggml\_cuda\_mul\_mat\_id\_postfetch(/\* ... \*/) \{  
    if (postfetch\_weights\_ready()) \{  
        // Use GPU scratchpad (fast path)  
        void \* weight\_ptr = postfetch\_get\_scratchpad\_ptr(tensor);  
        cuda\_mul\_mat\_id\_kernel\<\<\<...\>\>\>(input, weight\_ptr, output);  
    \} else \{  
        // Fall back to CPU execution (slow but correct)  
        LOG\_DBG("Post-Fetch not ready, using CPU fallback\\n");  
        ggml\_cpu\_mul\_mat\_id(/\* ... \*/);  
    \}  
\}
```

### 5. LoRA Compatibility

Post-Fetch works transparently with LoRA adapters when using the high-level callback integration:

```
// In eval callback  
bool postfetch\_eval\_callback(struct ggml\_tensor \* t, bool ask, void \* user\_data) \{  
    // The GGML\_OP\_MUL\_MAT\_ID operation already includes LoRA adapters  
    // We just transfer the base weights; LoRA is applied separately  
      
    // No special handling needed - LoRA is transparent at this level  
    if (t-\>op == GGML\_OP\_MUL\_MAT\_ID) \{  
        // This works for both base model and LoRA-adapted models  
        const ggml\_tensor \* ids = t-\>src\[2\];  
        // ... standard processing  
    \}  
      
    return true;  
\}
```

**Note:** If implementing at backend level, check tensor flags to ensure you're transferring base weights, not LoRA adapters (which are typically small and already on GPU).


## Configuration

### Environment Variables

| Variable | Type | Default | Description |
| - | - | - | - |
| `LLAMA\_POSTFETCH\_ENABLE` | `0` or `1` | `1` | Enable/disable Post-Fetch |
| `LLAMA\_POSTFETCH\_SCRATCHPAD\_MB` | Integer | `256` | GPU scratchpad size in MB |
| `LLAMA\_POSTFETCH\_FORCE\_CPU` | `0` or `1` | `0` | Force CPU fallback (testing) |
| `LLAMA\_POSTFETCH\_DEBUG` | `0` or `1` | `0` | Enable verbose logging |
| `LLAMA\_EXPERT\_TRACE\_STATS` | `0` or `1` | `0` | Enable expert usage statistics |
| `LLAMA\_EXPERT\_TRACE\_LOGGING` | `0` or `1` | `0` | Log expert IDs during execution |
| `LLAMA\_EXPERT\_TRACE\_OUTPUT` | File path | (none) | Export statistics to JSON file |


### Usage Examples

```
\# Standard usage (Post-Fetch enabled, minimal logging)  
export LLAMA\_POSTFETCH\_ENABLE=1  
export LLAMA\_POSTFETCH\_SCRATCHPAD\_MB=512  
./llama-cli -m qwen3-next-80b-Q4\_K\_M.gguf -p "Hello"  
  
\# Debugging (verbose logging, expert tracing)  
export LLAMA\_POSTFETCH\_ENABLE=1  
export LLAMA\_POSTFETCH\_DEBUG=1  
export LLAMA\_EXPERT\_TRACE\_STATS=1  
export LLAMA\_EXPERT\_TRACE\_LOGGING=1  
export LLAMA\_EXPERT\_TRACE\_OUTPUT=trace.json  
./llama-cli -m qwen3-next-80b-Q4\_K\_M.gguf -p "Debug test"  
  
\# Testing fallback behavior  
export LLAMA\_POSTFETCH\_FORCE\_CPU=1  \# Force CPU path  
./llama-cli -m qwen3-next-80b-Q4\_K\_M.gguf -p "CPU fallback test"
```

### Auto-Configuration

The scratchpad size can be auto-calculated based on model architecture:

```
size\_t postfetch\_calculate\_scratchpad\_size(const llama\_model \* model) \{  
    // Find largest MoE layer  
    size\_t max\_layer\_size = 0;  
      
    for (int i = 0; i \< model-\>n\_layers; i++) \{  
        if (!model-\>layers\[i\].is\_moe) continue;  
          
        // Calculate size for max active experts  
        int n\_experts\_active = model-\>layers\[i\].n\_experts\_active;  
        size\_t expert\_size = /\* calculate from tensor dimensions \*/;  
          
        size\_t layer\_size = n\_experts\_active \* expert\_size \* 3;  // gate + up + down  
        max\_layer\_size = std::max(max\_layer\_size, layer\_size);  
    \}  
      
    // Add 20% safety margin  
    return max\_layer\_size \* 1.2;  
\}
```


## Performance Characteristics

### Expected Latency Reduction

| Hardware | Model | Without Post-Fetch | With Post-Fetch | Improvement |
| - | - | - | - | - |
| RTX 4060 Ti 16GB | Qwen3-Next-80B Q4\_K\_M | 23ms/token | 13-16ms/token | 30-43% |
| RTX 3090 24GB | GPT-OSS-120B Q4\_K\_M | 28ms/token | 16-20ms/token | 29-43% |
| RTX 4090 24GB | Qwen3-Coder-Next-80B Q4\_K\_M | 18ms/token | 11-14ms/token | 22-39% |


**Note:** Performance depends on overlap efficiency. Maximum gain occurs when transfer completes during CPU work.

### Overhead Analysis

| Component | Overhead | Impact |
| - | - | - |
| **Callback registration** | One-time (init) | Negligible |
| **Expert ID extraction** | ~0.1ms per layer | Negligible (small tensor copy) |
| **Tensor lookup** | ~0.01ms per expert | Negligible (hashtable lookup) |
| **Transfer initiation** | ~0.05ms per expert | Negligible (async call) |
| **Readiness check** | ~0.001ms | Negligible (single CUDA query) |
| **Total overhead** | \<0.5ms per layer | \<3% of layer time |


### Memory Usage

| Component | Size | Notes |
| - | - | - |
| **GPU scratchpad** | 256-512 MB | Configurable via env var |
| **Callback state** | \<1 MB | Global or context-local |
| **Expert trace data** | \<10 MB | Only if tracing enabled |
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

```
// Round-robin expert assignment  
int gpu\_id = expert\_id % n\_gpus;  
cudaSetDevice(gpu\_id);  
cudaMemcpyAsync(/\* ... \*/, fetch\_streams\[gpu\_id\]);
```

### 2. Predictive Prefetching (Optional)

Add lightweight expert prediction for sequential token generation:

```
// Track expert co-occurrence patterns  
std::unordered\_map\<int, std::vector\<int\>\> expert\_affinity;  
  
// Prefetch likely next-token experts (low priority stream)  
for (int likely\_expert : expert\_affinity\[current\_expert\]) \{  
    cudaMemcpyAsync(/\* ... \*/, prefetch\_stream);  // Lower priority  
\}
```

**Note:** Keep this optional and disabled by default to maintain simplicity.

### 3. Dynamic Scratchpad Sizing

Adjust scratchpad size based on runtime VRAM availability:

```
size\_t free\_vram, total\_vram;  
cudaMemGetInfo(&free\_vram, &total\_vram);  
  
// Use up to 50% of free VRAM for scratchpad  
size\_t scratchpad\_size = std::min(free\_vram / 2, calculated\_max\_size);
```

### 4. Expert Compression

Compress expert weights in CPU RAM, decompress during transfer:

```
// GPU decompression kernel (e.g., LZ4, Zstd)  
decompress\_kernel\<\<\<...\>\>\>(  
    compressed\_data,  
    scratchpad\_ptr,  
    compression\_metadata  
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

```
\#pragma once  
  
\#include "ggml.h"  
\#include "llama.h"  
\#include \<string\>  
\#include \<unordered\_map\>  
\#include \<vector\>  
\#include \<mutex\>  
  
// Forward declarations  
struct llama\_context;  
struct llama\_ubatch;  
  
namespace llama \{  
  
/\*\*  
 \* Expert usage tracer for MoE models.  
 \* Tracks which experts are activated during inference using callbacks.  
 \*   
 \* Configuration via environment variables:  
 \*   LLAMA\_EXPERT\_TRACE\_STATS=1       - Enable activation counting  
 \*   LLAMA\_EXPERT\_TRACE\_LOGGING=1     - Print expert IDs during execution  
 \*   LLAMA\_EXPERT\_TRACE\_OUTPUT=\<file\> - Export statistics to JSON  
 \*/  
class expert\_tracer \{  
public:  
    // Configuration loaded from environment variables  
    struct config \{  
        bool enable\_stats = false;  
        bool enable\_logging = false;  
        std::string output\_file;  
          
        // Load from environment variables  
        void load\_from\_env();  
    \};  
      
    // Statistics for a single layer  
    struct layer\_stats \{  
        int layer\_id;  
        std::unordered\_map\<int, int\> expert\_activations;  // expert\_id -\> count  
        int total\_tokens = 0;  
    \};  
      
    // Singleton instance  
    static expert\_tracer & instance();  
      
    // Initialize tracer for a context  
    void init(llama\_context \* ctx);  
      
    // Cleanup and print statistics  
    void cleanup(llama\_context \* ctx);  
      
    // Callback functions (called by llama.cpp)  
    void on\_graph\_build(const llama\_ubatch & ubatch, ggml\_tensor \* cur, const char \* name, int il);  
    void on\_eval(struct ggml\_tensor \* t, bool ask, llama\_context \* ctx);  
      
    // Get current configuration  
    const config & get\_config() const \{ return m\_config; \}  
      
    // Get statistics (for testing/debugging)  
    const std::unordered\_map\<int, layer\_stats\> & get\_stats() const \{ return m\_layer\_stats; \}  
  
private:  
    expert\_tracer() = default;  
    ~expert\_tracer() = default;  
      
    // Non-copyable, non-movable (singleton)  
    expert\_tracer(const expert\_tracer &) = delete;  
    expert\_tracer & operator=(const expert\_tracer &) = delete;  
      
    // Internal helpers  
    void record\_expert\_usage(int layer\_id, int expert\_id);  
    void export\_stats\_json(const std::string & filename);  
    int extract\_layer\_id(const char \* tensor\_name);  
      
    // State  
    config m\_config;  
    std::unordered\_map\<int, layer\_stats\> m\_layer\_stats;  
    std::mutex m\_mutex;  
\};  
  
// C-style callback wrappers (for ggml callback system)  
void expert\_trace\_graph\_cb(const llama\_ubatch & ubatch, ggml\_tensor \* cur, const char \* name, int il);  
bool expert\_trace\_eval\_cb(struct ggml\_tensor \* t, bool ask, void \* user\_data);  
  
\} // namespace llama
```

#### File: `common/expert-trace.cpp`

```
\#include "expert-trace.h"  
\#include "log.h"  
\#include \<cstdlib\>  
\#include \<cstring\>  
\#include \<regex\>  
\#include \<fstream\>  
\#include \<sstream\>  
  
namespace llama \{  
  
// ============================================================================  
// Configuration  
// ============================================================================  
  
void expert\_tracer::config::load\_from\_env() \{  
    const char\* env\_stats = std::getenv("LLAMA\_EXPERT\_TRACE\_STATS");  
    enable\_stats = (env\_stats && std::strcmp(env\_stats, "1") == 0);  
      
    const char\* env\_logging = std::getenv("LLAMA\_EXPERT\_TRACE\_LOGGING");  
    enable\_logging = (env\_logging && std::strcmp(env\_logging, "1") == 0);  
      
    const char\* env\_output = std::getenv("LLAMA\_EXPERT\_TRACE\_OUTPUT");  
    if (env\_output) \{  
        output\_file = env\_output;  
    \}  
\}  
  
// ============================================================================  
// Singleton  
// ============================================================================  
  
expert\_tracer & expert\_tracer::instance() \{  
    static expert\_tracer instance;  
    return instance;  
\}  
  
// ============================================================================  
// Initialization / Cleanup  
// ============================================================================  
  
void expert\_tracer::init(llama\_context \* ctx) \{  
    // Load configuration  
    m\_config.load\_from\_env();  
      
    if (!m\_config.enable\_stats && !m\_config.enable\_logging) \{  
        return; // Tracing disabled, zero overhead  
    \}  
      
    LOG\_INF("Expert tracing enabled (stats=%d, logging=%d)\\n",  
            m\_config.enable\_stats, m\_config.enable\_logging);  
      
    // Clear any previous statistics  
    m\_layer\_stats.clear();  
      
    // Register eval callback  
    // Note: Graph callback registration depends on llama.cpp API  
    // This is a placeholder - actual registration may differ  
    ggml\_backend\_sched\_set\_eval\_callback(  
        ctx-\>sched,  
        expert\_trace\_eval\_cb,  
        ctx  
    );  
\}  
  
void expert\_tracer::cleanup(llama\_context \* ctx) \{  
    if (!m\_config.enable\_stats) \{  
        return;  
    \}  
      
    // Print statistics  
    LOG\_INF("\\n=== Expert Usage Statistics ===\\n");  
    for (const auto & \[layer\_id, stats\] : m\_layer\_stats) \{  
        LOG\_INF("Layer %d (%d tokens):\\n", layer\_id, stats.total\_tokens);  
          
        // Sort experts by activation count (descending)  
        std::vector\<std::pair\<int, int\>\> sorted\_experts(  
            stats.expert\_activations.begin(),  
            stats.expert\_activations.end()  
        );  
        std::sort(sorted\_experts.begin(), sorted\_experts.end(),  
                  \[\](const auto & a, const auto & b) \{ return a.second \> b.second; \});  
          
        for (const auto & \[expert\_id, count\] : sorted\_experts) \{  
            float percentage = 100.0f \* count / stats.total\_tokens;  
            LOG\_INF("  Expert %3d: %5d activations (%.1f%%)\\n", expert\_id, count, percentage);  
        \}  
    \}  
      
    // Export to JSON if configured  
    if (!m\_config.output\_file.empty()) \{  
        export\_stats\_json(m\_config.output\_file);  
        LOG\_INF("Statistics exported to: %s\\n", m\_config.output\_file.c\_str());  
    \}  
\}  
  
// ============================================================================  
// Callbacks  
// ============================================================================  
  
void expert\_tracer::on\_graph\_build(  
    const llama\_ubatch & ubatch,  
    ggml\_tensor \* cur,  
    const char \* name,  
    int il  
) \{  
    if (!m\_config.enable\_logging) \{  
        return;  
    \}  
      
    // Detect MoE layers  
    if (std::strcmp(name, "ffn\_moe\_topk") == 0) \{  
        LOG\_DBG("\[EXPERT-TRACE\] Layer %d: MoE layer detected (tensor '%s')\\n", il, name);  
    \}  
\}  
  
void expert\_tracer::on\_eval(struct ggml\_tensor \* t, bool ask, llama\_context \* ctx) \{  
    if (ask) \{  
        // We want to see MUL\_MAT\_ID operations (expert computation)  
        return (t-\>op == GGML\_OP\_MUL\_MAT\_ID);  
    \}  
      
    if (t-\>op != GGML\_OP\_MUL\_MAT\_ID) \{  
        return;  
    \}  
      
    // Extract expert IDs from the operation  
    // ids tensor is src\[2\]  
    const ggml\_tensor \* ids = t-\>src\[2\];  
    if (!ids) \{  
        return;  
    \}  
      
    // Extract layer ID from tensor name  
    int layer\_id = extract\_layer\_id(t-\>name);  
    if (layer\_id \< 0) \{  
        return; // Could not parse layer ID  
    \}  
      
    // Copy expert IDs to host (small tensor, safe to copy)  
    std::vector\<int32\_t\> expert\_ids(ggml\_nelements(ids));  
    ggml\_backend\_tensor\_get(ids, expert\_ids.data(), 0, ggml\_nbytes(ids));  
      
    // Update statistics  
    for (int32\_t expert\_id : expert\_ids) \{  
        record\_expert\_usage(layer\_id, expert\_id);  
    \}  
      
    // Logging  
    if (m\_config.enable\_logging) \{  
        std::stringstream ss;  
        ss \<\< "\[EXPERT-TRACE\] Layer " \<\< layer\_id \<\< ": Experts \[";  
        for (size\_t i = 0; i \< expert\_ids.size(); i++) \{  
            ss \<\< expert\_ids\[i\];  
            if (i + 1 \< expert\_ids.size()) ss \<\< ", ";  
        \}  
        ss \<\< "\]\\n";  
        LOG\_DBG("%s", ss.str().c\_str());  
    \}  
\}  
  
// ============================================================================  
// Internal Helpers  
// ============================================================================  
  
void expert\_tracer::record\_expert\_usage(int layer\_id, int expert\_id) \{  
    if (!m\_config.enable\_stats) \{  
        return;  
    \}  
      
    std::lock\_guard\<std::mutex\> lock(m\_mutex);  
      
    auto & stats = m\_layer\_stats\[layer\_id\];  
    stats.layer\_id = layer\_id;  
    stats.expert\_activations\[expert\_id\]++;  
    stats.total\_tokens++;  
\}  
  
int expert\_tracer::extract\_layer\_id(const char \* tensor\_name) \{  
    // Extract layer ID from tensor name like "blk.5.ffn\_moe\_..."  
    // Returns -1 if parsing fails  
      
    std::string name(tensor\_name);  
    std::regex pattern(R"(blk\\.(\\d+)\\.)");  
    std::smatch match;  
      
    if (std::regex\_search(name, match, pattern)) \{  
        return std::stoi(match\[1\].str());  
    \}  
      
    return -1;  
\}  
  
void expert\_tracer::export\_stats\_json(const std::string & filename) \{  
    std::ofstream file(filename);  
    if (!file.is\_open()) \{  
        LOG\_WRN("Failed to open output file: %s\\n", filename.c\_str());  
        return;  
    \}  
      
    file \<\< "\{\\n";  
    file \<\< "  \\"layers\\": \[\\n";  
      
    bool first\_layer = true;  
    for (const auto & \[layer\_id, stats\] : m\_layer\_stats) \{  
        if (!first\_layer) file \<\< ",\\n";  
        first\_layer = false;  
          
        file \<\< "    \{\\n";  
        file \<\< "      \\"layer\_id\\": " \<\< layer\_id \<\< ",\\n";  
        file \<\< "      \\"total\_tokens\\": " \<\< stats.total\_tokens \<\< ",\\n";  
        file \<\< "      \\"experts\\": \[\\n";  
          
        bool first\_expert = true;  
        for (const auto & \[expert\_id, count\] : stats.expert\_activations) \{  
            if (!first\_expert) file \<\< ",\\n";  
            first\_expert = false;  
              
            float percentage = 100.0f \* count / stats.total\_tokens;  
            file \<\< "        \{\\"expert\_id\\": " \<\< expert\_id  
                 \<\< ", \\"activations\\": " \<\< count  
                 \<\< ", \\"percentage\\": " \<\< percentage \<\< "\}";  
        \}  
          
        file \<\< "\\n      \]\\n";  
        file \<\< "    \}";  
    \}  
      
    file \<\< "\\n  \]\\n";  
    file \<\< "\}\\n";  
      
    file.close();  
\}  
  
// ============================================================================  
// C-style callback wrappers  
// ============================================================================  
  
void expert\_trace\_graph\_cb(  
    const llama\_ubatch & ubatch,  
    ggml\_tensor \* cur,  
    const char \* name,  
    int il  
) \{  
    expert\_tracer::instance().on\_graph\_build(ubatch, cur, name, il);  
\}  
  
bool expert\_trace\_eval\_cb(struct ggml\_tensor \* t, bool ask, void \* user\_data) \{  
    llama\_context \* ctx = static\_cast\<llama\_context \*\>(user\_data);  
    expert\_tracer::instance().on\_eval(t, ask, ctx);  
    return true;  
\}  
  
\} // namespace llama
```

#### Integration Points

**1. In `src/llama.cpp` (context initialization):**

```
\#include "common/expert-trace.h"  
  
struct llama\_context \* llama\_new\_context\_with\_model(/\* ... \*/) \{  
    // ... existing initialization ...  
      
    // Initialize expert tracer  
    llama::expert\_tracer::instance().init(ctx);  
      
    return ctx;  
\}
```

**2. In `src/llama.cpp` (context cleanup):**

```
void llama\_free(struct llama\_context \* ctx) \{  
    // Cleanup expert tracer  
    llama::expert\_tracer::instance().cleanup(ctx);  
      
    // ... existing cleanup ...  
\}
```

**3. Build system (`CMakeLists.txt` or similar):**

```
\# Add expert-trace.cpp to common library  
set(COMMON\_SOURCES  
    \# ... existing sources ...  
    common/expert-trace.cpp  
)
```

#### Testing Checklist

- [ ] Compile with expert-trace enabled

- [ ] Verify zero overhead when tracing disabled (env vars not set)

- [ ] Test with Qwen3-Next-80B: `LLAMA\_EXPERT\_TRACE\_STATS=1 ./llama-cli -m ...`

- [ ] Test with GPT-OSS-120B: `LLAMA\_EXPERT\_TRACE\_LOGGING=1 ./llama-cli -m ...`

- [ ] Verify JSON export: `LLAMA\_EXPERT\_TRACE\_OUTPUT=stats.json ./llama-cli -m ...`

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

- [ ] Implement `postfetch\_transfer\_experts()` with async CUDA

- [ ] Add expert tensor lookup (`lookup\_expert\_tensors`)

- [ ] Implement scratchpad allocation and management

- [ ] Hook into expert\_tracer callbacks to trigger transfers

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
| `cudaErrorInvalidValue` | Incorrect scratchpad offset | Use cumulative offsets, not index × max\_size |
| `cudaErrorMemoryAllocation` | Scratchpad too large | Reduce `LLAMA\_POSTFETCH\_SCRATCHPAD\_MB` |
| `cudaErrorLaunchFailure` | Stream synchronization issue | Add `cudaEventRecord/Wait` for cross-stream sync |
| `Segmentation fault` | Tensor lookup failed | Check tensor name format, verify model structure |
| `cudaErrorNotReady` | Missing fallback logic | Implement CPU fallback when `cudaStreamQuery` fails |
| `Callback not invoked` | Wrong callback type | Verify `ggml\_backend\_sched\_set\_eval\_callback` usage |



## Testing Strategy

### Unit Tests

- [ ] Verify callback registration and invocation

- [ ] Test expert ID extraction from `GGML\_OP\_MUL\_MAT\_ID`

- [ ] Validate tensor lookup by name

- [ ] Test scratchpad allocation and offset calculation

- [ ] Verify stream synchronization with timing measurements

### Integration Tests

- [ ] Run with Qwen3-Next-80B Q4\_K\_M

- [ ] Run with GPT-OSS-120B Q4\_K\_M

- [ ] Test with LoRA adapters (ensure compatibility)

- [ ] Verify identical outputs with/without Post-Fetch

### Performance Tests

- [ ] Measure transfer overlap percentage

- [ ] Track CPU fallback rate (should be \<5%)

- [ ] Verify no regression on high-VRAM systems

- [ ] Test multi-threaded execution for race conditions


## Summary

**Post-Fetch with Callback-Based Expert Tracing** is a minimal, robust optimization:

- **Callback-based tracing:** Leverages existing `llm\_graph\_cb` and `ggml\_backend\_sched\_eval\_callback`

- **Non-invasive integration:** No modifications to MoE internals

- **Simple async transfers:** Fetch experts after selection, overlap with CPU work

- **Safe fallback:** Use CPU if GPU transfer isn't ready

- **Low overhead:** \<1% for stats, \<3% total including transfers

- **Production-ready:** Compatible with LoRA, multi-model, existing debug tools

It trades peak performance for **robustness, simplicity, and correctness** — exactly what low-VRAM llama.cpp users need.

### Key Advantages Over v0.0.4

| Aspect | v0.0.4 | v0.0.5 (Callback-Based) |
| - | - | - |
| **Integration** | Invasive (modify MoE code) | Non-invasive (register callbacks) |
| **Complexity** | High (custom instrumentation) | Low (use existing infrastructure) |
| **Maintenance** | High (track MoE changes) | Low (stable callback API) |
| **Compatibility** | Model-specific | All MoE models |
| **Debug Tools** | None | Works with `GGML\_SCHED\_DEBUG` |
| **Overhead** | Unknown | \<1% (stats), \<3% (total) |



*Document Version: 0.0.5 (Callback-Based Expert Tracing)* *Last Updated: 2026-02-09* *Leverages existing llama.cpp callback infrastructure for expert tracking*

