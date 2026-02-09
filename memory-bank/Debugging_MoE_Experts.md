# Debug Capabilities for Tracking Expert Usage in MoE Pathway

This document describes the debug capabilities available in llama.cpp for tracking which experts are used in Mixture of Experts (MoE) models.

## Important Note: Existing vs. Planned Features

**This document describes the CURRENTLY AVAILABLE debug capabilities in llama.cpp.**

A separate design document, [`MoE_PostFetch_V0_0_4.md`](MoE_PostFetch_V0_0_4.md), describes **Task 0: Expert Usage Tracer**, which is a **planned feature** that is NOT yet implemented in the codebase.

### Planned Features (Not Yet Implemented)

The following features are described in [`MoE_PostFetch_V0_0_4.md`](MoE_PostFetch_V0_0_4.md) but are **NOT currently available**:

| Feature | Description | Status |
|---------|-------------|--------|
| `LLAMA_EXPERT_TRACE_STATS` | Environment variable to enable activation counting | **Planned** |
| `LLAMA_EXPERT_TRACE_NAMES` | Environment variable to print tensor names | **Planned** |
| `LLAMA_EXPERT_TRACE_PER_LAYER` | Environment variable for per-layer statistics | **Planned** |
| `LLAMA_EXPERT_TRACE_OUTPUT` | Environment variable to export to JSON/CSV | **Planned** |
| `expert_tracer_record_usage()` | Function to record expert usage | **Planned** |
| `expert_tracer_print_stats()` | Function to print statistics | **Planned** |

### Currently Available Debug Capabilities

The following debug capabilities are **currently available** in llama.cpp:

1. **Graph Callback System** (`llm_graph_cb`) - Track tensors during graph building
2. **`GGML_OP_MUL_MAT_ID` Operation Tracking** - Monitor expert selection during execution
3. **Environment Variables** - General debugging for scheduler and graph operations
4. **Logging Facilities** - `LOG_DBG()`, `LOG_INF()`, etc. from [`common/log.h`](../common/log.h)

For more details on the planned Expert Usage Tracer, see [`MoE_PostFetch_V0_0_4.md#task-0-expert-usage-tracer`](MoE_PostFetch_V0_0_4.md#task-0-expert-usage-tracer).

## 1. Graph Callback System (`llm_graph_cb`)

The primary debug mechanism is a callback system defined in [`src/llama-graph.h:487`](../src/llama-graph.h:487):

```cpp
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;
```

This callback is invoked throughout the graph building process via the `cb()` function in [`src/llama-graph.cpp:813-817`](../src/llama-graph.cpp:813-817):

```cpp
void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}
```

## 2. MoE-Specific Tensor Names for Debugging

The MoE pathway registers several key tensors with descriptive names that can be tracked via callback:

| Tensor Name | Description | Location |
|--------------|-------------|------------|
| `ffn_moe_logits` | Raw expert selection scores | [`src/llama-graph.cpp:1121`](../src/llama-graph.cpp:1121) |
| `ffn_moe_probs` | Expert probabilities after gating function | [`src/llama-graph.cpp:1148`](../src/llama-graph.cpp:1148) |
| `ffn_moe_probs_biased` | Biased probabilities for expert selection | [`src/llama-graph.cpp:1155`](../src/llama-graph.cpp:1155) |
| `ffn_moe_group_topk` | Top-k selected expert groups | [`src/llama-graph.cpp:1185`](../src/llama-graph.cpp:1185) |
| `ffn_moe_argsort` | Arguments before top-k selection | [`src/llama-graph.cpp:1196`](../src/llama-graph.cpp:1196) |
| `ffn_moe_topk` | **Top-k selected expert indices** | [`src/llama-graph.cpp:1197`](../src/llama-graph.cpp:1197) |
| `ffn_moe_weights` | Expert weights for selected experts | [`src/llama-graph.cpp:1209`](../src/llama-graph.cpp:1209) |
| `ffn_moe_weights_norm` | Normalized expert weights | [`src/llama-graph.cpp:1230`](../src/llama-graph.cpp:1230) |

The most important tensor for tracking which experts are used is **`ffn_moe_topk`**, which contains the indices of the selected experts.

## 3. Expert Selection Tracking via `GGML_OP_MUL_MAT_ID`

The imatrix tool demonstrates how to track expert usage by monitoring the `GGML_OP_MUL_MAT_ID` operation in [`tools/imatrix/imatrix.cpp:255-334`](../tools/imatrix/imatrix.cpp:255-334):

```cpp
if (t->op == GGML_OP_MUL_MAT_ID) {
    // ids  -> [n_experts_used, n_tokens]
    // src1 -> [cols, n_expert_used, n_tokens]
    const ggml_tensor * ids = t->src[2];
    
    // The top-k selected expert ids are stored in the ids tensor
    // for simplicity, always copy ids to host, because it is small
    // take into account that ids is not contiguous!
    
    m_ids.resize(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));
    
    // ... extract expert IDs and track usage
}
```

This operation contains the expert IDs selected for each token, allowing you to track which experts are being used.

## 4. Environment Variables for Debugging

| Environment Variable | Purpose | Location |
|---------------------|---------|------------|
| `LLAMA_GRAPH_RESULT_DEBUG` | Controls debug output for graph operations | [`src/llama-graph.cpp:650-651`](../src/llama-graph.cpp:650-651) |
| `LLAMA_GRAPH_INPUT_DEBUG` | Controls debug output for graph inputs | [`src/llama-graph.h:84-86`](../src/llama-graph.h:84-86) |
| `GGML_SCHED_DEBUG` | Controls scheduler debug output for graph execution | [`ggml/src/ggml-backend.cpp:1644-1645`](../ggml/src/ggml-backend.cpp:1644-1645) |
| `GGML_SCHED_DEBUG_REALLOC` | Controls debug output for graph reallocation tracking | [`ggml/src/ggml-backend.cpp:1651-1652`](../ggml/src/ggml-backend.cpp:1651-1652) |

### GGML_SCHED_DEBUG Details

When `GGML_SCHED_DEBUG` is set:
- **Level 0**: No debug output (default)
- **Level 1**: Prints graph split information and node assignments
- **Level 2+**: Additionally prints detailed source tensor information for each node

The debug output includes:
- Tensor names
- Operation types
- Tensor sizes
- Backend assignments (CPU/GPU)
- Use counts and compute flags

This is useful for understanding which tensors (including MoE expert tensors) are assigned to which backends.

## 5. Using the Debug Callback

To track expert usage, you can set up a debug callback similar to [`common/debug.cpp:115-161`](../common/debug.cpp:115-161):

```cpp
template <bool abort_on_nan> bool common_debug_cb_eval(
    struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (base_callback_data *) user_data;
    
    if (ask) {
        return true;  // Always retrieve data
    }
    
    // Filter for MoE-related tensors
    bool matches_filter = cb_data->tensor_filters.empty();
    if (!matches_filter) {
        for (const auto & filter : cb_data->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }
    
    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", 
            __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), 
            src1 ? src1_str : "", common_ggml_ne_string(t).c_str());
    }
    
    return true;
}
```

### Example: Filtering for MoE Tensors

To specifically track MoE expert tensors, you can filter by tensor name patterns:

```cpp
// Filter for MoE-related tensors
if (std::regex_search(t->name, "ffn_moe_.*")) {
    // Log tensor information including expert IDs
    LOG("%s: %s = %s\n", __func__, t->name, ggml_op_desc(t));
}
```

## 6. Key MoE Models Supported

The codebase supports multiple MoE architectures:

| Model | Source Files |
|--------|--------------|
| DeepSeek V2/V3 | [`src/models/deepseek.cpp`](../src/models/deepseek.cpp), [`src/models/deepseek2.cpp`](../src/models/deepseek2.cpp) |
| Qwen2-MoE | [`src/models/qwen2moe.cpp`](../src/models/qwen2moe.cpp) |
| GrovEMoE | [`src/models/grovemoe.cpp`](../src/models/grovemoe.cpp) |
| ExaOne-MoE | [`src/models/exaone-moe.cpp`](../src/models/exaone-moe.cpp) |
| BailingMoE | [`src/models/bailingmoe.cpp`](../src/models/bailingmoe.cpp) |
| GLM4-MoE | [`src/models/glm4-moe.cpp`](../src/models/glm4-moe.cpp) |
| Hunyuan-MoE | [`src/models/hunyuan-moe.cpp`](../src/models/hunyuan-moe.cpp) |
| LLaMA4-MoE | [`src/models/llama4.cpp`](../src/models/llama4.cpp) |
| And others | Various model files |

## 7. Summary: How to Track Which Experts Are Used

To track which experts are being used during inference:

1. **Set up a debug callback** via `params.cb_eval` (or `params.cb` for graph building)
2. **Filter for MoE tensors** with names matching the `ffn_moe_*` pattern
3. **Monitor the `ffn_moe_topk` tensor** which contains the selected expert indices
4. **Track `GGML_OP_MUL_MAT_ID` operations** - the `ids` tensor contains expert IDs for each token
5. **Use environment variables** for additional debug output:
   - `GGML_SCHED_DEBUG=1` or `GGML_SCHED_DEBUG=2` for scheduler details
   - `LLAMA_GRAPH_RESULT_DEBUG=1` for graph operation details

The callback system provides comprehensive visibility into the expert selection process, including:
- Raw logits for expert selection
- Expert probabilities (before and after bias)
- Selected expert indices (top-k)
- Expert weights (before and after normalization)

## 8. Callback Points Before and After Expert Selection

The MoE pathway provides **23 callback points** organized chronologically:

### Before Expert Selection (7 points)

| # | Tensor Name | Line | Description |
|---|--------------|------|-------------|
| 1 | `ffn_moe_logits` | 1121 | Raw expert selection scores from gate input |
| 2 | `ffn_moe_logits_biased` | 1128 | Expert scores after adding bias (if `gate_inp_b` exists) |
| 3 | `ffn_moe_probs` | 1148 | Expert probabilities after gating function (softmax/sigmoid) |
| 4 | `ffn_moe_probs_biased` | 1155 | Biased probabilities for expert selection (if `exp_probs_b` exists) |
| 5 | `ffn_moe_probs_biased` | 1166 | Biased probabilities for GROVEMOE architecture |
| 6 | `ffn_moe_probs_masked` | 1191 | Probabilities after masking non-selected expert groups |
| 7 | `ffn_moe_argsort` | 1196 | Arguments before top-k selection (input to argsort) |

### After Expert Selection (16 points)

| # | Tensor Name | Line | Description |
|---|--------------|------|-------------|
| 8 | **`ffn_moe_topk`** | 1197 | **Selected expert indices** - This is the KEY tensor for tracking which experts are used! |
| 9 | `ffn_moe_weights` | 1209 | Expert weights for selected experts |
| 10 | `ffn_moe_weights_softmax` | 1216 | Weights after softmax (if `SOFTMAX_WEIGHT` gating) |
| 11 | `ffn_moe_weights_sum` | 1223 | Sum of weights before normalization |
| 12 | `ffn_moe_weights_sum_clamped` | 1227 | Sum of weights after clamping to avoid division by zero |
| 13 | `ffn_moe_weights_norm` | 1230 | Normalized expert weights |
| 14 | `ffn_moe_weights_scaled` | 1236 | Weights after scaling (if `scale_w` is true) |
| 15 | `ffn_moe_weighted` | 1248 | Input tensor weighted by expert weights (for LLaMA4) |
| 16 | `ffn_moe_up` | 1252 | Expert up-projection output |
| 17 | `ffn_moe_up_biased` | 1256 | Expert up-projection with bias (if `up_exps_b` exists) |
| 18 | `ffn_moe_gate` | 1262 | Expert gate projection (if `gate_exps` exists) |
| 19 | `ffn_moe_gate_biased` | 1269 | Expert gate projection with bias (if `gate_exps_b` exists) |
| 20 | `ffn_moe_swiglu` | 1276 | Output after SwiGLU activation (if gated) |
| 21 | `ffn_moe_silu` | 1279 | Output after SiLU activation (if not gated) |
| 22 | `ffn_moe_geglu` | 1284 | Output after GeLU activation |
| 23 | `ffn_moe_down` | 1319 | Expert down-projection output |
| 24 | `ffn_moe_down_biased` | 1323 | Expert down-projection with bias (if `down_exps_b` exists) |

### Key Insight

The most important callback point for tracking which experts are used is **`ffn_moe_topk`** at line 1197. This tensor contains the actual indices of the selected experts for each token.

The sequence shows:
- **7 callback points** before expert selection (logits, probabilities, masking)
- **1 critical callback point** at expert selection (`ffn_moe_topk`) - **this is where you can see which experts are chosen**
- **16 callback points** after expert selection (weight processing, activations, projections)

This comprehensive callback system allows you to track the entire MoE pathway from initial gating through final expert outputs.
