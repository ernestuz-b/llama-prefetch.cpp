# Post-Fetch MoE Execution

**Version:** 0.0.3 (Implementation-Guided)
**Status:** Design Document
**Target:** llama.cpp MoE Optimization for Low-VRAM Consumer GPUs

---

## Document Revision History

### Version 0.0.3 (2026-02-07) - Implementation Guidance

**Major additions for implementation guidance:**
- **ADDED:** Critical Implementation Pitfalls section (expanded from v0.0.2)
- **ADDED:** CUDA Context Access Strategy (3 approaches with pros/cons)
- **ADDED:** LoRA Compatibility Matrix (hook-level behavior)
- **ADDED:** Stream Coordination Best Practices
- **ADDED:** Configuration Validation Checklist
- **ADDED:** Debugging Environment Variables
- **ADDED:** Testing Strategy (unit, integration, performance)
- **ADDED:** Implementation Roadmap (3 phases)
- **ADDED:** Common Error Messages and Solutions
- **ADDED:** Performance Monitoring section

**Addresses implementation concerns:**
1. ✅ Model access strategy documented with 3 viable approaches
2. ✅ LoRA compatibility clarified for both hook levels
3. ✅ Stream synchronization best practices documented
4. ✅ Configuration validation checklist added
5. ✅ Testing strategy with concrete checklists
6. ✅ Implementation roadmap with phased approach
7. ✅ Common error messages and solutions reference
8. ✅ Performance monitoring guidance

### Version 0.0.2 (2025-02-07) - Review Corrections

**Incorporates code review feedback:**
- **REMOVED:** Duplicate buggy implementation snippet (lines 1131-1164 from v0.1.0)
- **ADDED:** Model access strategies section (thread-local, extended context, tensor metadata)
- **ADDED:** LoRA adapter considerations and compatibility analysis
- **ADDED:** Stream coordination strategy with ggml's existing streams
- **RESTRUCTURED:** Presented high-level hook as RECOMMENDED approach, backend-level as advanced
- **CLARIFIED:** Implementation paths with clear trade-offs and recommendations

**Addresses reviewer concerns:**
1. ✅ Model context access mechanism explicitly defined
2. ✅ LoRA compatibility explicitly addressed at both hook levels
3. ✅ Stream management coordination with ggml clarified
4. ✅ Removed editing error (duplicate code with old bug)
5. ✅ Phased implementation approach (high-level → backend-level)

### Version 0.0.1b (2025-02-07) - Implementation Fixes

**Critical Implementation Fixes:**
- Fixed memory layout bug: Changed from incorrect `i * size` to cumulative offset calculation
- Fixed stream synchronization: Changed from `NULL` stream to proper `compute_stream` handling
- Added missing CPU fallback logic with `cudaEventQuery()` checks
- Updated hook point to reference actual llama.cpp code (`ggml_cuda_mul_mat_id`)
- Corrected expert tensor access to use actual model structure
- Added GPU scratchpad lifecycle management
- Introduced separate `fetch_stream` and `compute_stream` for proper overlap
- Added comprehensive debugging and validation sections

### Version 0.0.1 (Original)
- Initial conceptual design
- Basic architecture and principles defined

---

## Model Architecture Comparison

The Post-Fetch mechanism is designed for modern MoE models with varying expert configurations:

| Feature | GPT-OSS 120B | GPT-OSS 20B | Qwen3-Next (80B) | Qwen3-Coder-Next |
|---------|--------------|-------------|------------------|------------------|
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

---

## Table of Contents

1. [Overview](#overview)
2. [Key Principles](#key-principles)
3. [Target Use Case](#target-use-case)
4. [Technical Background](#technical-background)
5. [How Post-Fetch Works](#how-post-fetch-works)
6. [Architecture](#architecture)
7. [Implementation Details](#implementation-details)
8. [Configuration](#configuration)
9. [Performance Characteristics](#performance-characteristics)
10. [Future Extensions](#future-extensions)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

**Post-Fetch** is a minimal, targeted optimization for Mixture-of-Experts (MoE) inference in llama.cpp. It addresses a specific performance bottleneck: the latency of transferring expert weights from CPU to GPU over PCIe.

Unlike traditional caching or preloading strategies, Post-Fetch:

- **Does not cache expert weights** across tokens
- **Does not retain tensors** in VRAM
- **Does not change model semantics**
- **Does not require kernel rewrites** or partial tensor handling

Instead, it exploits a narrow but reliable overlap window that exists **after routing completes but before expert computation begins**. This allows PCIe transfers to be partially hidden under unavoidable CPU computation.

### Why "Post-Fetch"?

The name is intentionally counter-intuitive: weights are fetched *after* the model knows they are needed (post-routing), but *before* they are consumed (pre-computation). This timing is critical to the mechanism's correctness and safety.

---

## Key Principles

Post-Fetch is built on four foundational principles:

### 1. Statelessness

- No eviction policy required
- No VRAM accounting or tracking
- No reuse assumptions between tokens
- No long-term memory pressure

### 2. Tensor Atomicity

- Tensors are either fully on CPU or fully on GPU
- No partial availability or streaming
- Standard CUDA `memcpy` semantics
- No tiled or streamed kernels

### 3. CPU-First Semantics

- All expert matmuls may still run on CPU
- GPU serves as a temporary weight scratchpad
- Graceful fallback if GPU transfer is not ready
- Both blocking and non-blocking policies supported

### 4. Correctness by Construction

- No speculative fetches
- No use of incomplete tensors
- No semantic changes to the model
- Fully deterministic given routing decisions

---

## Target Use Case

Post-Fetch is explicitly designed for:

| Characteristic | Specification |
|----------------|---------------|
| **Hardware** | Consumer GPUs (4–8 GB VRAM) |
| **Bus** | PCIe-bound systems |
| **Batch Size** | 1 (interactive inference) |
| **Execution** | CPU experts (`-cmoe`-like behavior) |
| **Target Users** | Users who cannot afford persistent VRAM residency |

### Not Intended For

- Datacenter GPUs with abundant VRAM
- Large-batch throughput optimization
- Maximum GPU utilization scenarios
- Replacing expert caching strategies

---

## Technical Background

### MoE Expert Structure

Most llama.cpp MoE models (e.g., Mixtral-style) use an expert MLP with this structure:

```
x ──→ G (gate projection) ──┐
⊙ ──→ D (down projection)   ← Post-Fetch target
x ──→ U (up projection) ────┘
```

**Key properties:**

- Routing happens before any expert computation
- G and U are independent and computed in parallel
- D depends on both G and U
- All expert tensors (G, U, D) are known immediately after routing

### The Critical Path

```
┌─────────────────────────────────────────────────────────────┐
│ MoE Layer Execution Timeline                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Routing: Select experts E₁, E₂, ..., Eₙ                 │
│ 2. Compute G_E and U_E on CPU                               │
│ 3. Compute activation & multiply                            │
│ 4. Transfer D_E to GPU (if not cached)                      │
│ 5. Compute D_E on GPU/CPU                                   │
└─────────────────────────────────────────────────────────────┘
```

The bottleneck is step 4: transferring the down-projection tensor (D) to GPU. This is often:
- The largest tensor (down-projection is typically the biggest)
- Memory-bandwidth bound
- Occurs at the end of the critical path

### Why Only D?

| Tensor | Transfer Timing | Reason |
|--------|----------------|--------|
| **G** | Early | Consumed immediately; no time to hide latency |
| **U** | Early | Consumed immediately; no time to hide latency |
| **D** | Late | Last on critical path; CPU G/U computation provides overlap window |

---

## How Post-Fetch Works

### The Overlap Window

After routing selects expert `E`, the model knows `G_E`, `U_E`, and `D_E` will all be required. However:

1. **G and U** are computed on CPU first
2. **D** is computed last, after G and U are combined

This creates a natural overlap window:

```
┌──────────────────────────────────────────────────────────────────┐
│ Post-Fetch Execution Timeline                                    │
├──────────────────────────────────────────────────────────────────┤
│ routing                                                          │
│   │                                                              │
│   ├─► async memcpy(D_E → GPU)  ← starts immediately              │
│   │                                                              │
│   ├─► CPU: G_E + U_E           ← overlaps with transfer         │
│   │                                                              │
│   ├─► CPU: activation & multiply                                 │
│   │                                                              │
│   ├─► (wait if needed)                                           │
│   │                                                              │
│   └─► CPU or GPU: D_E                                            │
└──────────────────────────────────────────────────────────────────┘
```

### Transfer Strategy

1. **Trigger Point**: Immediately after MoE routing selects experts
2. **What to Transfer**: Only the down-projection tensor (D)
3. **Transfer Method**: `cudaMemcpyAsync` on a private CUDA stream
4. **Readiness Tracking**: CUDA events
5. **Fallback Policy**: CPU execution if transfer not ready

### Safety Guarantees

- **No incorrect execution**: D is only used after transfer completes
- **No partial tensors**: Full tensor transfer with standard semantics
- **No semantic changes**: Model produces identical outputs
- **No scheduling dependencies**: Behavior is deterministic

---

## Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    llama.cpp / ggml                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - Model loading                                          │  │
│  │ - Graph construction                                     │  │
│  │ - Routing logic                                          │  │
│  │ - CPU matmuls                                            │  │
│  │ - Execution order                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ Hook: After routing, before compute
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Post-Fetch Sidecar (Observer)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - Transfer scheduling                                    │  │
│  │ - CUDA stream management                                 │  │
│  │ - Readiness state tracking                               │  │
│  │ - Configuration loading                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Sidecar Responsibilities

The Post-Fetch sidecar owns exactly four responsibilities:

1. **Configuration Loading** (at startup)
2. **Transfer Scheduling** (after routing)
3. **State Tracking** (minimal state machine)
4. **Readiness Query** (before D execution)

### Sidecar Non-Responsibilities

The sidecar explicitly does **not**:

- ❌ Modify ggml matmul kernels
- ❌ Implement caching or eviction
- ❌ Split tensors or use partial availability
- ❌ Rewrite the execution graph
- ❌ Change model semantics

---

## Implementation Details

### Hook Point

The only required hook in llama.cpp:

> **Immediately after MoE routing selects experts, before any expert computation begins.**

At this point:
- `expert_id` is known
- Pointer to `D_E` tensor is known
- No expert matmul has started
- No ordering assumptions are violated

### State Machine

Each `(expert_id, tensor=D)` is tracked independently:

```
CPU_ONLY
   │
   ├─► (async memcpy scheduled)
   ▼
COPY_IN_FLIGHT
   │
   ├─► (CUDA event complete)
   ▼
GPU_READY
```

- State is discarded after tensor use
- No eviction states
- No reuse states

### Readiness Query Logic

Before `D_E` is consumed:

```
if GPU_READY:
    use GPU tensor
elif COPY_IN_FLIGHT:
    if block_on_miss:
        wait until ready
    else:
        fall back to CPU
else:  # CPU_ONLY
    CPU execution
```

The sidecar **never forces GPU usage**.

### CUDA Usage Model

The sidecar:

- Owns one or more dedicated CUDA streams
- Uses `cudaMemcpyAsync` for transfers
- Uses `cudaEventRecord` / `cudaEventQuery` for synchronization
- Does **not** inject work into llama.cpp's CUDA stream
- Does **not** assume stream ordering with ggml kernels
- Does **not** rely on unified memory

This isolation prevents interference with existing CUDA logic.

### Failure & Fallback Semantics

If anything goes wrong (CUDA unavailable, transfer fails, event not ready, VRAM allocation fails), execution silently falls back to CPU behavior:

- No crashes
- No correctness changes
- At worst, no speedup

---

## CUDA Stream Management Strategy

Post-Fetch uses multiple CUDA streams to maximize overlap between transfers and computation:

### Stream Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ CUDA Stream Organization                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  fetch_stream    ──→  D-weight transfers (H2D memcpy)       │
│                      └─→ Events: d_ready_event[i]           │
│                                                              │
│  compute_stream  ──→  GPU kernel execution (matmuls)        │
│                      └─→ Waits on: d_ready_event[i]         │
│                                                              │
│  (default stream) ──→ Rest of ggml graph execution          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Why Separate Streams?

**Problem**: Using the same stream for transfers and compute would serialize operations:
```
Stream 0: [Transfer D0] → [Transfer D1] → ... → [Compute G/U] → [Compute D]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          These block compute from starting early
```

**Solution**: Separate streams allow overlap:
```
fetch_stream:   [Transfer D0] [Transfer D1] ... [Transfer D10]
                      ↓             ↓                ↓
compute_stream:       [........Compute G/U........] [Compute D]
                      ↑
                      Starts immediately, doesn't wait for all transfers
```

### Synchronization Points

```cpp
// PHASE 1: Launch all D-weight transfers (no blocking)
for (int i = 0; i < n_experts; ++i) {
    cudaMemcpyAsync(dst, src, size, H2D, fetch_stream);
    cudaEventRecord(d_ready_event[i], fetch_stream);
}

// PHASE 2: Compute G/U (overlaps with transfers)
// This uses compute_stream or CPU threads
compute_gu(..., compute_stream);

// PHASE 3: Execute D-projection (wait for specific D-weight)
for (int i = 0; i < n_experts; ++i) {
    // Make compute_stream wait for this specific transfer
    cudaStreamWaitEvent(compute_stream, d_ready_event[i], 0);
    
    // Now safe to use D-weight in computation
    execute_d_matmul(..., compute_stream);
}
```

### Event-Based Synchronization

**Why not `cudaStreamSynchronize()`?**
- `cudaStreamSynchronize(fetch_stream)` would wait for ALL transfers
- We only need to wait for the specific D-weight we're about to use
- Events provide fine-grained, per-transfer synchronization

**Event workflow:**
1. `cudaEventRecord(event[i], fetch_stream)` - mark when transfer i completes
2. `cudaStreamWaitEvent(compute_stream, event[i], 0)` - make compute_stream wait for transfer i
3. `cudaEventQuery(event[i])` - check if transfer completed (for fallback logic)

---

## Configuration

Post-Fetch uses environment variables prefixed with `LLAMA_POSTFETCH_` for configuration. This approach:

- Requires no code changes for configuration
- Keeps behavior external to the core
- Enables runtime tuning without recompilation
- Simplifies experimentation

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_POSTFETCH_ENABLE` | `1` | Enable/disable Post-Fetch (0 = disabled, 1 = enabled) |
| `LLAMA_POSTFETCH_FORCE_CPU` | `0` | Force CPU execution (0 = GPU allowed, 1 = CPU only) |
| `LLAMA_POSTFETCH_BLOCK_ON_MISS` | `1` | Block on D readiness (0 = fallback to CPU, 1 = wait) |
| `LLAMA_POSTFETCH_MAX_TRANSFERS` | `8` | Maximum concurrent transfers |
| `LLAMA_POSTFETCH_SCRATCHPAD_MB` | `0` | Scratchpad size in MB (0 = auto-calculate) |
| `LLAMA_POSTFETCH_USE_DEDICATED_STREAMS` | `1` | Use separate fetch/compute streams (0 = single stream, 1 = separate) |

### Usage Examples

```bash
# Enable Post-Fetch (default)
export LLAMA_POSTFETCH_ENABLE=1

# Disable Post-Fetch and fall back to pure CPU execution
export LLAMA_POSTFETCH_ENABLE=0

# Force CPU execution even when GPU is available
export LLAMA_POSTFETCH_FORCE_CPU=1

# Allow fallback to CPU on transfer stalls (don't block)
export LLAMA_POSTFETCH_BLOCK_ON_MISS=0

# Limit concurrent transfers for memory-constrained systems
export LLAMA_POSTFETCH_MAX_TRANSFERS=4

# Manually set scratchpad size to 512MB (useful for limiting VRAM usage)
export LLAMA_POSTFETCH_SCRATCHPAD_MB=512

# Use single stream mode (simpler but less overlap)
export LLAMA_POSTFETCH_USE_DEDICATED_STREAMS=0

# Recommended settings for 4GB GPU with Qwen3-Next
export LLAMA_POSTFETCH_ENABLE=1
export LLAMA_POSTFETCH_SCRATCHPAD_MB=700
export LLAMA_POSTFETCH_BLOCK_ON_MISS=1
```

### Configuration Priority

Environment variables are checked at runtime in this order:

1. `LLAMA_POSTFETCH_ENABLE` - Global on/off switch
2. `LLAMA_POSTFETCH_FORCE_CPU` - Override to CPU-only mode
3. `LLAMA_POSTFETCH_BLOCK_ON_MISS` - Behavior on readiness check
4. `LLAMA_POSTFETCH_MAX_TRANSFERS` - Transfer capacity limit

---

## Performance Characteristics

### Expected Gains

- Reduced tail latency on MoE layers
- Best improvement on:
  - 4 GB GPUs
  - PCIe-limited systems
  - Quantized expert weights

### Expected Limits

- Does not improve throughput
- Does not eliminate transfers
- Does not help large-VRAM systems
- Does not replace caching

### Quantitative Estimate

For Qwen3-Next (80B) with 10 active experts:

| Component | Size Estimate |
|-----------|---------------|
| Per expert D-tensor (Q4_K_M) | ~50 MB |
| Total for 10 experts | ~500 MB |
| VRAM scratchpad needed | ~500–700 MB |

This fits comfortably within a 4 GB GPU's available VRAM.

---

## Future Extensions

Post-Fetch is intentionally designed as a foundation, not an endpoint. Possible future extensions:

### 1. Combine with Expert Caching

- Post-Fetch becomes the **miss path**
- Cached tensors skip sidecar logic
- Sidecar state machine remains unchanged

### 2. Extend to Additional Tensors

- When VRAM allows, prefetch G and U as well
- Use heuristics to decide which tensors to prefetch
- Balance memory usage against latency reduction

### 3. Dynamic Policy Optimization

- Learn block vs. fallback decisions
- Adapt to workload patterns
- Optimize transfer batching

### 4. Promote Hot Experts

- Track expert usage frequency
- Promote frequently-used experts to persistent residency
- Hybrid approach: cache hot experts, use Post-Fetch for cold

---

## Implementation Roadmap

### Phase 1: Core Implementation

**Goal**: Minimal working implementation with safety guarantees

1. **Hook Integration**
   - Locate MoE routing completion point
   - Insert Post-Fetch trigger
   - Verify correctness with unit tests

2. **Sidecar Skeleton**
   - Configuration loading
   - CUDA stream management
   - State tracking infrastructure

3. **Transfer Logic**
   - Implement async memcpy
   - Add event recording
   - Implement readiness query

4. **Fallback Handling**
   - CPU execution path
   - Error handling
   - Graceful degradation

### Phase 2: Optimization

**Goal**: Performance tuning and robustness

1. **Stream Management**
   - Multiple transfer streams
   - Overlap optimization
   - Priority handling

2. **Batching**
   - Batch transfers for multiple experts
   - Optimize for PCIe transfer characteristics

3. **Memory Management**
   - Pre-allocated scratchpad
   - Efficient VRAM utilization
   - Fragmentation avoidance

### Phase 3: Integration Testing

**Goal**: Validate across target models

1. **Model Coverage**
   - GPT-OSS 120B
   - GPT-OSS 20B
   - Qwen3-Next 80B
   - Qwen3-Coder-Next 80B

2. **Hardware Coverage**
   - 4 GB GPUs
   - 6–8 GB GPUs
   - Different PCIe generations

3. **Benchmarking**
   - Latency measurements
   - Throughput impact
   - Memory usage profiling

### Phase 4: Documentation

**Goal**: Enable community adoption

1. **User Guide**
   - Configuration examples
   - Performance expectations
   - Troubleshooting

2. **Developer Guide**
   - Architecture overview
   - Extension points
   - Contribution guidelines

3. **Performance Reports**
   - Model-specific results
   - Hardware comparisons
   - Optimization opportunities

---

## Implementation Sketch (Conceptual)

### Configuration via Environment Variables

All Post-Fetch behavior is controlled via environment variables prefixed with `LLAMA_POSTFETCH_`:

```cpp
// Example configuration check at startup
const char* enable_str = getenv("LLAMA_POSTFETCH_ENABLE");
int enable_postfetch = (enable_str == NULL || atoi(enable_str) != 0);

const char* force_cpu_str = getenv("LLAMA_POSTFETCH_FORCE_CPU");
int force_cpu = (force_cpu_str != NULL && atoi(force_cpu_str) != 0);

const char* block_on_miss_str = getenv("LLAMA_POSTFETCH_BLOCK_ON_MISS");
int block_on_miss = (block_on_miss_str == NULL || atoi(block_on_miss_str) != 0);
```

This approach:
- Requires no code changes for configuration
- Keeps Post-Fetch behavior external to the core
- Enables runtime tuning without recompilation

---

### Trigger Point

The only required hook is:

> **Immediately after MoE routing selects experts**

At this point:
- `expert_id` is known
- Pointer to `D_E` tensor is known
- No expert computation has started

---

### Transfer Logic

For each selected expert `E`:

1. Check `LLAMA_POSTFETCH_ENABLE` - skip if disabled
2. Check `LLAMA_POSTFETCH_FORCE_CPU` - fallback if set
3. Launch `cudaMemcpyAsync(D_E_cpu → D_E_gpu)` on a private stream
4. Record a CUDA event
5. Proceed with normal CPU execution

Before executing D:
- If event is complete → use GPU tensor
- Else if `LLAMA_POSTFETCH_BLOCK_ON_MISS=1` → wait
- Else → CPU fallback

---

## Sidecar Architecture & Integration Notes

### Design Goal

The Post-Fetch mechanism is intentionally implemented as a **sidecar**, not as a core rewrite.

The sidecar must:
- Avoid modifying ggml matmul kernels
- Avoid graph rewrites or tensor splitting
- Avoid persistent caching or eviction logic
- Integrate at a **single, well-defined semantic boundary**
- Fail safely (no correctness regression)

This document describes **how Post-Fetch is wired**, not why it exists.

---

### Architectural Principle

> **llama.cpp remains the executor. The sidecar remains the scheduler.**

The sidecar:
- Observes *what will happen next*
- Initiates asynchronous transfers
- Tracks readiness
- Never changes model semantics

---

### High-Level Structure

```
┌────────────────────┐
│ llama.cpp / ggml   │
│                    │
│  - routing         │
│  - CPU matmuls     │
│  - execution order │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Post-Fetch Sidecar │
│                    │
│  - transfer logic  │
│  - CUDA streams    │
│  - readiness state │
└────────────────────┘
```

The sidecar is **event-driven**, not execution-driven.

---

### Required Hook (Single Semantic Hook)

#### Hook Location (Conceptual)

> **Immediately after MoE routing selects experts, before any expert computation begins.**

At this moment:
- Expert IDs are known
- All expert tensors (G, U, D) are known
- No expert matmul has started
- No ordering assumptions are violated

This is the **earliest and safest hook**.

---

#### What the Hook Provides

The hook must expose (directly or indirectly):
- `expert_id`
- Pointer / handle to expert tensors:
  - `D_E` (down-projection tensor)
- Execution context (CPU vs GPU allowed)

No changes to routing logic are required.

---

### Sidecar Responsibilities

The sidecar owns **exactly four responsibilities**.

---

#### 0. Configuration Loading (at startup)

Before any expert routing occurs, the sidecar loads configuration from environment variables:

```cpp
// Load configuration once at startup
const char* enable_str = getenv("LLAMA_POSTFETCH_ENABLE");
int enable_postfetch = (enable_str == NULL || atoi(enable_str) != 0);

const char* force_cpu_str = getenv("LLAMA_POSTFETCH_FORCE_CPU");
int force_cpu = (force_cpu_str != NULL && atoi(force_cpu_str) != 0);

const char* block_on_miss_str = getenv("LLAMA_POSTFETCH_BLOCK_ON_MISS");
int block_on_miss = (block_on_miss_str == NULL || atoi(block_on_miss_str) != 0);

const char* max_transfers_str = getenv("LLAMA_POSTFETCH_MAX_TRANSFERS");
int max_transfers = (max_transfers_str == NULL) ? 8 : atoi(max_transfers_str);
```

This ensures:
- No code changes needed for configuration
- Runtime behavior can be tuned without recompilation
- Default values provide sensible behavior

---

#### 1. Transfer Scheduling

For each selected expert `E`:
- Check `LLAMA_POSTFETCH_ENABLE` - skip if disabled
- Check `LLAMA_POSTFETCH_FORCE_CPU` - fallback if set
- Schedule an **asynchronous CPU → GPU transfer** of `D_E`
- Use a **private CUDA stream**
- Record a **CUDA event** on completion

Important:
- Transfers are issued *after routing*
- Transfers are issued *before G/U compute*
- Transfers are never speculative
- Limited by `LLAMA_POSTFETCH_MAX_TRANSFERS`

---

#### 2. State Tracking (Minimal State Machine)

Each `(expert_id, tensor=D)` is tracked independently.

```
CPU_ONLY
   │
   ├─► (async memcpy scheduled)
   ▼
COPY_IN FLIGHT
   │
   ├─► (CUDA event complete)
   ▼
GPU_READY
```

No eviction states. No reuse states. State is discarded after tensor use.

---

#### 3. Readiness Query

Before `D_E` is consumed:
- Sidecar checks CUDA event status

Possible outcomes:
- **GPU_READY** → use GPU tensor
- **COPY_IN FLIGHT** → policy decision:
  - block until ready, or
  - fall back to CPU
- **CPU_ONLY** → CPU execution

The sidecar **never forces GPU usage**.

---

#### 4. Cleanup

After `D_E` is consumed:
- GPU buffer may be:
  - immediately freed, or
  - returned to a scratch allocator
- State entry is destroyed

No persistent residency.

---

### What the Sidecar Does *Not* Do

Explicit non-goals:
- ❌ No caching
- ❌ No eviction policy
- ❌ No tensor reuse across tokens
- ❌ No modification of ggml kernels
- ❌ No partial tensor usage
- ❌ No graph reordering

This keeps failure modes narrow and understandable.

---

### Execution Timeline (Sidecar View)

For one expert `E`:
```
routing(E)
  │
  ├─► sidecar: schedule memcpy(D_E)
  │
  CPU: G_E + U_E
  CPU: activation & multiply
  │
  ├─► sidecar: check D_E readiness
  │
  CPU or GPU: D_E
  cleanup
```

The sidecar **never blocks earlier than D**.

---

### CUDA Usage Model

The sidecar:
- Owns one or more CUDA streams
- Uses `cudaMemcpyAsync`
- Uses `cudaEventRecord` / `cudaEventQuery`

It does **not**:
- Inject work into llama.cpp's CUDA stream
- Assume stream ordering with ggml kernels
- Rely on unified memory

This isolation prevents interference with existing CUDA logic.

---

### Failure & Fallback Semantics

Post-Fetch must be **fail-safe**.

If anything goes wrong:
- CUDA unavailable
- Transfer fails
- Event not ready
- VRAM allocation fails

Then:
> **Execution silently falls back to CPU behavior.**

No crashes. No correctness changes. At worst, no speedup.

---

### Environment Variable Configuration

Post-Fetch uses environment variables prefixed with `LLAMA_POSTFETCH_` for configuration. This approach minimizes code changes and keeps configuration external to the implementation.

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_POSTFETCH_ENABLE` | `1` | Enable/disable Post-Fetch (0 = disabled, 1 = enabled) |
| `LLAMA_POSTFETCH_BLOCK_ON_MISS` | `1` | Block on D readiness (0 = fallback to CPU, 1 = wait) |
| `LLAMA_POSTFETCH_MAX_TRANSFERS` | `8` | Maximum concurrent transfers |
| `LLAMA_POSTFETCH_FORCE_CPU` | `0` | Force CPU execution (0 = GPU allowed, 1 = CPU only) |

#### Usage Examples

```bash
# Enable Post-Fetch (default)
export LLAMA_POSTFETCH_ENABLE=1

# Disable Post-Fetch and fall back to pure CPU execution
export LLAMA_POSTFETCH_ENABLE=0

# Force CPU execution even when GPU is available
export LLAMA_POSTFETCH_FORCE_CPU=1

# Allow fallback to CPU on transfer stalls
export LLAMA_POSTFETCH_BLOCK_ON_MISS=0

# Limit concurrent transfers for memory-constrained systems
export LLAMA_POSTFETCH_MAX_TRANSFERS=4
```

#### Configuration Priority

Environment variables are checked at runtime in this order:
1. `LLAMA_POSTFETCH_ENABLE` - Global on/off switch
2. `LLAMA_POSTFETCH_FORCE_CPU` - Override to CPU-only mode
3. `LLAMA_POSTFETCH_BLOCK_ON_MISS` - Behavior on readiness check
4. `LLAMA_POSTFETCH_MAX_TRANSFERS` - Transfer capacity limit

---

### Why This is a True Sidecar

This design:
- Touches **no math**
- Touches **no kernel code**
- Touches **no graph structure**
- Requires **one semantic hook**
- Can be compiled out cleanly

That makes it suitable for:
- Experimental flags
- Low-risk PRs
- Incremental iteration

---

### Interaction with Future Caching (Non-Conflicting)

If expert caching is added later:
- Post-Fetch becomes the **miss path**
- Cached tensors skip sidecar logic
- Sidecar state machine remains unchanged

No redesign needed.

---

## Technical Implementation Mapping

### Hook Point (ggml-cuda.cu or MoE Kernel)

In llama.cpp, routing for Mixtral/Qwen MoE usually happens within the `llm_build_moe` graph construction. However, since you want to avoid graph changes, you should hook into the `ggml_compute_forward_sparse_moe` (or the specific MoE dispatcher).

**Logic**: At the moment the logits for the gate are computed (usually a small vector on CPU), the expert_ids are determined. The Change: Insert your `cudaMemcpyAsync` here. At this point, the CPU has the indices but hasn't started the up/gate projections.

---

### Identifying the "D" Tensors

For the models you listed, the "D" (Down-projection) tensors follow a predictable naming convention in the GGUF file. You can identify them during model loading and flag them:

- **GPT-OSS**: `blk.N.ffn_down.M.weight`
- **Qwen3-Next**: `blk.N.ffn_down.M.weight` (where N is layer, M is expert)

---

### VRAM "Scratchpad" Allocation

Since you have ~4GB VRAM, you don't need a cache. You only need a Static Buffer large enough to hold the active experts for one layer.

- **GPT-OSS 120B**: Active Experts = 4. If each "D" tensor (quantized) is ~50MB, you only need a fixed 200MB buffer on the GPU to act as the "Post-Fetch Destination."
- **Qwen3-Next**: Active Experts = 10. You need a buffer for 10 "D" tensors.

---

### Minimal Implementation Path

Instead of modifying the whole ggml backend, focus on `llama.cpp/src/llama.cpp`:

1. **Intercept the Gate Result**: Find where `dst->op == GGML_OP_NONE` (or the specific MoE op) is evaluated.
2. **Async Launch**: Use a dedicated CUDA stream (`cudaStream_t post_fetch_stream`) to avoid blocking the main compute stream.
3. **The "D" Wait**: Modify the D projection kernel call to `cudaStreamWaitEvent`. This tells the GPU: "Don't run this Down-projection until the post_fetch_stream says the weights are there."

---

### Why this works for Qwen3-Next

Qwen3-Next uses 512 experts. In a standard `-cmoe` (CPU MoE) setup, the PCIe bus is hit three times per expert (Gate, Up, Down).

**Your Post-Fetch**: The GPU starts pulling the Down-projections (the largest weights) while the CPU is still grinding through the 10 Gate/Up pairs. For Qwen3, this overlap is significant because the Gated DeltaNet attention provides a larger CPU-side compute window than standard Attention.

---

### Recommended Implementation Approach

To implement this "Hybrid-Parallel MoE" strategy, you need to split the expert execution into three concurrent pipelines:

- **CPU Core**: Computes `G_i/U_i` for the first batch of experts
- **GPU Async Stream**: Prefetches all `D_0...9` weights to a reserved scratchpad
- **GPU Compute Stream**: Computes `G_j/U_j` for the remaining experts, then executes all `D_0...9` once the weights and CPU results arrive

---

### C++ Implementation Sketch

#### 1. Header & Configuration Setup

Add this to the top of your MoE implementation file (e.g., within `ggml-cuda.cu` or the backend dispatcher).

```cpp
#include <cuda_runtime.h>
#include <cstdlib>

struct PostFetchConfig {
    bool enabled;
    int cpu_threads;
    cudaStream_t fetch_stream;
    cudaEvent_t d_ready_event[10]; // Max active experts for Qwen3

    PostFetchConfig() {
        const char* en = getenv("LLAMA_POSTFETCH_ENABLE");
        enabled = (en == nullptr || atoi(en) != 0);
        
        // Match this to your physical core count
        const char* thr = getenv("LLAMA_POSTFETCH_CPU_LIMIT");
        cpu_threads = thr ? atoi(thr) : 6;

        if (enabled) {
            cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
            for(int i=0; i<10; i++) cudaEventCreate(&d_ready_event[i]);
        }
    }
};

static PostFetchConfig PF_CONFIG;
```

#### 2. The Split-Execution Dispatcher

This function replaces the standard serial loop. It triggers the D-Prefetch immediately, then splits `G/U` work.

```cpp
void dispatch_postfetch_moe(
    const int* expert_ids,     // Selected by router
    const int num_active,      // 4 for GPT-OSS, 11 for Qwen3
    struct ggml_tensor** d_weights_cpu,  // Source tensors (array of tensor pointers)
    void* d_weights_gpu_scratch,         // Destination (VRAM Scratchpad)
    size_t* gpu_offsets                   // Pre-computed offsets for each expert
) {
    if (!PF_CONFIG.enabled) {
        // Fallback to standard llama.cpp CPU/GPU path
        return;
    }

    // PHASE 1: Immediate Async Prefetch of ALL D-tensors
    // Track cumulative offsets for proper memory layout
    size_t cumulative_offset = 0;
    for (int i = 0; i < num_active; ++i) {
        size_t d_size = ggml_nbytes(d_weights_cpu[i]);
        
        // Store offset for later use
        gpu_offsets[i] = cumulative_offset;
        
        cudaMemcpyAsync(
            (char*)d_weights_gpu_scratch + cumulative_offset,  // Correct offset
            d_weights_cpu[i]->data,
            d_size,
            cudaMemcpyHostToDevice,
            PF_CONFIG.fetch_stream
        );
        cudaEventRecord(PF_CONFIG.d_ready_event[i], PF_CONFIG.fetch_stream);
        
        // Accumulate offset for next expert
        cumulative_offset += d_size;
    }

    // PHASE 2: Parallel G/U Execution
    // Experts 0 to (cpu_threads - 1) on CPU
    #pragma omp parallel for num_threads(PF_CONFIG.cpu_threads)
    for (int i = 0; i < std::min(num_active, PF_CONFIG.cpu_threads); ++i) {
        compute_expert_gu_cpu(expert_ids[i]);
    }

    // Experts cpu_threads to num_active (GPU takes the overflow)
    // Use separate compute stream to avoid serialization with fetch_stream
    if (num_active > PF_CONFIG.cpu_threads) {
        for (int i = PF_CONFIG.cpu_threads; i < num_active; ++i) {
            // Use separate stream for G/U compute to avoid blocking D transfers
            compute_gu_on_gpu_async(expert_ids[i], PF_CONFIG.compute_stream);
        }
    }

    // PHASE 3: The Tail (D-Projection)
    for (int i = 0; i < num_active; ++i) {
        // Wait for D-weight transfer to complete on the compute stream
        cudaStreamWaitEvent(PF_CONFIG.compute_stream, PF_CONFIG.d_ready_event[i], 0);
        
        // Check if transfer actually completed successfully
        cudaError_t status = cudaEventQuery(PF_CONFIG.d_ready_event[i]);
        if (status == cudaSuccess) {
            // Execute D-projection on GPU using the correct offset
            execute_d_projection_gpu(i, 
                (char*)d_weights_gpu_scratch + gpu_offsets[i],
                PF_CONFIG.compute_stream);
        } else {
            // Fallback to CPU if transfer failed or not ready
            execute_d_projection_cpu(i, d_weights_cpu[i]);
        }
    }
}
```

---

### Key Optimization Notes

- **The "6-Core" Window**: On Qwen3-Next (11 active experts), the CPU handles experts 0-5. While the CPU is crunching those, the GPU has time to finish the PCIe transfer for all 11 `D` weights and compute `G/U` for experts 6-10
- **VRAM Management**: For a 4GB GPU, your `d_weights_gpu_scratch` should be a pre-allocated block. At Q4_K_M quantization, 11 experts for Qwen3-Next 80B will occupy approximately 500-700MB, well within your limits
- **Memory Layout**: Each expert's D tensor may have different sizes. Use cumulative offsets when packing into the GPU scratchpad to ensure correct memory layout
- **Stream Separation**: Use separate CUDA streams for weight transfers (`fetch_stream`) and compute operations (`compute_stream`) to avoid serialization bottlenecks
- **CPU Fallback**: Always check transfer completion status and fall back to CPU execution if the transfer fails or times out

---

### Why This Approach Works With Minimal Graph Changes

This approach minimizes changes to the ggml execution graph by implementing at the Backend Dispatch level.

In llama.cpp, the graph is essentially a list of instructions ("Compute this tensor using these inputs"). By the time the graph reaches the CUDA/CPU backend, the instructions are already fixed. To implement Post-Fetch, you intercept the execution of the MoE Op itself.

#### Why Minimal Graph Changes are Needed

- **Backend-Level Implementation**: The MoE layer uses `GGML_OP_MUL_MAT_ID` which dispatches to `ggml_cuda_mul_mat_id()`. You modify the implementation of this function in the backend (`ggml-cuda.cu`)
- **Internal Pipelining**: Inside the backend execution, you spawn your own CUDA streams and manage async transfers. The graph just sees "MoE operation started" and "MoE operation finished." What happens inside—including your async PCIe transfers and CPU/GPU work splitting—is mostly opaque to the graph
- **External Memory Management**: By using a pre-allocated scratchpad in VRAM (managed outside the standard `ggml_allocator`), you avoid triggering graph-level memory allocation logic

---

#### The Hook Point in llama.cpp

The actual integration point in llama.cpp is in the `ggml_cuda_mul_mat_id()` function in `ggml-cuda.cu`, which handles expert selection and execution.

**Current llama.cpp MoE flow:**
1. `llama_build_graph()` creates a `GGML_OP_MUL_MAT_ID` operation for expert routing
2. The graph scheduler dispatches this to `ggml_cuda_mul_mat_id()` 
3. Inside `ggml_cuda_mul_mat_id()`, the selected expert IDs are extracted from the ID tensor
4. Expert weights are loaded and matmuls are executed

**Post-Fetch integration:**
You modify `ggml_cuda_mul_mat_id()` to:
1. Extract the selected expert IDs (already done by the function)
2. Get pointers to the expert FFN-down tensors from the model structure
3. Launch async transfers of D tensors while G/U computation proceeds
4. Synchronize before D computation

```cpp
// In ggml-cuda.cu, inside ggml_cuda_mul_mat_id()
static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // IDs tensor
    const ggml_tensor * src1 = dst->src[1]; // Input activations
    
    // Extract expert IDs (llama.cpp already does this)
    // ...existing code...
    
    // POST-FETCH INTEGRATION POINT
    if (PF_CONFIG.enabled && is_ffn_down_projection(dst)) {
        // Trigger async D-weight transfers
        postfetch_schedule_transfers(expert_ids, num_experts, layer_ctx);
    }
    
    // Continue with normal G/U execution
    // ...existing matmul code...
}
```

---

#### Potential "Smallest Change" Roadblock

The only "Graph-adjacent" issue is Tensor Ownership. Standard llama.cpp expects tensors to live in one place.

**The Fix**: You must ensure the `ffn_down` weights are flagged as `GGML_BACKEND_TYPE_CPU` so the graph doesn't try to move them itself. Your manual `cudaMemcpyAsync` then acts as a "shadow" transfer that the graph isn't even aware of.

---

#### Implementation Integration

Based on the actual llama.cpp codebase, there are two viable approaches:

**Approach A: High-Level Hook (RECOMMENDED for initial implementation)**

Integration in `src/llama-graph.cpp` within the `build_moe_ffn()` function:

```cpp
// In llm_graph_context::build_moe_ffn()
// Location: After G/U computation, before D-projection

struct ggml_tensor * llm_graph_context::build_moe_ffn(
    struct ggml_tensor * cur,
    const llama_layer & layer,
    const llm_ffn_params & ffn_params
) {
    // ... Routing phase (lines 1119-1197) ...
    // ... G/U computation (lines 1251-1270) ...
    // ... Activation (lines 1272-1316) ...
    
    // POST-FETCH HOOK POINT
    if (postfetch_ctx && layer.ffn_down_exps != nullptr) {
        // We have direct access to model structure here
        postfetch_schedule_d_transfers(
            selected_expert_ids,  // From routing result
            n_active_experts,
            layer.ffn_down_exps,  // Direct access to expert tensors
            postfetch_ctx
        );
    }
    
    // ... D-projection (line 1318) ...
    // build_lora_mm_id() will use prefetched weights if ready
    
    return result;
}
```

**Advantages:**
- Direct access to model structure (no context passing needed)
- Occurs before LoRA application (works transparently with LoRA)
- Simpler synchronization with graph execution
- Easier to debug and maintain

**Approach B: Backend-Level Hook (for advanced optimization)**

Integration in `ggml/src/ggml-cuda/ggml-cuda.cu`:

```cpp
// Requires model context access (see Model Access Strategies section)
static void ggml_cuda_mul_mat_id(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    // Extract expert IDs (existing code)
    const ggml_tensor * ids = dst->src[2];
    
    // POST-FETCH: Requires model context (via thread-local or extended context)
    if (g_postfetch_state && is_down_projection(dst)) {
        const auto& model = g_postfetch_state->model;
        int layer_idx = extract_layer_index(dst);
        
        postfetch_schedule_transfers(
            expert_ids,
            n_experts,
            model->layers[layer_idx].ffn_down_exps,
            g_postfetch_state->pf_ctx
        );
    }
    
    // ... existing gather-sort-scatter logic ...
}
```

**Challenges:**
- Requires model context access mechanism
- Must coordinate with existing gather-sort-scatter approach
- More complex synchronization with ggml's stream management

**Recommendation:** Start with Approach A, migrate to Approach B if performance profiling shows benefit.

---

### Model Access Strategies (for Backend-Level Hook)

If implementing at the backend level, model context must be accessible:

---

### Model Access Strategies (for Backend-Level Hook)

If implementing at the backend level, model context must be accessible:

**Strategy 1: Thread-Local State (Simple)**

```cpp
// Global state accessible across compilation units
struct postfetch_global_state {
    const llama_model* model;
    postfetch_context* pf_ctx;
};

thread_local postfetch_global_state* g_postfetch_state = nullptr;

// Initialize during llama context creation
void llama_new_context_with_model(
    llama_model* model,
    llama_context_params params
) {
    // ... existing initialization ...
    
    // Setup Post-Fetch state
    g_postfetch_state = new postfetch_global_state();
    g_postfetch_state->model = model;
    g_postfetch_state->pf_ctx = postfetch_init(model);
    
    // ... rest of initialization ...
}

// Access in CUDA backend
void ggml_cuda_mul_mat_id(...) {
    if (g_postfetch_state && g_postfetch_state->pf_ctx->enabled) {
        const auto& model = g_postfetch_state->model;
        const auto& layer = model->layers[layer_idx];
        // Now we have model access
    }
}
```

**Strategy 2: Extended Backend Context (Cleaner)**

```cpp
// Extend ggml backend context with userdata pointer
struct ggml_backend_cuda_context_extended {
    ggml_backend_cuda_context base;
    void* llama_userdata;  // Points to llama_context
};

// Modified backend initialization
ggml_backend_t ggml_backend_cuda_init_with_userdata(
    int device,
    void* userdata  // llama_context pointer
) {
    auto* ctx = new ggml_backend_cuda_context_extended();
    ctx->llama_userdata = userdata;
    // ... rest of initialization ...
    return (ggml_backend_t)ctx;
}

// Access model in backend
void ggml_cuda_mul_mat_id(
    ggml_backend_cuda_context & ctx_base,
    ggml_tensor * dst
) {
    auto* ctx = (ggml_backend_cuda_context_extended*)&ctx_base;
    auto* lctx = (llama_context*)ctx->llama_userdata;
    const auto& model = lctx->model;
    // Now we have model access
}
```

**Strategy 3: Tensor Metadata (Most Elegant)**

```cpp
// Attach model/layer info to tensor during graph construction
void llm_graph_context::build_moe_ffn(...) {
    // ... create expert tensors ...
    
    // Tag tensors with layer information
    for (auto* tensor : expert_tensors) {
        tensor->op_params[0] = layer_idx;  // Store layer index
        tensor->op_params[1] = (int64_t)&layer;  // Store layer pointer
    }
}

// Extract in backend
void ggml_cuda_mul_mat_id(..., ggml_tensor * dst) {
    int layer_idx = dst->op_params[0];
    const llama_layer* layer = (const llama_layer*)dst->op_params[1];
    
    // Now we have layer access without global state
    struct ggml_tensor* d_weight = layer->ffn_down_exps[expert_idx];
}
```

**Recommendation:** 
- For initial implementation: Strategy 1 (thread-local) - simplest
- For production: Strategy 3 (tensor metadata) - cleanest architecture

---

### LoRA Adapter Considerations

The interaction with LoRA adapters depends on the hook level:

**High-Level Hook (build_moe_ffn):**

LoRA is transparently compatible:

```cpp
// In build_moe_ffn(), we schedule transfers BEFORE build_lora_mm_id() is called
postfetch_schedule_d_transfers(...);  // Transfers base expert weights

// Later, build_lora_mm_id() is called
struct ggml_tensor * down_result = build_lora_mm_id(
    ctx0,
    layer.ffn_down_exps,  // Base weights (potentially prefetched to GPU)
    cur,
    selected_expert_ids
);

// Inside build_lora_mm_id() (from llama-graph.cpp:856-874):
ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
for (const auto & lora : *loras) {
    // LoRA deltas are applied AFTER base matmul
    // Post-Fetch doesn't interfere with this
}
```

**Result:** Post-Fetch accelerates the base matmul, LoRA application happens afterward. No special handling needed.

**Backend-Level Hook (ggml_cuda_mul_mat_id):**

Must detect whether tensor is a base weight or LoRA-adapted result:

```cpp
void ggml_cuda_mul_mat_id(..., ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    
    // Check if this is a base expert weight tensor
    bool is_base_weight = (weights->flags & GGML_TENSOR_FLAG_MODEL_WEIGHT);
    
    if (is_base_weight && postfetch_enabled) {
        // Safe to prefetch - this is the original model weight
        postfetch_schedule_transfer(weights, expert_ids);
    } else {
        // This might be a LoRA-adapted ephemeral tensor
        // Skip prefetching (tensor lifecycle is different)
    }
}
```

**Recommendation:** High-level hook avoids LoRA complexity entirely.

---

### Stream Coordination with ggml

Post-Fetch streams must coordinate with ggml's existing CUDA stream:

```cpp
struct postfetch_context {
    cudaStream_t fetch_stream;     // Our stream for H2D transfers
    cudaStream_t compute_stream;   // Our stream for compute
    // ggml has its own stream via ctx.stream()
};

// Synchronization strategy
void postfetch_sync_with_ggml(
    ggml_backend_cuda_context* ggml_ctx,
    postfetch_context* pf_ctx
) {
    cudaEvent_t sync_event;
    cudaEventCreate(&sync_event);
    
    // Record when our compute finishes
    cudaEventRecord(sync_event, pf_ctx->compute_stream);
    
    // Make ggml's stream wait for our work
    cudaStreamWaitEvent(ggml_ctx->stream(), sync_event, 0);
    
    cudaEventDestroy(sync_event);
}

// Usage in MoE execution
void execute_moe_with_postfetch(...) {
    // 1. Schedule transfers on fetch_stream
    postfetch_schedule_transfers(..., pf_ctx->fetch_stream);
    
    // 2. Compute G/U (uses ggml's stream)
    ggml_compute_gu(...);
    
    // 3. Compute D (uses compute_stream, waits for fetch_stream)
    cudaStreamWaitEvent(pf_ctx->compute_stream, d_ready_event, 0);
    execute_d_projection(..., pf_ctx->compute_stream);
    
    // 4. Sync back to ggml's stream
    postfetch_sync_with_ggml(ggml_ctx, pf_ctx);
}
```

**Key insight:** Multiple streams are standard CUDA practice. llama.cpp already uses multiple streams in flash attention and other optimizations.

---

### Implementation Scope

**What needs to be modified:**

- **Backend Integration** (`ggml-cuda.cu`): Modify `ggml_cuda_mul_mat_id()` to trigger async D-weight transfers
- **Memory Management**: Add GPU scratchpad allocation/deallocation hooks
- **Stream Management**: Create and manage dedicated CUDA streams (`fetch_stream`, `compute_stream`)
- **Configuration System**: Add environment variable parsing for Post-Fetch settings

**What remains unchanged:**

- **Graph Structure**: The `ggml_graph` sees the same sequence of operations
- **Model Semantics**: Output tensors remain bit-identical to baseline execution
- **Kernel Code**: No modifications to matmul kernels or other compute primitives
- **Allocator Logic**: GPU scratchpad managed separately from `ggml_allocr`

**Memory Footprint:**

- **VRAM**: Only need space for one layer's active experts (typically 400-800MB for Q4_K_M quantization)
- **No persistent caching**: Scratchpad is reused across layers, no cross-token state
- **Fallback safety**: If VRAM is exhausted, execution falls back to CPU seamlessly

---

### GPU Scratchpad Management

The GPU scratchpad is a critical component that requires careful lifecycle management:

#### Allocation Strategy

```cpp
// During context initialization (llama_new_context_with_model)
struct postfetch_context {
    void* gpu_scratchpad;
    size_t scratchpad_size;
    cudaStream_t fetch_stream;
    cudaStream_t compute_stream;
    cudaEvent_t d_ready_event[MAX_EXPERTS];
    size_t current_gpu_offsets[MAX_EXPERTS];
    int transfers_in_flight;
    std::mutex scratchpad_mutex;  // For thread safety
};

void postfetch_init(postfetch_context* pf_ctx, const llama_model* model) {
    if (!PF_CONFIG.enabled) return;
    
    // Calculate required scratchpad size
    // Find the largest possible expert activation across all layers
    size_t max_experts_per_layer = 0;
    size_t max_expert_size = 0;
    
    for (const auto& layer : model->layers) {
        if (!layer.ffn_down_exps.empty()) {
            max_experts_per_layer = std::max(
                max_experts_per_layer, 
                layer.ffn_down_exps.size()
            );
            
            for (const auto* tensor : layer.ffn_down_exps) {
                max_expert_size = std::max(
                    max_expert_size,
                    ggml_nbytes(tensor)
                );
            }
        }
    }
    
    // Allocate scratchpad: max_active_experts * max_expert_size
    // For Qwen3-Next: 11 experts * ~60MB = ~660MB
    pf_ctx->scratchpad_size = max_experts_per_layer * max_expert_size;
    
    cudaMalloc(&pf_ctx->gpu_scratchpad, pf_ctx->scratchpad_size);
    
    // Create dedicated streams
    cudaStreamCreate(&pf_ctx->fetch_stream);
    cudaStreamCreate(&pf_ctx->compute_stream);
    
    // Create events for synchronization
    for (int i = 0; i < MAX_EXPERTS; ++i) {
        cudaEventCreate(&pf_ctx->d_ready_event[i]);
    }
}

void postfetch_free(postfetch_context* pf_ctx) {
    if (pf_ctx->gpu_scratchpad) {
        cudaFree(pf_ctx->gpu_scratchpad);
    }
    cudaStreamDestroy(pf_ctx->fetch_stream);
    cudaStreamDestroy(pf_ctx->compute_stream);
    for (int i = 0; i < MAX_EXPERTS; ++i) {
        cudaEventDestroy(pf_ctx->d_ready_event[i]);
    }
}
```

#### Multi-Layer Concurrency Handling

**Challenge**: Multiple MoE layers might execute concurrently in multi-threaded scenarios.

**Solution**: Use mutex protection around scratchpad access:

```cpp
void llama_postfetch_schedule_d_transfers(
    const int32_t* expert_ids,
    const int n_experts,
    const int layer_idx,
    postfetch_context* pf_ctx
) {
    // Lock scratchpad for this layer's execution
    std::lock_guard<std::mutex> lock(pf_ctx->scratchpad_mutex);
    
    // Scratchpad is now exclusively ours until function returns
    // Proceed with transfers...
}
```

**Alternative for higher throughput**: Allocate per-thread scratchpads if memory permits:

```cpp
// In postfetch_init, allocate N scratchpads for N threads
const int n_threads = /* from context */;
pf_ctx->gpu_scratchpads = new void*[n_threads];
for (int i = 0; i < n_threads; ++i) {
    cudaMalloc(&pf_ctx->gpu_scratchpads[i], scratchpad_size);
}

// During execution, use thread-local scratchpad
int thread_id = omp_get_thread_num();
void* my_scratchpad = pf_ctx->gpu_scratchpads[thread_id];
```

---

### Identification of Expert Tensors in llama.cpp

Expert weights in llama.cpp are stored in the model structure with a predictable layout:

```cpp
// In llama.cpp's model structure (llama.h / llama.cpp)
struct llama_layer {
    // ... other layer components ...
    
    // MoE expert weights (if layer is MoE)
    std::vector<struct ggml_tensor *> ffn_gate_exps;  // Gate projections
    std::vector<struct ggml_tensor *> ffn_up_exps;    // Up projections
    std::vector<struct ggml_tensor *> ffn_down_exps;  // Down projections (target for Post-Fetch)
};

// Accessing expert tensors at runtime:
void get_expert_d_tensor(
    const llama_model& model,
    int layer_idx,
    int expert_idx,
    struct ggml_tensor** out_tensor
) {
    const auto& layer = model.layers[layer_idx];
    
    // Verify this is an MoE layer
    if (layer.ffn_down_exps.empty()) {
        *out_tensor = nullptr;
        return;
    }
    
    // Bounds check
    if (expert_idx >= layer.ffn_down_exps.size()) {
        *out_tensor = nullptr;
        return;
    }
    
    *out_tensor = layer.ffn_down_exps[expert_idx];
}
```

**During model loading**, expert tensors are identified by their naming pattern in the GGUF file:
- Format: `blk.{layer}.ffn_down.{expert}.weight`
- Example: `blk.0.ffn_down.5.weight` (layer 0, expert 5, down projection)

The model loader populates the `ffn_down_exps` vector during `llama_model_load()`, making tensors accessible at runtime by layer and expert index.

---

## Implementation Pitfalls and Debugging

### Common Implementation Mistakes

#### 1. Incorrect Memory Layout (Critical Bug)
```cpp
// ❌ WRONG: Assumes all experts have same size
cudaMemcpyAsync(scratchpad + (i * MAX_SIZE), ...);

// ✅ CORRECT: Use cumulative offsets
size_t offset = 0;
for (int i = 0; i < n; ++i) {
    cudaMemcpyAsync(scratchpad + offset, ..., sizes[i], ...);
    offsets[i] = offset;
    offset += sizes[i];
}
```

#### 2. Stream Synchronization Errors
```cpp
// ❌ WRONG: Waits on NULL (default) stream
cudaStreamWaitEvent(NULL, event[i], 0);

// ✅ CORRECT: Wait on the actual compute stream
cudaStreamWaitEvent(compute_stream, event[i], 0);
```

#### 3. Missing CPU Fallback
```cpp
// ❌ WRONG: Assumes transfer always succeeds
cudaStreamWaitEvent(stream, event, 0);
execute_on_gpu(gpu_ptr);  // What if transfer failed?

// ✅ CORRECT: Check and fallback
cudaStreamWaitEvent(stream, event, 0);
if (cudaEventQuery(event) == cudaSuccess) {
    execute_on_gpu(gpu_ptr);
} else {
    execute_on_cpu(cpu_ptr);  // Fallback
}
```

#### 4. Tensor Ownership Confusion
```cpp
// ❌ WRONG: Assumes tensors stay on CPU
// The graph might have moved them to GPU already!
cudaMemcpyAsync(gpu, tensor->data, ...);

// ✅ CORRECT: Check backend type
if (tensor->backend == GGML_BACKEND_TYPE_CPU) {
    cudaMemcpyAsync(gpu, tensor->data, ...);
} else {
    // Already on GPU, skip transfer or copy GPU→GPU
}
```

### Debugging Strategies

#### Enable CUDA Error Checking
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
} while(0)

// Use in all CUDA calls during development
CUDA_CHECK(cudaMemcpyAsync(...));
CUDA_CHECK(cudaStreamWaitEvent(...));
```

#### Log Transfer Activity
```cpp
void postfetch_schedule_transfers(...) {
    if (getenv("LLAMA_POSTFETCH_DEBUG")) {
        fprintf(stderr, "[PF] Layer %d: Transferring %d experts, total %zu MB\n",
                layer_idx, n_experts, cumulative_size / (1024*1024));
    }
    
    for (int i = 0; i < n_experts; ++i) {
        // ... transfer code ...
        
        if (getenv("LLAMA_POSTFETCH_DEBUG")) {
            fprintf(stderr, "[PF]   Expert %d: %zu bytes @ offset %zu\n",
                    expert_ids[i], sizes[i], offsets[i]);
        }
    }
}
```

#### Measure Transfer vs. Compute Overlap
```cpp
// Add timing instrumentation
cudaEvent_t start_transfer, end_transfer, end_compute;
cudaEventCreate(&start_transfer);
cudaEventCreate(&end_transfer);
cudaEventCreate(&end_compute);

cudaEventRecord(start_transfer, fetch_stream);
// ... schedule transfers ...
cudaEventRecord(end_transfer, fetch_stream);

// ... compute G/U ...
cudaEventRecord(end_compute, compute_stream);

cudaEventSynchronize(end_compute);

float transfer_ms, total_ms;
cudaEventElapsedTime(&transfer_ms, start_transfer, end_transfer);
cudaEventElapsedTime(&total_ms, start_transfer, end_compute);

fprintf(stderr, "[PF] Transfer: %.2f ms, Total: %.2f ms, Overlap: %.1f%%\n",
        transfer_ms, total_ms, 
        100.0 * (1.0 - total_ms / (transfer_ms + compute_ms)));
```

### Validation Checklist

Before considering the implementation complete:

- [ ] Memory layout uses cumulative offsets, not fixed strides
- [ ] Stream synchronization uses correct stream handles
- [ ] CPU fallback is implemented and tested
- [ ] Tensor backend types are checked before transfers
- [ ] CUDA errors are checked in debug builds
- [ ] Multi-layer concurrent access is thread-safe
- [ ] Scratchpad size is sufficient for largest layer
- [ ] Event arrays are properly sized for max experts
- [ ] Configuration is loaded at context initialization
- [ ] Resources are freed in context destruction

---

## Critical Implementation Pitfalls

This section expands on the common mistakes identified in the "Implementation Pitfalls and Debugging" section with additional context and solutions.

### 5. CUDA Context Access Strategy

**The Problem:** The `ggml_backend_cuda_context` does NOT expose the model directly.

**Three Viable Approaches:**

#### Strategy 1: Thread-Local State (Simplest for initial implementation)
```cpp
// Global state accessible across compilation units
struct postfetch_global_state {
    const llama_model* model;
    postfetch_context* pf_ctx;
};

thread_local postfetch_global_state* g_postfetch_state = nullptr;

// Initialize during llama context creation
void llama_new_context_with_model(
    llama_model* model,
    llama_context_params params
) {
    // ... existing initialization ...
    
    // Setup Post-Fetch state
    g_postfetch_state = new postfetch_global_state();
    g_postfetch_state->model = model;
    g_postfetch_state->pf_ctx = postfetch_init(model);
    
    // ... rest of initialization ...
}

// Access in CUDA backend
void ggml_cuda_mul_mat_id(...) {
    if (g_postfetch_state && g_postfetch_state->pf_ctx->enabled) {
        const auto& model = g_postfetch_state->model;
        const auto& layer = model->layers[layer_idx];
        // Now we have model access
    }
}
```

**Pros:** Easy to implement, minimal code changes
**Cons:** Global state, potential thread safety issues

#### Strategy 2: Extended Backend Context (Cleaner architecture)
```cpp
// Extend ggml backend context with userdata pointer
struct ggml_backend_cuda_context_extended {
    ggml_backend_cuda_context base;
    void* userdata;  // Points to llama_context
};

// Modified backend initialization
ggml_backend_t ggml_backend_cuda_init_with_userdata(
    int device,
    void* userdata  // llama_context pointer
) {
    auto* ctx = new ggml_backend_cuda_context_extended();
    ctx->userdata = userdata;
    // ... rest of initialization ...
    return (ggml_backend_t)ctx;
}

// Access model in backend
void ggml_cuda_mul_mat_id(
    ggml_backend_cuda_context & ctx_base,
    ggml_tensor * dst
) {
    auto* ctx = (ggml_backend_cuda_context_extended*)&ctx_base;
    auto* lctx = (llama_context*)ctx->userdata;
    const auto& model = lctx->model;
    // Now we have model access
}
```

**Pros:** Proper encapsulation, no global state
**Cons:** Requires modifying ggml backend initialization

#### Strategy 3: Tensor Metadata (Most elegant)
```cpp
// Attach model/layer info to tensor during graph construction
void llm_graph_context::build_moe_ffn(...) {
    // ... create expert tensors ...
    
    // Tag tensors with layer information
    for (auto* tensor : expert_tensors) {
        tensor->op_params[0] = layer_idx;  // Store layer index
        tensor->op_params[1] = (int64_t)&layer;  // Store layer pointer
    }
}

// Extract in backend
void ggml_cuda_mul_mat_id(..., ggml_tensor * dst) {
    int layer_idx = dst->op_params[0];
    const llama_layer* layer = (const llama_layer*)dst->op_params[1];
    
    // Now we have layer access without global state
    struct ggml_tensor* d_weight = layer->ffn_down_exps[expert_idx];
}
```

**Pros:** No global state, clean separation
**Cons:** Requires modifying graph construction

**Recommendation:** Start with Strategy 1 (thread-local) for Phase 1, migrate to Strategy 3 (tensor metadata) for production.

### 6. LoRA Compatibility Matrix

| Hook Level | LoRA Compatibility | Special Handling Required |
|------------|-------------------|--------------------------|
| High-level (`build_moe_ffn`) | ✅ Transparent | None - LoRA applied after Post-Fetch |
| Backend-level (`ggml_cuda_mul_mat_id`) | ⚠️ Conditional | Must detect base weights vs LoRA-adapted tensors |

**At high-level hook (`build_moe_ffn`):**
- LoRA is applied INSIDE `build_lora_mm_id()`
- Post-Fetch triggers BEFORE `build_lora_mm_id()` is called
- We transfer base expert weights; LoRA deltas are applied during matmul
- **This works correctly without special handling**

**At backend-level hook (`ggml_cuda_mul_mat_id`):**
- LoRA has already been applied by the time we reach the backend
- The tensor we see is potentially LoRA-adapted
- **We need to detect if tensor is base weight or LoRA-adapted**

```cpp
void ggml_cuda_mul_mat_id(..., ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    
    // Check if this is a base weight tensor
    bool is_base_weight = (weights->flags & GGML_TENSOR_FLAG_MODEL_WEIGHT);
    
    if (is_base_weight && postfetch_enabled) {
        // Safe to prefetch - this is the original model weight
        postfetch_schedule_transfer(weights, expert_ids);
    } else {
        // This might be a LoRA-adapted ephemeral tensor
        // Skip prefetching (tensor lifecycle is different)
    }
}
```

### 7. Stream Coordination Best Practices

**The Problem:** Creating additional streams requires careful synchronization with ggml's existing stream management.

**Our Solution:** Post-Fetch streams are ADDITIONAL, not replacing ggml's stream.

#### Stream Architecture
```
fetch_stream:     H2D transfers only
compute_stream:   D-projection compute only
ggml_stream:      All other graph operations
```

#### Synchronization Pattern
```cpp
struct postfetch_context {
    cudaStream_t fetch_stream;     // Our stream for H2D transfers
    cudaStream_t compute_stream;   // Our stream for compute
    // ggml has its own stream via ctx.stream()
};

// Synchronization strategy
void postfetch_sync_with_ggml(
    ggml_backend_cuda_context* ggml_ctx,
    postfetch_context* pf_ctx
) {
    cudaEvent_t sync_event;
    cudaEventCreate(&sync_event);
    
    // Record when our compute finishes
    cudaEventRecord(sync_event, pf_ctx->compute_stream);
    
    // Make ggml's stream wait for our work
    cudaStreamWaitEvent(ggml_ctx->stream(), sync_event, 0);
    
    cudaEventDestroy(sync_event);
}
```

**Critical Rule:** Never assume stream ordering. Always use events for cross-stream synchronization.

**Precedent in llama.cpp:** Multiple CUDA streams are already used in flash attention and other optimizations. This is standard CUDA practice.

### 8. Configuration Validation

Add a **"Configuration Validation Checklist"** that users should verify before testing:

| Variable | Default | Validation Check |
|----------|---------|------------------|
| `LLAMA_POSTFETCH_ENABLE` | `1` | Set explicitly to 1 for testing |
| `LLAMA_POSTFETCH_FORCE_CPU` | `0` | Set to 0 unless debugging |
| `LLAMA_POSTFETCH_BLOCK_ON_MISS` | `1` | Set based on use case (1=latency-sensitive, 0=throughput) |
| `LLAMA_POSTFETCH_MAX_TRANSFERS` | `8` | Verify ≤ GPU memory allows |
| `LLAMA_POSTFETCH_SCRATCHPAD_MB` | `0` | Calculate for largest layer |

**Before testing, verify:**
- [ ] `LLAMA_POSTFETCH_ENABLE=1` explicitly set
- [ ] `LLAMA_POSTFETCH_FORCE_CPU=0` (unless debugging)
- [ ] `LLAMA_POSTFETCH_BLOCK_ON_MISS` matches use case
- [ ] `LLAMA_POSTFETCH_MAX_TRANSFERS` ≤ actual GPU memory allows
- [ ] `LLAMA_POSTFETCH_SCRATCHPAD_MB` calculated correctly for largest layer

### 9. Debugging Environment Variables

Add documentation for debugging-specific environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLAMA_POSTFETCH_DEBUG` | Enable verbose logging | `export LLAMA_POSTFETCH_DEBUG=1` |
| `CUDA_LAUNCH_BLOCKING=1` | Synchronize all CUDA calls | For debugging race conditions |
| `CUDA_ERROR_CHECKING=1` | Enable CUDA error checking | For development builds |

**Usage example:**
```bash
# Enable Post-Fetch with debug logging
export LLAMA_POSTFETCH_ENABLE=1
export LLAMA_POSTFETCH_DEBUG=1

# For debugging race conditions
export CUDA_LAUNCH_BLOCKING=1

# Run with debugging enabled
./main -m models/qwen3-8b.Q4_K_M.gguf -p "Hello" --postfetch-enable 1
```

### 10. Testing Strategy

Add a **"Testing Checklist"** section:

#### Unit Tests
- [ ] Verify memory layout with known tensor sizes
- [ ] Test stream synchronization with timing measurements
- [ ] Validate CPU fallback when GPU transfer fails

#### Integration Tests
- [ ] Run with all target models (GPT-OSS 120B, 20B, Qwen3-Next 80B)
- [ ] Test with and without LoRA adapters
- [ ] Verify identical outputs with/without Post-Fetch

#### Performance Tests
- [ ] Measure PCIe transfer overlap percentage
- [ ] Verify no regression on high-VRAM systems
- [ ] Test multi-threaded execution for race conditions

### 11. Implementation Roadmap

Add a clear **"Implementation Roadmap"** showing the phased approach:

```
Phase 1 (Week 1-2): High-level hook
├── Hook in build_moe_ffn()
├── Thread-local model access
├── Basic transfer logic
└── CPU fallback

Phase 2 (Week 3-4): Backend optimization
├── Tensor metadata approach
├── Extended backend context
├── Multi-stream optimization
└── LoRA detection

Phase 3 (Week 5-6): Production hardening
├── Memory profiling
├── Multi-GPU support
├── Performance tuning
└── Documentation
```

### 12. Common Error Messages and Solutions

Add a reference table for debugging:

| Error | Cause | Solution |
|-------|-------|----------|
| `cudaErrorInvalidValue` | Incorrect offset calculation | Use cumulative offsets |
| `cudaErrorLaunchFailure` | Stream synchronization issue | Add proper event waits |
| `Segmentation fault` | Tensor ownership confusion | Check backend type first |
| `cudaErrorNotReady` | Missing fallback logic | Implement CPU fallback |
| `cudaErrorInvalidDevicePointer` | Wrong scratchpad allocation | Verify VRAM allocation size |

### 13. Performance Monitoring

Add guidance on measuring Post-Fetch effectiveness:

#### Key Metrics to Track
- Transfer overlap percentage: `(total_time - transfer_time) / total_time`
- GPU utilization during MoE layers
- VRAM usage patterns
- Fallback rate (CPU execution when GPU expected)

#### Timing Instrumentation
```cpp
cudaEvent_t t_start, t_end;
cudaEventCreate(&t_start);
cudaEventCreate(&t_end);

cudaEventRecord(t_start, fetch_stream);
// ... transfers ...
cudaEventRecord(t_end, compute_stream);

cudaEventSynchronize(t_end);
float ms;
cudaEventElapsedTime(&ms, t_start, t_end);

fprintf(stderr, "[PF] Transfer completed in %.2f ms\n", ms);
```

#### Expected Performance Targets
| Metric | Target | Measurement |
|--------|--------|-------------|
| Transfer overlap | ≥70% | `cudaEventElapsedTime` |
| Fallback rate | ≤5% | Counter in readiness query |
| VRAM overhead | ≤10% of available | `cudaMemGetInfo` |

---

## Response to Code Review

This section addresses concerns raised in the codebase review and indicates where to find corrections.

### Issue 1: CUDA Context Access - ADDRESSED ✅

**Reviewer concern:** "The `ggml_backend_cuda_context` does NOT expose `model` directly."

**Our response:** Correct. We now explicitly document three model access strategies:
- **Where to look:** "Model Access Strategies" section (lines ~1237-1330)
- **Recommended approach:** Thread-local state (simplest) or tensor metadata (cleanest)

### Issue 2: Duplicate Buggy Code - FIXED ✅

**Reviewer concern:** "The earlier implementation sketch (line 1134) still contains the bug."

**Our response:** Confirmed and removed. The duplicate snippet has been deleted.
- **What was removed:** Second implementation snippet that incorrectly used `i * MAX_SIZE`
- **What remains:** Only the CORRECT implementation using cumulative offsets (lines 1013-1024)

### Issue 3: LoRA Support - ADDRESSED ✅

**Reviewer concern:** "The implementation snippets do not account for LoRA adapters."

**Our response:** Added comprehensive LoRA considerations:
- **Where to look:** "LoRA Adapter Considerations" section (lines ~1332-1395)
- **Key insight:** High-level hook (build_moe_ffn) works transparently with LoRA
- **Backend hook:** Requires base weight detection via tensor flags

### Issue 4: Stream Management - CLARIFIED ✅

**Reviewer concern:** "Creating additional streams requires careful synchronization with ggml's existing stream management."

**Our response:** Documented stream coordination strategy:
- **Where to look:** "Stream Coordination with ggml" section (lines ~1397-1450)
- **Key point:** Post-Fetch streams are ADDITIONAL, not replacing ggml's stream
- **Precedent:** Multiple streams already used in llama.cpp (flash attention, etc.)

### Issue 5: Implementation Approach - RESTRUCTURED ✅

**Reviewer recommendation:** "Implement at the llama.cpp level (not ggml backend) to access model context"

**Our response:** Restructured document to emphasize phased approach:
- **Where to look:** "Implementation Integration" section (lines ~1209-1305)
- **Phase 1 (RECOMMENDED):** High-level hook in build_moe_ffn
- **Phase 2 (Advanced):** Backend-level hook in ggml_cuda_mul_mat_id
- Clear trade-offs and migration path documented

### What We Agree With

1. ✅ High-level hook (build_moe_ffn) is simpler and recommended for initial implementation
2. ✅ Backend-level hook requires model context access mechanism
3. ✅ LoRA support must be explicitly handled at backend level
4. ✅ Duplicate code snippet was an editing error and has been removed

### What We Defend

1. **Cumulative offset implementation (lines 1013-1024) was ALREADY CORRECT in v0.1.0**
   - Reviewer missed that we had TWO implementations: one correct, one buggy (duplicate)
   - The buggy one has now been removed

2. **Tensor backend checking (lines 1588+) was ALREADY in v0.1.0**
   - Documented in "Implementation Pitfalls" section
   - Explicitly checks `tensor->backend == GGML_BACKEND_TYPE_CPU` before transfer

3. **Multiple CUDA streams are standard practice**
   - Not replacing ggml's stream, but adding our own for async operations
   - Standard CUDA pattern used throughout high-performance codebases

### Validation Points for Reviewer

**To verify our fixes, check:**

| Concern | Document Location | Status |
|---------|------------------|--------|
| Memory layout bug | Lines 1013-1024 | ✅ CORRECT (uses cumulative offsets) |
| Duplicate buggy code | Removed (was ~1131-1164) | ✅ DELETED |
| Model access strategy | Lines 1237-1330 | ✅ ADDED (3 strategies documented) |
| LoRA considerations | Lines 1332-1395 | ✅ ADDED (both hook levels) |
| Stream coordination | Lines 1397-1450 | ✅ ADDED (sync strategy) |
| Tensor backend check | Lines 1588+ | ✅ EXISTS (in pitfalls section) |
| High-level hook priority | Lines 1209-1235 | ✅ RESTRUCTURED (now recommended) |

---

## Summary

**Post-Fetch** is a minimal, counter-intuitive, but correct optimization:

- Fetch weights *after* they are known to be needed
- Hide PCIe latency under unavoidable CPU computation
- Avoid all caching complexity
- Target the weakest consumer hardware first

It trades peak performance for **robustness, simplicity, and correctness** — exactly what low-VRAM llama.cpp users need.

### Key Takeaways

| Aspect | Post-Fetch Approach |
|--------|---------------------|
| **Complexity** | Minimal (stateless, atomic transfers) |
| **Risk** | Low (safe fallback, no semantics change) |
| **Impact** | Focused (tail latency reduction) |
| **Integration** | Easy (single hook, external config) |
| **Maintenance** | Sustainable (small codebase, clear boundaries) |

---

## Related Documentation

- [`memory-bank/MoE_Postfetch_Design.md`](./MoE_Postfetch_Design.md) - Original design notes
- [`memory-bank/MOE_CRITICAL_UPDATES.md`](./MOE_CRITICAL_UPDATES.md) - Critical updates and changes
- [`memory-bank/MOE_UNIFIED_DOCUMENTATION3.md`](./MOE_UNIFIED_DOCUMENTATION3.md) - Unified MoE documentation

---

*Document Version: 0.0.3 (Implementation-Guided)*
*Last Updated: 2026-02-07*
*Incorporates implementation guidance and critical pitfalls from codebase review*
