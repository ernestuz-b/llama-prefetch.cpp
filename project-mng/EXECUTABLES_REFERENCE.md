# llama.cpp Executables Reference

This document describes all executables produced by the llama.cpp build system, their purposes, usage examples, and relevance to the MoE Post-Fetch optimization project.

---

## Table of Contents

1. [Core Inference Tools](#core-inference-tools)
   - [llama-cli](#llama-cli)
   - [llama-server](#llama-server)
   - [llama-completion](#llama-completion)
2. [Benchmarking Tools](#benchmarking-tools)
   - [llama-bench](#llama-bench)
   - [llama-batched-bench](#llama-batched-bench)
   - [llama-perplexity](#llama-perplexity)
3. [Model Processing Tools](#model-processing-tools)
   - [llama-quantize](#llama-quantize)
   - [llama-imatrix](#llama-imatrix)
   - [llama-gguf-split](#llama-gguf-split)
4. [Utility Tools](#utility-tools)
   - [llama-tokenize](#llama-tokenize)
   - [llama-fit-params](#llama-fit-params)
5. [Specialized Tools](#specialized-tools)
   - [llama-tts](#llama-tts)
   - [llama-mtmd](#llama-mtmd)
   - [llama-cvector-generator](#llama-cvector-generator)
   - [llama-export-lora](#llama-export-lora)
   - [llama-rpc-server](#llama-rpc-server)
6. [MoE Post-Fetch Relevance](#moe-post-fetch-relevance)

---

## Core Inference Tools

### llama-cli

**Purpose:** Interactive command-line interface for conversing with LLMs. Provides a chat-like experience with support for multimodal inputs, conversation history, and various sampling parameters.

**Source:** [`tools/cli/cli.cpp`](../tools/cli/cli.cpp)

**Usage Examples:**

```bash
# Basic chat with a model
./llama-cli -m model.gguf -sys "You are a helpful assistant"

# Text generation (non-conversational)
./llama-cli -m model.gguf -p "I believe the meaning of life is" -n 128 -no-cnv

# With GPU offloading
./llama-cli -m model.gguf -ngl 99 -sys "You are a helpful assistant"

# With specific sampling parameters
./llama-cli -m model.gguf --temp 0.7 --top-k 40 --top-p 0.9
```

**Key Options:**
- `-m, --model` - Path to GGUF model file
- `-sys, --system-prompt` - System prompt for chat mode
- `-p, --prompt` - Input prompt
- `-n, --n-predict` - Number of tokens to predict
- `-ngl, --n-gpu-layers` - Number of layers to offload to GPU
- `--temp` - Temperature for sampling
- `-no-cnv` - Disable conversation mode

---

### llama-server

**Purpose:** HTTP server providing an OpenAI-compatible API for LLM inference. Enables integration with applications that expect OpenAI API endpoints.

**Source:** [`tools/server/server.cpp`](../tools/server/server.cpp)

**Documentation:** [`tools/server/README.md`](../tools/server/README.md)

**Usage Examples:**

```bash
# Start server on default port (8080)
./llama-server -m model.gguf

# With GPU offloading and custom port
./llama-server -m model.gguf -ngl 99 --port 8000

# With context size and batching configuration
./llama-server -m model.gguf -c 4096 -b 512 -ub 256

# With API key authentication
./llama-server -m model.gguf --api-key "your-api-key"
```

**API Endpoints:**
- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Generate embeddings
- `GET /v1/models` - List available models
- `GET /health` - Health check endpoint

**MoE Post-Fetch Relevance:** The server architecture is a primary target for Post-Fetch optimization, as it handles concurrent requests and would benefit from efficient expert prefetching for MoE models.

---

### llama-completion

**Purpose:** Simple text completion tool for generating text continuations. Useful for scripting and batch processing.

**Source:** [`tools/completion/completion.cpp`](../tools/completion/completion.cpp)

**Usage Examples:**

```bash
# Basic text completion
./llama-completion -m model.gguf -p "The quick brown fox"

# With specific number of tokens
./llama-completion -m model.gguf -p "Once upon a time" -n 100

# Interactive mode
./llama-completion -m model.gguf -i

# With antiprompt (stop sequence)
./llama-completion -m model.gguf -p "Q: What is AI? A:" -r "Q:"
```

---

## Benchmarking Tools

### llama-bench

**Purpose:** Comprehensive performance benchmarking tool for measuring inference speed across various configurations. Tests prompt processing (pp), token generation (tg), and combined workloads.

**Source:** [`tools/llama-bench/llama-bench.cpp`](../tools/llama-bench/llama-bench.cpp)

**Usage Examples:**

```bash
# Basic benchmark
./llama-bench -m model.gguf

# Test specific prompt/generation sizes
./llama-bench -m model.gguf -p 512,1024,2048 -n 128,256

# With GPU offloading
./llama-bench -m model.gguf -ngl 99

# Output in different formats
./llama-bench -m model.gguf -o json
./llama-bench -m model.gguf -o csv
./llama-bench -m model.gguf -o md

# Multiple repetitions for statistical accuracy
./llama-bench -m model.gguf -r 10

# Test different batch sizes
./llama-bench -m model.gguf -b 512,1024,2048 -ub 128,256,512
```

**Output Metrics:**
- `pp` - Prompt processing (tokens/second)
- `tg` - Token generation (tokens/second)
- `tpp` - Time per prompt token (ms)
- `ttg` - Time per generated token (ms)
- Memory usage and KV cache statistics

**MoE Post-Fetch Relevance:** Essential for measuring the performance impact of Post-Fetch optimization. Use to compare baseline vs. optimized inference speeds for MoE models.

---

### llama-batched-bench

**Purpose:** Benchmark tool specifically designed for testing batched decoding performance. Measures how efficiently the system handles multiple parallel sequences.

**Source:** [`tools/batched-bench/batched-bench.cpp`](../tools/batched-bench/batched-bench.cpp)

**Usage Examples:**

```bash
# Basic batched benchmark
./llama-batched-bench -m model.gguf -c 2048 -b 2048 -ub 512

# Test different batch sizes and parallel sequences
./llama-batched-bench -m model.gguf -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32

# With shared prompt processing
./llama-batched-bench -m model.gguf -npp 512 -ntg 128 -npl 8 -pps
```

**Key Options:**
- `-npp` - Number of prompt tokens to process
- `-ntg` - Number of tokens to generate
- `-npl` - Number of parallel sequences (batch size)
- `-pps` - Shared prompt processing mode
- `-ub` - Micro-batch size for decoding

**MoE Post-Fetch Relevance:** Critical for evaluating Post-Fetch under batched inference scenarios, where multiple experts may need to be prefetched simultaneously.

---

### llama-perplexity

**Purpose:** Calculate perplexity and other quality metrics for model evaluation. Useful for comparing model quality across different quantization levels or configurations.

**Source:** [`tools/perplexity/perplexity.cpp`](../tools/perplexity/perplexity.cpp)

**Usage Examples:**

```bash
# Calculate perplexity on a text file
./llama-perplexity -m model.gguf -f test_data.txt

# With specific context window
./llama-perplexity -m model.gguf -f test_data.txt -c 2048

# Compute KL divergence
./llama-perplexity -m model.gguf -f test_data.txt --kl-divergence

# Save logits for analysis
./llama-perplexity -m model.gguf -f test_data.txt --save-logits logits.bin
```

**Output Metrics:**
- Perplexity value
- Log probability statistics
- Entropy measurements
- Token-level analysis

**MoE Post-Fetch Relevance:** Use to verify that Post-Fetch optimization doesn't degrade model quality. Compare perplexity scores between baseline and optimized runs.

---

## Model Processing Tools

### llama-quantize

**Purpose:** Convert and quantize GGUF models to different precision levels. Reduces model size and memory requirements while trading off some quality.

**Source:** [`tools/quantize/quantize.cpp`](../tools/quantize/quantize.cpp)

**Usage Examples:**

```bash
# List available quantization types
./llama-quantize --help

# Quantize to Q4_K_M (recommended balance)
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Quantize to Q5_K_M for better quality
./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M

# Quantize with importance matrix for better quality
./llama-quantize --imatrix imatrix.gguf model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Quantize only specific tensors
./llama-quantize --tensor-type "blk.*ffn_up=Q6_K" model-f16.gguf model-mixed.gguf Q4_K_M
```

**Common Quantization Types:**
| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| Q4_K_M | ~4.5 GB/7B | Good | General use |
| Q5_K_M | ~5.3 GB/7B | Better | Quality-focused |
| Q6_K | ~6.1 GB/7B | Best | Maximum quality |
| Q4_0 | ~4.3 GB/7B | Fair | Speed-focused |
| IQ4_XS | ~4.2 GB/7B | Good | Size-constrained |

**MoE Post-Fetch Relevance:** Quantization affects expert tensor sizes, directly impacting Post-Fetch transfer times. Test different quantization levels to find optimal balance.

---

### llama-imatrix

**Purpose:** Generate importance matrices for improved quantization quality. Analyzes activation patterns to guide quantization decisions.

**Source:** [`tools/imatrix/imatrix.cpp`](../tools/imatrix/imatrix.cpp)

**Usage Examples:**

```bash
# Generate importance matrix from training data
./llama-imatrix -m model.gguf -f training_data.txt -o imatrix.gguf

# With specific chunk size
./llama-imatrix -m model.gguf -f training_data.txt --chunk 512

# Process multiple files
./llama-imatrix -m model.gguf -f data1.txt -f data2.txt -o imatrix.gguf

# Continue from previous imatrix
./llama-imatrix -m model.gguf -f more_data.txt --in-file imatrix-prev.gguf -o imatrix-new.gguf

# Show detailed statistics
./llama-imatrix -m model.gguf -f training_data.txt --show-statistics
```

**Key Options:**
- `-o, --output` - Output file path
- `--chunk` - Chunk size for processing
- `--process-output` - Include output layers
- `--save-frequency` - Save interval during processing
- `--output-frequency` - Print progress interval

**MoE Post-Fetch Relevance:** The imatrix tool demonstrates expert tracking via `GGML_OP_MUL_MAT_ID` operations (see [`tools/imatrix/imatrix.cpp:255-334`](../tools/imatrix/imatrix.cpp:255-334)). This pattern is used in the Expert Usage Tracer implementation.

---

### llama-gguf-split

**Purpose:** Split large GGUF files into smaller chunks or merge split files back together. Useful for distributing large models or fitting models on limited storage.

**Source:** [`tools/gguf-split/gguf-split.cpp`](../tools/gguf-split/gguf-split.cpp)

**Usage Examples:**

```bash
# Split by tensor count
./llama-gguf-split --split --split-max-tensors 128 model.gguf model-split-

# Split by size (e.g., 4GB chunks)
./llama-gguf-split --split --split-max-size 4G model.gguf model-split-

# Merge split files
./llama-gguf-split --merge model-split-00001-of-00004.gguf model-merged.gguf

# Dry run to see split plan
./llama-gguf-split --split --split-max-size 2G --dry-run model.gguf model-split-
```

**Key Options:**
- `--split` - Split mode (default)
- `--merge` - Merge mode
- `--split-max-tensors` - Maximum tensors per split
- `--split-max-size` - Maximum size per split (M/G suffix)
- `--no-tensor-first-split` - Keep first split metadata-only
- `--dry-run` - Show plan without writing files

---

## Utility Tools

### llama-tokenize

**Purpose:** Tokenize text using a model's tokenizer. Useful for debugging tokenization, counting tokens, or preprocessing text.

**Source:** [`tools/tokenize/tokenize.cpp`](../tools/tokenize/tokenize.cpp)

**Usage Examples:**

```bash
# Tokenize a string
./llama-tokenize -m model.gguf -p "Hello, world!"

# Read from file
./llama-tokenize -m model.gguf -f input.txt

# Read from stdin
echo "Hello, world!" | ./llama-tokenize -m model.gguf --stdin

# Output only token IDs (parseable by Python)
./llama-tokenize -m model.gguf -p "Hello, world!" --ids

# Show token count only
./llama-tokenize -m model.gguf -p "Hello, world!" --show-count

# Don't add BOS token
./llama-tokenize -m model.gguf -p "Hello, world!" --no-bos
```

**Key Options:**
- `-p, --prompt` - Input prompt string
- `-f, --file` - Read from file
- `--stdin` - Read from standard input
- `--ids` - Output only token IDs
- `--no-bos` - Don't add BOS token
- `--show-count` - Print total token count

---

### llama-fit-params

**Purpose:** Automatically fit CLI parameters to available GPU memory. Helps find optimal configuration for a given model and hardware.

**Source:** [`tools/fit-params/fit-params.cpp`](../tools/fit-params/fit-params.cpp)

**Usage Examples:**

```bash
# Fit parameters for a model
./llama-fit-params -m model.gguf

# With specific target (VRAM utilization)
./llama-fit-params -m model.gguf --fit-ppl-target 0.9

# With minimum context size constraint
./llama-fit-params -m model.gguf --fit-min-ctx 2048
```

**Output:** Prints optimized CLI arguments like:
```
-c 4096 -ngl 35 -ts 50,50
```

**MoE Post-Fetch Relevance:** Useful for determining optimal GPU layer distribution for MoE models, especially when expert tensors need to be dynamically managed.

---

## Specialized Tools

### llama-tts

**Purpose:** Text-to-speech generation using compatible TTS models (e.g., OuteTTS). Generates audio from text input.

**Source:** [`tools/tts/tts.cpp`](../tools/tts/tts.cpp)

**Usage Examples:**

```bash
# Generate speech from text
./llama-tts -m tts-model.gguf -p "Hello, this is a test."

# Save to WAV file
./llama-tts -m tts-model.gguf -p "Hello, world!" -o output.wav

# With specific speaker
./llama-tts -m tts-model.gguf -p "Hello!" --speaker speaker1.wav
```

---

### llama-mtmd

**Purpose:** Multimodal processing tool for handling image and audio inputs with vision/audio-enabled models.

**Source:** [`tools/mtmd/mtmd.cpp`](../tools/mtmd/mtmd.cpp)

**Usage Examples:**

```bash
# Process image with vision model
./llama-mtmd -m vision-model.gguf --image photo.jpg -p "Describe this image:"

# Process audio
./llama-mtmd -m audio-model.gguf --audio recording.wav -p "Transcribe this:"
```

---

### llama-cvector-generator

**Purpose:** Generate control vectors for steering model behavior. Creates vectors that can modify model outputs in specific directions.

**Source:** [`tools/cvector-generator/cvector-generator.cpp`](../tools/cvector-generator/cvector-generator.cpp)

**Usage Examples:**

```bash
# Generate control vector using PCA method
./llama-cvector-generator -m model.gguf -ngl 99

# With specific PCA parameters
./llama-cvector-generator -m model.gguf --pca-iter 2000 --pca-batch 100

# Using mean method instead of PCA
./llama-cvector-generator -m model.gguf --method mean
```

**Key Options:**
- `--method` - Generation method (pca, mean)
- `--pca-iter` - PCA iterations
- `--pca-batch` - PCA batch size

---

### llama-export-lora

**Purpose:** Export LoRA adapters by merging them with base model weights or converting to standalone format.

**Source:** [`tools/export-lora/export-lora.cpp`](../tools/export-lora/export-lora.cpp)

**Usage Examples:**

```bash
# Export LoRA to GGUF format
./llama-export-lora -m base-model.gguf -l lora-adapter.gguf -o merged.gguf

# With specific scale factor
./llama-export-lora -m base.gguf -l lora.gguf -o merged.gguf --scale 0.8
```

---

### llama-rpc-server

**Purpose:** RPC server for distributed inference across multiple machines. Enables offloading computation to remote backends.

**Source:** [`tools/rpc/rpc-server.cpp`](../tools/rpc/rpc-server.cpp)

**Usage Examples:**

```bash
# Start RPC server on default port
./llama-rpc-server

# With specific host and port
./llama-rpc-server --host 0.0.0.0 --port 50052
```

**MoE Post-Fetch Relevance:** RPC architecture could enable distributed expert storage and retrieval, extending Post-Fetch concepts across network boundaries.

---

## MoE Post-Fetch Relevance

### Primary Tools for MoE Development

| Tool | Relevance | Use Case |
|------|-----------|----------|
| **llama-cli** | High | Interactive testing of MoE models with expert tracing |
| **llama-bench** | Critical | Performance benchmarking before/after Post-Fetch |
| **llama-batched-bench** | Critical | Batched inference performance with Post-Fetch |
| **llama-perplexity** | High | Quality verification after optimization |
| **llama-imatrix** | Reference | Expert tracking pattern implementation |

### Expert Tracing Integration

The Expert Usage Tracer (Phase 0 of MoE Post-Fetch) integrates with these tools via environment variables:

```bash
# Enable expert tracing with any tool
export LLAMA_EXPERT_TRACE_STATS=1
export LLAMA_EXPERT_TRACE_OUTPUT=expert_stats.json

# Run with expert tracing
./llama-cli -m qwen3-next-80b.gguf -p "Hello"
./llama-bench -m qwen3-next-80b.gguf
./llama-perplexity -m qwen3-next-80b.gguf -f test.txt
```

### Benchmarking Post-Fetch Performance

```bash
# Baseline measurement
./llama-bench -m moe-model.gguf -ngl 99 -o json > baseline.json

# With expert tracing (to understand expert usage patterns)
LLAMA_EXPERT_TRACE_STATS=1 ./llama-bench -m moe-model.gguf -ngl 99 -o json > with_trace.json

# Compare results
python compare_benchmarks.py baseline.json with_trace.json
```

### Related Documentation

- **[MoE_PostFetch.md](MoE_PostFetch.md)** - Main design document for Post-Fetch optimization
- **[Debugging_MoE_Experts.md](Debugging_MoE_Experts.md)** - Debug capabilities for expert tracking
- **[MoE_PostFetch_Phase_0.md](MoE_PostFetch_Phase_0.md)** - Implementation guide for Expert Usage Tracer

---

## Build Instructions

For CUDA-enabled builds (recommended for MoE models):

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

Executables are output to `build/bin/`.

For CPU-only builds:

```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

---

## See Also

- [Build Documentation](../docs/build.md) - Detailed build instructions
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [CPP_STYLE_GUIDE.md](CPP_STYLE_GUIDE.md) - C++ coding standards for this project
