# vLLM: Architecture & Key Concepts

A technical guide to vLLM's inference engine, covering PagedAttention, continuous batching, prefix caching, and KV cache management.

---

## Table of Contents

1. [What is vLLM?](#what-is-vllm)
2. [PagedAttention](#pagedattention)
3. [Continuous Batching](#continuous-batching)
4. [Prefix Caching (APC)](#prefix-caching)
5. [KV Cache Management](#kv-cache-management)
6. [GPU Memory Optimization](#gpu-memory-optimization)
7. [OpenAI-Compatible API](#openai-compatible-api)

---

## What is vLLM?

vLLM is a high-throughput LLM inference and serving engine. Its key innovations:

- **PagedAttention**: Virtual memory–inspired KV cache management
- **Continuous batching**: Iteration-level request scheduling
- **Prefix caching**: KV cache reuse across requests
- **Optimized kernels**: Marlin (INT4), FlashAttention, FlashInfer

vLLM serves models via an OpenAI-compatible API, making it a drop-in replacement for the OpenAI API endpoint.

---

## PagedAttention

**Paper**: *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023)

### The Problem: KV Cache Fragmentation

In standard transformer inference, each request needs a KV cache that grows linearly with sequence length. With naive memory allocation:

```
Traditional KV Cache Allocation:
┌──────────────────────────────────────────┐
│ Request A: [KV_1][KV_2][KV_3][   waste  ]│  ← Pre-allocated for max_seq_len
├──────────────────────────────────────────┤
│ Request B: [KV_1][KV_2][      waste     ]│  ← Most memory wasted
├──────────────────────────────────────────┤
│ Request C: [KV_1][KV_2][KV_3][KV_4][ w  ]│
└──────────────────────────────────────────┘
  Total GPU memory: Allocated for worst case → massive waste
```

Problems:
- **Internal fragmentation**: Each request pre-allocates for `max_seq_len`
- **External fragmentation**: Variable-length sequences leave gaps
- **Over-reservation**: Typically 60–80% of KV cache memory is wasted

### The Solution: Virtual Memory for KV Caches

PagedAttention borrows the concept of **paging** from operating systems:

```
PagedAttention:
┌─────────────────────────────────────────────┐
│ Physical KV Cache Blocks (GPU Memory)       │
│ [Block 0][Block 1][Block 2][Block 3][Blk 4] │
│ [Block 5][Block 6][Block 7][Block 8][Free ] │
└─────────────────────────────────────────────┘
          ↑ mapped by block tables ↓
┌──────────────────────────────────┐
│ Block Table (per request):       │
│ Request A: [0, 3, 5]            │  ← 3 blocks, non-contiguous
│ Request B: [1, 6]               │  ← 2 blocks
│ Request C: [2, 4, 7, 8]         │  ← 4 blocks
└──────────────────────────────────┘
```

Key ideas:
1. **Fixed-size blocks**: KV cache is divided into fixed-size blocks (e.g., 16 tokens each)
2. **Block table**: Each request has a mapping from logical block index → physical block
3. **On-demand allocation**: Blocks are allocated only as needed during generation
4. **Non-contiguous storage**: A request's KV cache blocks need not be contiguous in GPU memory

### Memory Savings

```
                     Traditional    PagedAttention
Avg utilization:        20-40%         >95%
Max batch size:           4x           10-20x
Throughput:              1x           2-4x higher
```

### How It Works at Inference Time

1. **Prefill**: Process prompt tokens, allocate KV blocks as needed
2. **Decode**: For each new token:
   a. Check if current block has space → append KV
   b. If full → allocate a new block, update block table
3. **Complete**: When request finishes → free all blocks

---

## Continuous Batching

### The Problem: Static Batching

Traditional serving uses **static batching**: collect N requests, process them together, wait for ALL to finish, then start a new batch.

```
Static Batching:
Time ──▶
Request A: [████████████████████]──────────────idle──────────
Request B: [████████]──────────────idle──────────────────────
Request C: [████████████████████████████████]────────────────
                                              ↑ All must wait
                                                for C to finish
```

Problems:
- Short requests wait for long ones (convoy effect)
- GPU is underutilized during the waiting period
- Low throughput

### The Solution: Iteration-Level Scheduling

Continuous batching schedules at the **iteration** (token) level:

```
Continuous Batching:
Time ──▶
Iteration:  1    2    3    4    5    6    7    8    9
Request A: [█]  [█]  [█]  [█]  [█]  DONE
Request B: [█]  [█]  [█]  DONE
Request D:                [█]  [█]  [█]  [█]  DONE    ← Starts immediately
Request C: [█]  [█]  [█]  [█]  [█]  [█]  [█]  [█]  [█]
Request E:           [█]  [█]  [█]  [█]  [█]  [█]  DONE

GPU Util:  100% 100% 100% 100% 100% 100% 100% 100% 100%
```

Key behaviors:
1. When a request completes, its slot is **immediately** filled by a new request
2. No waiting for the entire batch to complete
3. GPU stays busy with maximum parallelism

### Throughput Scaling

As concurrency increases, aggregate throughput scales because:
- More tokens are processed per GPU cycle
- GPU compute units are better utilized
- Memory bandwidth is amortized across more requests

This repo benchmarks this with `benchmark/continuous_batching.py`.

---

## Prefix Caching

**Also called**: Automatic Prefix Caching (APC)

### The Problem

Many requests share common prefixes (system prompts, few-shot examples). Without caching, each request recomputes the KV cache for the shared prefix.

```
Request 1: [System Prompt ████████] [User: "What is ML?"]
Request 2: [System Prompt ████████] [User: "Explain DNNs"]
Request 3: [System Prompt ████████] [User: "What is RL?"]
                ↑ Recomputed 3 times!
```

### The Solution: KV Cache Block Hashing

vLLM hashes the token content of each KV block. When a new request has the same prefix tokens, it reuses the existing KV blocks:

```
Block Hash Table:
  hash("System prompt tokens 1-16")  → Block #42
  hash("System prompt tokens 17-32") → Block #43
  hash("System prompt tokens 33-48") → Block #44

Request 1: Compute blocks [42, 43, 44] + new blocks for user query
Request 2: Reuse  blocks [42, 43, 44] + new blocks (TTFT ↓↓↓)
Request 3: Reuse  blocks [42, 43, 44] + new blocks (TTFT ↓↓↓)
```

### Benefits

- **Reduced TTFT**: Skip recomputation of shared prefix → first token arrives faster
- **Lower GPU compute**: Fewer FLOPs for the prefill phase
- **Higher throughput**: More GPU cycles available for decode

### Enabling Prefix Caching

```bash
# vLLM server
vllm serve models/base --enable-prefix-caching

# This repo's launch script
bash serving/serve_vllm.sh --enable-prefix-caching
```

This repo benchmarks prefix caching with `benchmark/prefix_caching.py`.

---

## KV Cache Management

### What is the KV Cache?

During autoregressive generation, each transformer layer stores **Key** and **Value** tensors for all previously processed tokens. This avoids recomputing attention over the entire sequence at each step.

```
KV Cache size per token per layer:
  = 2 × hidden_dim × sizeof(dtype)
  = 2 × 1024 × 2 bytes (FP16)
  = 4 KB

Total KV cache for Qwen3-0.6B (28 layers, 2048 tokens):
  = 28 × 2048 × 4 KB ≈ 224 MB
```

### Memory Budget

vLLM allocates GPU memory as:

```
Total GPU Memory (4096 MB)
├── Model Weights:    ~1200 MB (FP16) or ~400 MB (INT4)
├── KV Cache Pool:    ~2400 MB (managed by PagedAttention)
├── Activation Memory: ~200 MB (temporary, per-batch)
└── CUDA Overhead:     ~300 MB (driver, kernels)
```

The `--gpu-memory-utilization` flag controls what fraction of GPU memory vLLM uses:

```bash
vllm serve model --gpu-memory-utilization 0.85  # Use 85% of VRAM
```

---

## GPU Memory Optimization

### Strategies Used in This Pipeline

| Strategy | Impact | How |
| --- | --- | --- |
| INT4 Quantization | ~3.5x weight reduction | AWQ/GPTQ |
| PagedAttention | ~4x batch size increase | vLLM built-in |
| Prefix Caching | Reduced KV recomputation | `--enable-prefix-caching` |
| `max_model_len` | Caps KV cache allocation | `--max-model-len 2048` |
| `gpu_memory_utilization` | Controls VRAM budget | `--gpu-memory-utilization 0.85` |

### For 4GB VRAM GPUs

With only 4GB VRAM, memory management is critical:

```
FP16 model (~1.2 GB) → KV cache budget ~2.4 GB → ~10K tokens capacity
INT4 model (~0.4 GB) → KV cache budget ~3.2 GB → ~14K tokens capacity
                        ↑ 33% more KV cache available!
```

---

## OpenAI-Compatible API

vLLM exposes an API that is wire-compatible with OpenAI's API:

```python
from openai import OpenAI

# Just change the base_url — everything else is identical
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Same API as OpenAI
response = client.chat.completions.create(
    model="models/base",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
```

Supported endpoints:
- `POST /v1/completions` — Text completion
- `POST /v1/chat/completions` — Chat completion
- `GET /v1/models` — List available models
- `GET /health` — Health check

This means any application built for OpenAI's API works with vLLM with zero code changes (just swap the URL).
