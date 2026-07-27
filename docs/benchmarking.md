# Benchmarking Methodology

This document describes the benchmarking methodology, metrics definitions, and reproducibility guidelines used in this project.

---

## Metrics Definitions

### Latency Metrics

| Metric | Definition | Unit |
| --- | --- | --- |
| **Total Latency** | Wall-clock time from input submission to final token generation | milliseconds |
| **TTFT** (Time to First Token) | Time from input to the first generated token (prefill phase) | milliseconds |
| **Per-Token Latency** | Total latency / number of tokens generated | milliseconds |
| **P50/P95/P99** | Percentile latencies across iterations | milliseconds |

### Throughput Metrics

| Metric | Definition | Unit |
| --- | --- | --- |
| **Prompt Throughput** | Prompt tokens processed / prefill time | tokens/sec |
| **Generation Throughput** | Generated tokens / decode time | tokens/sec |
| **Total Throughput** | All tokens / total time | tokens/sec |
| **Aggregate Throughput** | Total tokens across concurrent requests / wall-clock time | tokens/sec |

### Memory Metrics

| Metric | Definition | Unit |
| --- | --- | --- |
| **GPU Allocated** | Memory actively used by tensors (`torch.cuda.memory_allocated`) | MB |
| **GPU Reserved** | Memory reserved by the CUDA allocator (`torch.cuda.memory_reserved`) | MB |
| **GPU Peak** | Maximum GPU memory allocated during operation | MB |
| **CPU RSS** | Resident Set Size — physical memory used by the process | MB |
| **Disk Size** | Total file size of the model on disk | MB |
| **GPU Footprint** | GPU memory increase from model loading | MB |

---

## Benchmark Configuration

### Test Matrix

Benchmarks test a **matrix** of prompt lengths × generation lengths:

| Prompt Lengths | Generation Lengths |
| --- | --- |
| 32, 128, 512, 1024, 2048 tokens | 64, 128, 256, 512 tokens |

This produces 20 (5×4) configurations per model variant.

### Statistical Rigor

Each configuration is measured with:
- **3 warmup iterations** — Discarded to stabilize GPU clocks, JIT compilation, CUDA context
- **10 measurement iterations** — Used for statistics
- **Greedy decoding** (temperature=0) — Ensures reproducible token sequences
- **CUDA synchronization** — `torch.cuda.synchronize()` before/after timing for accurate GPU measurement

Reported statistics:
- Mean ± standard deviation
- P50 (median), P95, P99 percentiles

### Continuous Batching Test

| Concurrency Levels | Prompt | Max Tokens |
| --- | --- | --- |
| 1, 8, 16, 32, 64 | Fixed technical prompt | 128 |

Uses `asyncio` + OpenAI async SDK to launch concurrent requests.

### Prefix Caching Test

| Parameter | Value |
| --- | --- |
| System prompt | ~150 words (shared across all requests) |
| User prompts | 10 varied technical questions |
| Total requests | 50 |
| Max tokens | 64 |

Measures cold (1st request) vs warm (subsequent) latency.

---

## Reproducibility Guide

### Prerequisites

```bash
# Same hardware & software environment
nvidia-smi  # Verify GPU model and driver
python --version  # Python 3.10+
pip list | grep -E "torch|transformers|vllm|llmcompressor"
```

### Running the Full Benchmark Suite

```bash
# 1. Download model
python scripts/download_model.py --verify

# 2. Quantize
python scripts/quantize_awq.py
python scripts/quantize_gptq.py

# 3. Run benchmarks (all models)
python scripts/benchmark.py --models base,awq,gptq

# 4. Run evaluation
python scripts/evaluate.py --models base,awq,gptq

# 5. Generate plots and report
# (automatically done by benchmark.py, or manually:)
python -c "from benchmark.plot import generate_all_plots; generate_all_plots('results')"
python -c "from benchmark.report import generate_benchmark_report; generate_benchmark_report('results')"
```

### Factors That Affect Results

| Factor | Impact | Mitigation |
| --- | --- | --- |
| GPU temperature | Thermal throttling reduces clocks | Warmup + multiple iterations |
| Background processes | CPU/GPU contention | Close other applications |
| CUDA version | Kernel performance varies | Document exact version |
| Driver version | Affects memory management | Document exact version |
| Power mode | Laptop vs desktop performance | Use maximum performance mode |
| Batch size | Affects throughput per request | Fixed at 1 for latency, varied for batching |

### Interpreting Results

**Latency increases with prompt length** because:
- More tokens to process in the prefill phase
- Larger KV cache to attend over

**Quantized models are faster** because:
- Smaller weights → faster memory transfers
- LLM inference is **memory-bandwidth bound**, not compute-bound
- Quantized models benefit from optimized kernels (Marlin for INT4)

**Throughput scales sub-linearly with concurrency** because:
- GPU compute saturates at some point
- Memory bandwidth becomes the bottleneck
- KV cache grows with concurrent sequences

---

## Output Files

| File | Contents |
| --- | --- |
| `results/latency.csv` | Per-config latency measurements |
| `results/latency.json` | Same data in JSON format |
| `results/throughput.csv` | Per-config throughput measurements |
| `results/memory.csv` | Memory snapshots per model per prompt length |
| `results/memory.json` | Detailed memory data |
| `results/continuous_batching.csv` | Throughput vs concurrency |
| `results/prefix_caching.csv` | Cold vs warm latency |
| `results/evaluation.csv` | NLP benchmark scores |
| `results/plots/` | PNG charts |
| `results/BENCHMARK_REPORT.md` | Auto-generated markdown report |
