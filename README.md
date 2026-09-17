<p align="center">
  <h1 align="center">🚀 Production LLM Optimization</h1>
  <p align="center">
    <strong>AWQ & GPTQ Quantization • vLLM Serving • Comprehensive Benchmarking</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#benchmarks">Benchmarks</a> •
    <a href="#quantization">Quantization</a> •
    <a href="#vllm-concepts">vLLM Concepts</a> •
    <a href="#documentation">Docs</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/hv0911/llm-inference-optimization/actions/workflows/ci.yml"><img src="https://github.com/hv0911/llm-inference-optimization/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.1+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/quantization-AWQ%20%7C%20GPTQ-orange.svg" alt="Quantization">
  <img src="https://img.shields.io/badge/serving-vLLM-purple.svg" alt="vLLM">
  <img src="https://img.shields.io/badge/model-Qwen3--0.6B-teal.svg" alt="Model">
</p>

---

A production-quality repository demonstrating modern **LLM inference optimization** techniques. Downloads [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), applies **AWQ** and **GPTQ** quantization via `llm-compressor`, benchmarks performance, evaluates quality, and serves via **vLLM** with an OpenAI-compatible API.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │        HuggingFace Hub              │
                    │      Qwen/Qwen3-0.6B                │
                    └──────────────┬──────────────────────┘
                                   │ download
                                   ▼
                    ┌──────────────────────────┐
                    │      models/base/        │
                    │    (FP16, ~1.2 GB)       │
                    └─────┬──────────┬─────────┘
                          │          │
                ┌─────────▼──┐  ┌───▼───────────┐
                │  AWQ Quant  │  │  GPTQ Quant   │
                │ (W4A16)     │  │  (W4A16)      │
                └─────┬──────┘  └───┬───────────┘
                      │             │
              ┌───────▼──┐   ┌─────▼───────┐
              │models/awq│   │ models/gptq │
              │ (~400 MB)│   │  (~400 MB)  │
              └───┬──────┘   └──┬──────────┘
                  │             │
    ┌─────────────┴─────────────┴──────────────┐
    │                                          │
    ▼                    ▼                     ▼
┌────────┐      ┌──────────────┐      ┌────────────┐
│Benchmark│      │  Evaluation  │      │  vLLM      │
│Suite    │      │  lm-eval     │      │  Serving   │
└────┬───┘      └──────┬───────┘      └─────┬──────┘
     │                 │                     │
     ▼                 ▼                     ▼
  results/         results/           OpenAI API
  ├── latency     evaluation.csv     localhost:8000
  ├── throughput
  ├── memory
  └── plots/
```

---

## Repository Structure

```
llm-inference-optimization/
├── config/
│   └── default.yaml              # Central configuration
├── src/
│   ├── __init__.py
│   ├── config.py                 # Typed dataclass configuration
│   └── utils.py                  # GPU detection, memory, timing, I/O
├── scripts/
│   ├── download_model.py         # Download from HuggingFace
│   ├── quantize_awq.py           # AWQ quantization
│   ├── quantize_gptq.py          # GPTQ quantization
│   ├── benchmark.py              # Master benchmark orchestrator
│   ├── evaluate.py               # lm-evaluation-harness runner
│   └── upload_to_hf.py           # HuggingFace upload + model card
├── benchmark/
│   ├── latency.py                # Latency + TTFT measurement
│   ├── throughput.py             # Tokens/sec measurement
│   ├── memory.py                 # GPU/CPU memory profiling
│   ├── continuous_batching.py    # Concurrent request scaling
│   ├── prefix_caching.py         # KV cache reuse benchmark
│   ├── plot.py                   # Chart generation
│   └── report.py                 # Markdown report generator
├── evaluation/
│   └── lm_eval_runner.py         # lm-eval-harness wrapper
├── serving/
│   ├── serve_vllm.sh             # vLLM launch script
│   ├── openai_client.py          # OpenAI SDK examples
│   ├── curl_examples.sh          # curl API examples
│   └── docker-compose.yml        # Multi-model Docker serving
├── docs/
│   ├── architecture.md           # System architecture
│   ├── quantization.md           # AWQ/GPTQ theory
│   ├── vllm.md                   # PagedAttention, batching, caching
│   └── benchmarking.md           # Methodology & reproducibility
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_config.py            # Configuration tests
│   ├── test_utils.py             # Utility function tests
│   ├── test_benchmark.py         # Benchmark output tests
│   └── test_evaluation.py        # Evaluation tests
├── notebooks/
│   └── benchmark_analysis.py     # Results analysis script
├── Dockerfile                    # Multi-stage Docker build
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata
├── .env.example                  # Environment variable template
├── .gitignore
├── LICENSE                       # Apache 2.0
└── README.md                     # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~5GB disk space for models

### Installation

```bash
# Clone the repository
git clone https://github.com/hv0911/llm-inference-optimization.git
cd llm-inference-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your HuggingFace token
```

### Step-by-Step Pipeline

```bash
# 1. Download the base model
python scripts/download_model.py --verify

# 2. Quantize with AWQ
python scripts/quantize_awq.py

# 3. Quantize with GPTQ
python scripts/quantize_gptq.py

# 4. Benchmark all variants
python scripts/benchmark.py --models base,awq,gptq

# 5. Evaluate quality
python scripts/evaluate.py --models base,awq,gptq

# 6. Serve with vLLM
bash serving/serve_vllm.sh --variant awq

# 7. Test the API
python serving/openai_client.py
```

---

## Quantization

This project implements two quantization methods using `llm-compressor`:

### AWQ (Activation-Aware Weight Quantization)

AWQ identifies **salient weight channels** by analyzing activation magnitudes and scales them to preserve precision during INT4 quantization.

```python
# How AWQ works (simplified)
# 1. Observe: Which channels have large activations?
# 2. Scale: Multiply salient weights by s (protects them)
# 3. Compensate: Divide activations by s (exact in FP16)
# 4. Quantize: Apply INT4 with minimal quality loss
```

```bash
python scripts/quantize_awq.py --bits 4 --group-size 128 --num-samples 256
```

### GPTQ (Generative Pre-trained Transformer Quantization)

GPTQ uses **layer-wise Hessian-based optimization** to find the INT4 weights that minimize output reconstruction error.

```python
# How GPTQ works (simplified)
# For each layer:
#   1. Compute H = X^T × X (second-order information)
#   2. For each weight column:
#      a. Quantize to nearest INT4 value
#      b. Compute error from rounding
#      c. Distribute error to remaining columns via H^-1
```

```bash
python scripts/quantize_gptq.py --bits 4 --group-size 128 --num-samples 256
```

### Comparison

| Aspect | AWQ | GPTQ |
| --- | --- | --- |
| Approach | Activation-aware scaling | Hessian-based error correction |
| Accuracy | Generally better | Very competitive |
| Speed | Faster calibration | Slower (Hessian computation) |
| Inference | Same (Marlin kernels) | Same (Marlin kernels) |

> 📖 **Deep dive**: See [docs/quantization.md](docs/quantization.md) for the full theory, mathematical formulation, and implementation details.

---

## Benchmarks

### Metrics Measured

| Category | Metrics |
| --- | --- |
| **Latency** | Total latency, TTFT, per-token latency (P50/P95/P99) |
| **Throughput** | Prompt tok/s, generation tok/s, total tok/s |
| **Memory** | GPU footprint, peak GPU, CPU RSS, disk size |
| **Scaling** | Continuous batching throughput at 1–64 concurrency |
| **Caching** | Prefix cache cold vs warm TTFT |

### Running Benchmarks

```bash
# Full suite
python scripts/benchmark.py --models base,awq,gptq

# Specific benchmarks
python scripts/benchmark.py --models base,awq --skip-memory
python scripts/benchmark.py --models awq --skip-latency --skip-throughput

# Quick test (memory only)
python scripts/benchmark.py --models base --skip-latency --skip-throughput
```

Results are saved to `results/` as CSV, JSON, plots, and a comprehensive Markdown report.

> 📖 **Methodology**: See [docs/benchmarking.md](docs/benchmarking.md) for metric definitions, statistical approach, and reproducibility guide.

---

## Evaluation

Quality is assessed using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on 6 standard NLP benchmarks:

| Benchmark | Category |
| --- | --- |
| HellaSwag | Commonsense reasoning |
| ARC Easy | Science reasoning |
| ARC Challenge | Science reasoning (harder) |
| BoolQ | Boolean question answering |
| PIQA | Physical intuition |
| Winogrande | Coreference resolution |

```bash
# Evaluate all models
python scripts/evaluate.py --models base,awq,gptq

# Quick test (limited examples)
python scripts/evaluate.py --models base --limit 100

# Single task
python scripts/evaluate.py --model-path models/awq --tasks hellaswag
```

---

## vLLM Concepts

### PagedAttention

vLLM uses **PagedAttention** to manage KV caches like virtual memory — fixed-size blocks are allocated on demand and mapped via block tables. This eliminates the 60–80% memory waste of traditional pre-allocation.

```
Traditional: [Request A: ████░░░░░░] [Request B: ██░░░░░░░░]
                         ↑ wasted         ↑ wasted

PagedAttention: [A₁][B₁][A₂][B₂][A₃][Free][Free]
                Non-contiguous blocks, >95% utilization
```

### Continuous Batching

Instead of waiting for all requests to finish (static batching), vLLM schedules at the **iteration level** — when a request completes, a new one immediately takes its slot:

```
Static:      All wait for slowest request → GPU idle time
Continuous:  New requests fill completed slots → GPU always busy
```

### Prefix Caching

When multiple requests share the same prefix (e.g., system prompt), vLLM hashes and reuses the KV cache blocks:

```
Request 1: [Compute system prompt KV] [Compute user query]  ← Cold
Request 2: [Reuse cached KV blocks  ] [Compute user query]  ← Warm (fast!)
```

> 📖 **Deep dive**: See [docs/vllm.md](docs/vllm.md) for detailed explanations with diagrams.

---

## Serving

### Launch vLLM Server

```bash
# Base model
bash serving/serve_vllm.sh --variant base

# AWQ quantized
bash serving/serve_vllm.sh --variant awq

# GPTQ with prefix caching
bash serving/serve_vllm.sh --variant gptq --enable-prefix-caching
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="models/awq",
    messages=[{"role": "user", "content": "What is quantization?"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "models/awq", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Docker

```bash
# Single model
docker compose -f serving/docker-compose.yml up vllm-awq

# All variants
docker compose -f serving/docker-compose.yml up
```

---

## Docker

```bash
# Build full image (quantization + serving)
docker build -t llm-optimization .

# Build lean serving image
docker build --target runtime -t llm-optimization-serve .

# Run
docker run --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  llm-optimization-serve --model /app/models/awq
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=benchmark --cov=evaluation --cov-report=term-missing

# Specific test file
pytest tests/test_config.py -v
```

---

## HuggingFace Upload

```bash
# Upload AWQ model with auto-generated model card
python scripts/upload_to_hf.py \
  --model-path models/awq \
  --variant awq \
  --token $HF_TOKEN

# Upload GPTQ model
python scripts/upload_to_hf.py \
  --model-path models/gptq \
  --variant gptq
```

The upload script automatically generates a comprehensive model card including benchmark results, quantization configuration, evaluation scores, and usage examples.

---

## Configuration

All parameters are centralized in `config/default.yaml`:

```yaml
model:
  model_id: "Qwen/Qwen3-0.6B"
  torch_dtype: "auto"

quantization:
  num_calibration_samples: 256
  max_seq_length: 2048
  awq:
    scheme: "W4A16"
    bits: 4
    group_size: 128

benchmark:
  prompt_lengths: [32, 128, 512, 1024, 2048]
  generation_lengths: [64, 128, 256, 512]
  num_warmup: 3
  num_iterations: 10

serving:
  gpu_memory_utilization: 0.85
  max_model_len: 2048
```

CLI arguments always override config file values.

---

## Hardware Tested

| Component | Specification |
| --- | --- |
| GPU | NVIDIA GeForce (4 GB VRAM) |
| CUDA | 11.6 |
| Driver | 511.65 |
| OS | Windows |

The pipeline is designed to work on limited-VRAM GPUs. All scripts include automatic GPU detection with CPU fallback.

---

## Key Technologies

| Technology | Purpose |
| --- | --- |
| [Transformers](https://github.com/huggingface/transformers) | Model loading and inference |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | AWQ & GPTQ quantization |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput inference serving |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Quality benchmarking |
| [HuggingFace Hub](https://huggingface.co) | Model distribution |

---

## References

1. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
2. **GPTQ**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2022)
3. **PagedAttention**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
4. **vLLM**: Kwon et al., "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" (2023)
5. **llm-compressor**: [github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
6. **Qwen3**: [Qwen Team](https://huggingface.co/Qwen)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---
