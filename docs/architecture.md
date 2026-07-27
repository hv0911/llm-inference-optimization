# System Architecture

This document describes the architecture of the production LLM optimization pipeline.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production LLM Optimization                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Download │───▶│ Quantization │───▶│     Evaluation       │  │
│  │  (HF Hub) │    │  AWQ / GPTQ  │    │  (lm-eval-harness)  │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
│       │                │                        │               │
│       ▼                ▼                        ▼               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  models/  │    │   models/    │    │     results/         │  │
│  │  base/    │    │  awq/ gptq/  │    │  evaluation.csv      │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
│       │                │                        │               │
│       └───────┬────────┘                        │               │
│               ▼                                 │               │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │     Benchmarking     │    │         Reporting            │  │
│  │  Latency/Throughput  │───▶│   CSV + Plots + Markdown     │  │
│  │  Memory / TTFT       │    │   BENCHMARK_REPORT.md        │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
│               │                                                 │
│               ▼                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    vLLM Serving                           │  │
│  │  ┌─────────┐  ┌──────────────┐  ┌────────────────────┐  │  │
│  │  │ OpenAI  │  │  Continuous  │  │   Prefix Caching   │  │  │
│  │  │  API    │  │   Batching   │  │   (KV Cache Reuse) │  │  │
│  │  └─────────┘  └──────────────┘  └────────────────────┘  │  │
│  │               PagedAttention (Memory Management)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│               │                                                 │
│               ▼                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │   HuggingFace Hub    │    │    Docker Deployment         │  │
│  │   Model + Card       │    │    docker-compose.yml        │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Model Acquisition

```
HuggingFace Hub ──download──▶ models/base/
                                ├── config.json
                                ├── model.safetensors
                                ├── tokenizer.json
                                └── tokenizer_config.json
```

The `download_model.py` script fetches the model and tokenizer from HuggingFace, auto-detecting GPU availability. On systems with limited VRAM (like 4GB), it uses `float16` to minimize memory footprint.

### 2. Quantization Pipeline

```
models/base/ ──AWQ Pipeline──▶ models/awq/
             ──GPTQ Pipeline──▶ models/gptq/
```

Both quantization pipelines use `llm-compressor`'s `oneshot()` API:

| Step | AWQ | GPTQ |
| --- | --- | --- |
| 1. Load model | `AutoModelForCausalLM` | `AutoModelForCausalLM` |
| 2. Define recipe | `AWQModifier` + `QuantizationModifier` | `GPTQModifier` |
| 3. Calibrate | Forward pass on WikiText-2 | Forward pass on WikiText-2 |
| 4. Quantize | Scale-aware rounding | Hessian-based optimal rounding |
| 5. Save | `compressed-tensors` format | `compressed-tensors` format |

### 3. Evaluation Flow

```
models/{base,awq,gptq}/ ──lm-eval──▶ results/evaluation.csv
```

The `lm-evaluation-harness` evaluates each model variant on 6 NLP benchmarks, producing accuracy scores that quantify the quality impact of quantization.

### 4. Benchmark Flow

```
models/{base,awq,gptq}/ ──benchmark──▶ results/
                                         ├── latency.csv
                                         ├── throughput.csv
                                         ├── memory.csv
                                         ├── plots/
                                         └── BENCHMARK_REPORT.md
```

### 5. Serving Architecture

```
Client (OpenAI SDK) ──HTTP──▶ vLLM Server ──GPU──▶ Model
     │                           │
     │                    ┌──────┴──────┐
     │                    │  Scheduler  │
     │                    │  (Continuous│
     │                    │   Batching) │
     │                    └──────┬──────┘
     │                           │
     │                    ┌──────┴──────┐
     │                    │ KV Cache    │
     │                    │ Manager     │
     │                    │ (Paged      │
     │                    │  Attention) │
     │                    └──────┬──────┘
     │                           │
     │                    ┌──────┴──────┐
     │                    │ Prefix      │
     │                    │ Cache       │
     │                    │ (APC)       │
     │                    └─────────────┘
```

---

## Component Dependencies

```
src/config.py ◀── all scripts (configuration management)
src/utils.py  ◀── all scripts (GPU, memory, timing, I/O)

scripts/download_model.py    ─depends─▶ transformers
scripts/quantize_awq.py      ─depends─▶ llmcompressor, transformers
scripts/quantize_gptq.py     ─depends─▶ llmcompressor, transformers
scripts/benchmark.py         ─depends─▶ benchmark/*, transformers
scripts/evaluate.py          ─depends─▶ evaluation/*, lm_eval
scripts/upload_to_hf.py      ─depends─▶ huggingface_hub

benchmark/latency.py         ─depends─▶ src/utils
benchmark/throughput.py      ─depends─▶ src/utils
benchmark/memory.py          ─depends─▶ src/utils, transformers
benchmark/continuous_batching.py ─depends─▶ openai (async)
benchmark/prefix_caching.py  ─depends─▶ openai (async)
benchmark/plot.py            ─depends─▶ matplotlib, seaborn
benchmark/report.py          ─depends─▶ pandas

evaluation/lm_eval_runner.py ─depends─▶ lm_eval
```

---

## Configuration Architecture

All configuration flows through a centralized YAML-based system:

```
config/default.yaml
       │
       ▼
  src/config.py
       │
       ├── ModelConfig (model paths, dtype)
       ├── QuantizationConfig (AWQ/GPTQ params)
       ├── BenchmarkConfig (lengths, iterations)
       ├── EvaluationConfig (tasks, few-shot)
       ├── ServingConfig (host, port, memory)
       └── UploadConfig (HF credentials)
```

Configuration precedence: **CLI arguments > YAML config > dataclass defaults**

Environment variables are expanded via `${VAR}` syntax in YAML values, with `.env` file support via `python-dotenv`.
