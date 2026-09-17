#!/usr/bin/env python3
"""
Upload quantized models to HuggingFace Hub.
=============================================
Automatically generates a model card with benchmark results,
quantization configuration, evaluation scores, and usage examples.

Usage:
    python scripts/upload_to_hf.py --model-path models/awq --repo-id username/model-awq
    python scripts/upload_to_hf.py --model-path models/gptq --variant gptq
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import get_model_size_mb, setup_logging

logger = setup_logging("upload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a quantized model to HuggingFace Hub.",
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model directory to upload")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--variant", type=str, default="awq",
                        choices=["awq", "gptq"],
                        help="Quantization variant")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML path")
    return parser.parse_args()


def generate_model_card(
    model_path: str,
    variant: str,
    base_model_id: str,
    repo_id: str,
    config: dict | None = None,
    results_dir: str = "results",
) -> str:
    """
    Generate a comprehensive HuggingFace model card.

    Args:
        model_path: Path to the quantized model.
        variant: Quantization variant (awq/gptq).
        base_model_id: Original model identifier.
        repo_id: Target HuggingFace repo the model is uploaded to.
        config: Quantization configuration.
        results_dir: Path to benchmark results.

    Returns:
        Model card content as a string.
    """
    variant_upper = variant.upper()
    size_mb = get_model_size_mb(model_path)
    results_path = Path(results_dir)

    # Load evaluation results if available
    eval_table = ""
    eval_csv = results_path / "evaluation.csv"
    if eval_csv.exists():
        import pandas as pd
        df = pd.read_csv(eval_csv)
        variant_row = df[df["model"] == variant]
        if not variant_row.empty:
            eval_lines = ["\n| Benchmark | Score |", "| --- | ---: |"]
            for col in variant_row.columns:
                if col != "model":
                    val = variant_row[col].iloc[0]
                    eval_lines.append(f"| {col} | {val:.4f} |")
            eval_table = "\n".join(eval_lines)

    # Load benchmark summary if available
    benchmark_info = ""
    latency_csv = results_path / "latency.csv"
    if latency_csv.exists():
        import pandas as pd
        df = pd.read_csv(latency_csv)
        variant_data = df[df["model"] == variant]
        if not variant_data.empty:
            sample = variant_data.iloc[0]
            benchmark_info = f"""
### Benchmark Results (prompt_len={int(sample.get('prompt_length', 128))})

| Metric | Value |
| --- | ---: |
| Total Latency | {sample.get('total_latency_mean_ms', 'N/A'):.1f} ms |
| TTFT | {sample.get('ttft_mean_ms', 'N/A'):.1f} ms |
| Per Token | {sample.get('per_token_ms', 'N/A'):.1f} ms |
"""

    card = f"""---
library_name: transformers
license: apache-2.0
base_model: {base_model_id}
tags:
  - quantized
  - {variant}
  - vllm
  - llm-compressor
  - compressed-tensors
  - int4
  - w4a16
model-index:
  - name: {base_model_id.split('/')[-1]}-{variant_upper}
    results: []
---

# {base_model_id.split('/')[-1]} — {variant_upper} Quantized

This is a **{variant_upper} (W4A16)** quantized version of [{base_model_id}](https://huggingface.co/{base_model_id}),
created using [`llm-compressor`](https://github.com/vllm-project/llm-compressor).

## Model Details

| Property | Value |
| --- | --- |
| Base Model | [{base_model_id}](https://huggingface.co/{base_model_id}) |
| Quantization | {variant_upper} (4-bit weights, 16-bit activations) |
| Format | `compressed-tensors` |
| Size | {size_mb:.0f} MB |
| Framework | `llm-compressor` + `transformers` |

## Quantization Configuration

- **Scheme**: W4A16 (4-bit weight-only quantization)
- **Group Size**: 128
- **Targets**: Linear layers
- **Ignored**: `lm_head`
- **Calibration Dataset**: WikiText-2
- **Calibration Samples**: 256

{f"## Evaluation Results{eval_table}" if eval_table else ""}

{benchmark_info}

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "{repo_id}"
)

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With vLLM

```bash
vllm serve {repo_id} \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len 2048
```

### With OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="{repo_id}",
    messages=[{{"role": "user", "content": "Hello!"}}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## Hardware

This model was quantized and tested on:
- **GPU**: NVIDIA GeForce (4GB VRAM)
- **CUDA**: 11.6

## Limitations

- INT4 quantization introduces a small accuracy degradation compared to FP16
- Very long sequences (>2048 tokens) may be limited by available VRAM
- Performance varies by hardware and batch size

## License

This model inherits the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
from the base model.
"""
    return card.strip()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = args.model_path
    variant = args.variant
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        logger.error("HuggingFace token required. Set HF_TOKEN env var or use --token")
        sys.exit(1)

    if not Path(model_path).exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)

    # Resolve repo ID
    base_model = config.model.model_id
    hf_username = config.upload.hf_username
    model_name = base_model.split("/")[-1]
    repo_id = args.repo_id or f"{hf_username}/{model_name}-{variant.upper()}"

    logger.info("=" * 60)
    logger.info("HuggingFace Upload")
    logger.info("=" * 60)
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Repo: {repo_id}")
    logger.info(f"  Variant: {variant}")

    # Generate model card
    model_card = generate_model_card(
        model_path=model_path,
        variant=variant,
        base_model_id=base_model,
        repo_id=repo_id,
        results_dir=str(config.benchmark.results_path),
    )

    # Save model card
    card_path = Path(model_path) / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    logger.info(f"  Model card saved to: {card_path}")

    # Upload
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    logger.info("  Creating repository...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    logger.info("  Uploading model files...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {variant.upper()} quantized model",
    )

    logger.info("=" * 60)
    logger.info(f"Upload complete ✓")
    logger.info(f"  https://huggingface.co/{repo_id}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
