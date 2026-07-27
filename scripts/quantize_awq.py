#!/usr/bin/env python3
"""
AWQ Quantization using llm-compressor.
=======================================
Applies Activation-Aware Weight Quantization (AWQ) via the
llm-compressor library's AWQModifier + QuantizationModifier pipeline.

AWQ identifies salient weight channels based on activation magnitudes
and scales them to reduce quantization error — producing higher quality
INT4 models compared to naive rounding.

Usage:
    python scripts/quantize_awq.py
    python scripts/quantize_awq.py --model-path models/base --output-dir models/awq
    python scripts/quantize_awq.py --bits 4 --group-size 128 --num-samples 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import (
    clear_gpu_memory,
    detect_gpu,
    get_memory_snapshot,
    get_model_size_mb,
    reset_peak_memory,
    setup_logging,
    timer,
)

logger = setup_logging("quantize-awq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a model using AWQ (Activation-Aware Weight Quantization).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quantize_awq.py
  python scripts/quantize_awq.py --model-path models/base --output-dir models/awq
  python scripts/quantize_awq.py --bits 4 --group-size 128 --num-samples 512 --seq-length 4096
        """,
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to base model")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=None, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=None, help="Quantization group size (default: 128)")
    parser.add_argument("--num-samples", type=int, default=None, help="Calibration samples (default: 256)")
    parser.add_argument("--seq-length", type=int, default=None, help="Max sequence length (default: 2048)")
    parser.add_argument("--dataset", type=str, default=None, help="Calibration dataset")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset config name")
    parser.add_argument("--scheme", type=str, default=None, help="Quantization scheme (default: W4A16)")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    return parser.parse_args()


def quantize_awq(
    model_path: str,
    output_dir: str,
    scheme: str = "W4A16",
    targets: str = "Linear",
    ignore: list[str] | None = None,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_calibration_samples: int = 256,
    max_seq_length: int = 2048,
) -> Path:
    """
    Apply AWQ quantization using llm-compressor.

    The AWQ pipeline consists of two modifiers:
    1. AWQModifier — computes activation-aware scaling factors
    2. QuantizationModifier — applies the actual weight quantization

    Args:
        model_path: Path to the base model directory or HF model ID.
        output_dir: Directory to save the quantized model.
        scheme: Quantization scheme (e.g., "W4A16").
        targets: Layer types to quantize.
        ignore: Layer names to skip.
        dataset: Calibration dataset name.
        dataset_config: Dataset configuration.
        num_calibration_samples: Number of calibration examples.
        max_seq_length: Maximum sequence length for calibration.

    Returns:
        Path to the saved quantized model.
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ignore = ignore or ["lm_head"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track memory before
    reset_peak_memory()
    mem_before = get_memory_snapshot()
    logger.info(f"Memory before quantization: GPU={mem_before.gpu_allocated_mb:.0f}MB, CPU={mem_before.cpu_rss_mb:.0f}MB")

    # Load base model
    logger.info(f"Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    mem_loaded = get_memory_snapshot()
    logger.info(f"Memory after model load: GPU={mem_loaded.gpu_allocated_mb:.0f}MB")

    # Define AWQ recipe
    # Step 1: AWQModifier identifies salient channels via activation analysis
    # Step 2: QuantizationModifier applies INT4 quantization with group-wise scaling
    recipe = [
        AWQModifier(),
        QuantizationModifier(
            targets=targets,
            scheme=scheme,
            ignore=ignore,
        ),
    ]

    logger.info("AWQ Recipe:")
    logger.info(f"  Scheme: {scheme}")
    logger.info(f"  Targets: {targets}")
    logger.info(f"  Ignore: {ignore}")
    logger.info(f"  Dataset: {dataset} ({dataset_config})")
    logger.info(f"  Calibration samples: {num_calibration_samples}")
    logger.info(f"  Max seq length: {max_seq_length}")

    # Run one-shot quantization
    logger.info("Starting AWQ quantization (this may take several minutes)...")

    with timer("awq_quantization") as t:
        oneshot(
            model=model,
            dataset=dataset,
            dataset_config_name=dataset_config,
            recipe=recipe,
            output_dir=str(output_path),
            max_seq_length=max_seq_length,
            num_calibration_samples=num_calibration_samples,
        )

    # Save tokenizer alongside quantized model
    tokenizer.save_pretrained(str(output_path))

    # Report results
    mem_after = get_memory_snapshot()
    quant_time = t["elapsed_seconds"]
    base_size = get_model_size_mb(model_path)
    quant_size = get_model_size_mb(output_path)
    compression = base_size / quant_size if quant_size > 0 else 0

    logger.info("=" * 60)
    logger.info("AWQ Quantization Complete")
    logger.info("=" * 60)
    logger.info(f"  Time: {quant_time:.1f}s")
    logger.info(f"  Base model size: {base_size:.1f} MB")
    logger.info(f"  Quantized size: {quant_size:.1f} MB")
    logger.info(f"  Compression ratio: {compression:.2f}x")
    logger.info(f"  Peak GPU memory: {mem_after.gpu_peak_mb:.0f} MB")
    logger.info(f"  Output: {output_path}")

    clear_gpu_memory()
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Resolve parameters (CLI > config > defaults)
    model_path = args.model_path or str(config.model.base_path)
    output_dir = args.output_dir or str(config.model.awq_path)
    scheme = args.scheme or config.quantization.awq.scheme
    num_samples = args.num_samples or config.quantization.num_calibration_samples
    seq_length = args.seq_length or config.quantization.max_seq_length
    dataset = args.dataset or config.quantization.calibration_dataset
    dataset_config = args.dataset_config or config.quantization.calibration_dataset_config

    # GPU check
    gpu = detect_gpu()
    if gpu.available:
        logger.info(f"GPU: {gpu.device_name} ({gpu.total_memory_mb:.0f} MB)")
    else:
        logger.warning("No GPU detected — AWQ quantization requires CUDA!")
        sys.exit(1)

    quantize_awq(
        model_path=model_path,
        output_dir=output_dir,
        scheme=scheme,
        targets=config.quantization.awq.targets,
        ignore=config.quantization.awq.ignore,
        dataset=dataset,
        dataset_config=dataset_config,
        num_calibration_samples=num_samples,
        max_seq_length=seq_length,
    )


if __name__ == "__main__":
    main()
