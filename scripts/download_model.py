#!/usr/bin/env python3
"""
Download a HuggingFace model and tokenizer to local storage.
=============================================================
Supports CUDA with automatic fallback to CPU.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model-id Qwen/Qwen3-0.6B --output-dir models/base
    python scripts/download_model.py --dtype float16 --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config
from src.utils import (
    clear_gpu_memory,
    detect_gpu,
    generate_text,
    get_model_size_mb,
    get_param_count,
    get_torch_dtype,
    setup_logging,
    timed,
)

logger = setup_logging("download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model for quantization benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_model.py
  python scripts/download_model.py --model-id Qwen/Qwen3-0.6B --output-dir models/base
  python scripts/download_model.py --dtype float16 --device cpu
        """,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace model ID (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save model (default: from config)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model precision (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to load model on (default: auto)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a quick generation test after download",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    return parser.parse_args()


@timed
def download_model(
    model_id: str,
    output_dir: Path,
    dtype: str = "auto",
    device_map: str = "auto",
) -> tuple:
    """
    Download and save a model and tokenizer from HuggingFace.

    Args:
        model_id: HuggingFace model identifier.
        output_dir: Local save directory.
        dtype: Torch dtype string.
        device_map: Device map strategy.

    Returns:
        Tuple of (model, tokenizer).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = get_torch_dtype(dtype)
    logger.info(f"Downloading model: {model_id}")
    logger.info(f"  dtype: {torch_dtype}")
    logger.info(f"  device_map: {device_map}")
    logger.info(f"  output: {output_dir}")

    # Download tokenizer
    logger.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # Download model
    logger.info("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Save model and tokenizer separately
    logger.info("Saving model...")
    model.save_pretrained(output_dir)

    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def verify_model(model, tokenizer) -> None:
    """Run a quick generation test to verify the downloaded model works."""
    logger.info("Running verification generation...")

    prompt = "The key to efficient LLM inference is"
    generated, elapsed = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7,
    )

    logger.info(f"  Prompt: {prompt}")
    logger.info(f"  Output: {generated[:200]}")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info("Verification PASSED ✓")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Resolve parameters (CLI > config)
    model_id = args.model_id or config.model.model_id
    output_dir = Path(args.output_dir) if args.output_dir else config.model.base_path

    # Resolve device
    if args.device == "auto":
        device_map = "auto"
    elif args.device == "cpu":
        device_map = "cpu"
    else:
        device_map = "auto"

    # GPU info
    gpu_info = detect_gpu()
    logger.info("=" * 60)
    logger.info("GPU Detection")
    logger.info("=" * 60)
    if gpu_info.available:
        logger.info(f"  GPU: {gpu_info.device_name}")
        logger.info(f"  VRAM: {gpu_info.total_memory_mb:.0f} MB")
        logger.info(f"  Free: {gpu_info.free_memory_mb:.0f} MB")
        logger.info(f"  CUDA: {gpu_info.cuda_version}")
        logger.info(f"  Compute: {gpu_info.compute_capability}")
    else:
        logger.warning("  No CUDA GPU detected — using CPU")
        device_map = "cpu"

    # Download
    logger.info("=" * 60)
    logger.info("Downloading Model")
    logger.info("=" * 60)
    model, tokenizer = download_model(
        model_id=model_id,
        output_dir=output_dir,
        dtype=args.dtype,
        device_map=device_map,
    )

    # Report
    logger.info("=" * 60)
    logger.info("Model Summary")
    logger.info("=" * 60)
    params = get_param_count(model)
    size_mb = get_model_size_mb(output_dir)
    logger.info(f"  Parameters: {params['total']:,}")
    logger.info(f"  Disk size: {size_mb:.1f} MB")
    logger.info(f"  Saved to: {output_dir}")

    # Verify
    if args.verify:
        verify_model(model, tokenizer)

    # Cleanup
    clear_gpu_memory()
    logger.info("Download complete ✓")


if __name__ == "__main__":
    main()
