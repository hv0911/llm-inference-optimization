#!/usr/bin/env python3
"""
Evaluation CLI — run lm-evaluation-harness benchmarks.
=======================================================
Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --models base,awq,gptq
    python scripts/evaluate.py --tasks hellaswag,arc_challenge --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import detect_gpu, setup_logging

logger = setup_logging("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model variants using lm-evaluation-harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py
  python scripts/evaluate.py --models base,awq,gptq
  python scripts/evaluate.py --model-path models/base --tasks hellaswag,boolq
  python scripts/evaluate.py --models base --limit 100  # Quick test
        """,
    )
    parser.add_argument("--models", type=str, default="base,awq,gptq",
                        help="Comma-separated model variants")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Direct path to a single model (overrides --models)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks (default: all)")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Few-shot examples (default: 0)")
    parser.add_argument("--batch-size", type=str, default=None,
                        help="Batch size (default: auto)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda:0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (for quick testing)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Resolve parameters
    tasks = args.tasks.split(",") if args.tasks else config.evaluation.tasks
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else config.evaluation.num_fewshot
    batch_size = args.batch_size or config.evaluation.batch_size
    device = args.device or config.evaluation.device
    results_dir = args.results_dir or str(config.benchmark.results_path)

    # GPU info
    gpu = detect_gpu()
    if gpu.available:
        logger.info(f"GPU: {gpu.device_name} ({gpu.total_memory_mb:.0f} MB)")
    else:
        logger.info("No GPU — running on CPU")
        device = "cpu"

    from evaluation.lm_eval_runner import evaluate_all_models

    if args.model_path:
        # Single model evaluation
        model_configs = [{"name": "model", "path": args.model_path}]
    else:
        # Multiple model variants
        variants = [v.strip() for v in args.models.split(",")]
        path_map = {
            "base": config.model.base_path,
            "fp16": config.model.base_path,
            "awq": config.model.awq_path,
            "gptq": config.model.gptq_path,
        }
        model_configs = [
            {"name": v, "path": str(path_map.get(v, config.model.base_path))}
            for v in variants
        ]

    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Models: {[c['name'] for c in model_configs]}")
    logger.info(f"Tasks: {tasks}")

    df = evaluate_all_models(
        model_configs=model_configs,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        results_dir=results_dir,
        limit=args.limit,
    )

    if not df.empty:
        logger.info("\nEvaluation complete ✓")
    else:
        logger.warning("No evaluation results produced")


if __name__ == "__main__":
    main()
