#!/usr/bin/env python3
"""
Master benchmark orchestrator.
================================
Runs latency, throughput, and memory benchmarks for all model variants
and generates plots + a comprehensive report.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --models base,awq,gptq
    python scripts/benchmark.py --models base --skip-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import clear_gpu_memory, detect_gpu, setup_logging

logger = setup_logging("benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks for LLM model variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py
  python scripts/benchmark.py --models base,awq,gptq --results-dir results
  python scripts/benchmark.py --models base --skip-plots --skip-report
        """,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="base,awq,gptq",
        help="Comma-separated model variants to benchmark (default: base,awq,gptq)",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Results output directory")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput benchmark")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory benchmark")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    return parser.parse_args()


def get_model_path(config, variant: str) -> Path:
    """Resolve model variant to filesystem path."""
    paths = {
        "base": config.model.base_path,
        "fp16": config.model.base_path,
        "awq": config.model.awq_path,
        "gptq": config.model.gptq_path,
    }
    return paths.get(variant, config.model.base_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    results_dir = args.results_dir or str(config.benchmark.results_path)
    model_variants = [v.strip() for v in args.models.split(",")]

    # GPU check
    gpu = detect_gpu()
    logger.info("=" * 60)
    logger.info("Benchmark Suite")
    logger.info("=" * 60)
    if gpu.available:
        logger.info(f"GPU: {gpu.device_name} ({gpu.total_memory_mb:.0f} MB)")
    else:
        logger.info("Running on CPU")
    logger.info(f"Models: {model_variants}")
    logger.info(f"Results: {results_dir}")

    # ---- Latency + Throughput ----
    # These require loading each model into memory
    if not args.skip_latency or not args.skip_throughput:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for variant in model_variants:
            model_path = get_model_path(config, variant)

            if not model_path.exists():
                logger.warning(f"Model path not found: {model_path} — skipping {variant}")
                continue

            logger.info(f"\nLoading model: {variant} from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto" if gpu.available else "cpu",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

            if not args.skip_latency:
                logger.info(f"\n--- Latency Benchmark: {variant} ---")
                from benchmark.latency import run_latency_benchmark
                run_latency_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=variant,
                    prompt_lengths=config.benchmark.prompt_lengths,
                    generation_lengths=config.benchmark.generation_lengths,
                    num_warmup=config.benchmark.num_warmup,
                    num_iterations=config.benchmark.num_iterations,
                    results_dir=results_dir,
                )

            if not args.skip_throughput:
                logger.info(f"\n--- Throughput Benchmark: {variant} ---")
                from benchmark.throughput import run_throughput_benchmark
                run_throughput_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=variant,
                    prompt_lengths=config.benchmark.prompt_lengths,
                    generation_lengths=config.benchmark.generation_lengths,
                    num_warmup=config.benchmark.num_warmup,
                    num_iterations=config.benchmark.num_iterations,
                    results_dir=results_dir,
                )

            # Free memory before next model
            del model
            clear_gpu_memory()

    # ---- Memory ----
    if not args.skip_memory:
        logger.info("\n--- Memory Benchmark ---")
        from benchmark.memory import run_memory_benchmark

        model_configs = []
        for variant in model_variants:
            path = get_model_path(config, variant)
            if path.exists():
                model_configs.append({"name": variant, "path": str(path)})

        if model_configs:
            run_memory_benchmark(
                model_configs=model_configs,
                prompt_lengths=config.benchmark.prompt_lengths[:3],  # Subset for memory
                results_dir=results_dir,
            )

    # ---- Plots ----
    if not args.skip_plots:
        logger.info("\n--- Generating Plots ---")
        from benchmark.plot import generate_all_plots
        generate_all_plots(results_dir)

    # ---- Report ----
    if not args.skip_report:
        logger.info("\n--- Generating Report ---")
        from benchmark.report import generate_benchmark_report
        generate_benchmark_report(results_dir)

    logger.info("\n" + "=" * 60)
    logger.info("All benchmarks complete ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
