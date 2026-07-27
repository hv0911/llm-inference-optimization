"""
Memory benchmarking for LLM inference.
========================================
Measures GPU memory (allocated, reserved, peak), CPU RAM usage,
and tracks memory throughout the model lifecycle.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    clear_gpu_memory,
    get_memory_snapshot,
    get_model_size_mb,
    get_torch_dtype,
    reset_peak_memory,
    save_results_csv,
    save_results_json,
    setup_logging,
    timer,
)

logger = setup_logging("benchmark.memory")


def measure_model_memory(
    model_path: str,
    model_name: str = "model",
    dtype: str = "auto",
    prompt_lengths: list[int] | None = None,
    generation_length: int = 128,
) -> dict[str, Any]:
    """
    Measure memory usage at each stage of model lifecycle.

    Stages:
    1. Before loading (baseline)
    2. After model load
    3. After inference (various prompt lengths)
    4. Peak memory during generation

    Args:
        model_path: Path to model directory.
        model_name: Human-readable model name.
        dtype: Torch dtype string.
        prompt_lengths: Prompt lengths to test.
        generation_length: Tokens to generate per test.

    Returns:
        Dict with memory measurements at each stage.
    """
    prompt_lengths = prompt_lengths or [32, 128, 512]

    # 1. Baseline
    clear_gpu_memory()
    reset_peak_memory()
    baseline = get_memory_snapshot()

    logger.info(f"Measuring memory for: {model_name}")
    logger.info(f"  Baseline GPU: {baseline.gpu_allocated_mb:.0f} MB")
    logger.info(f"  Baseline CPU: {baseline.cpu_rss_mb:.0f} MB")

    # 2. Load model
    logger.info("  Loading model...")
    torch_dtype = get_torch_dtype(dtype)
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    after_load = get_memory_snapshot()
    logger.info(f"  After load GPU: {after_load.gpu_allocated_mb:.0f} MB")
    logger.info(f"  After load CPU: {after_load.cpu_rss_mb:.0f} MB")

    # 3. Inference at various prompt lengths
    device = next(model.parameters()).device
    inference_snapshots: list[dict[str, Any]] = []

    for prompt_len in prompt_lengths:
        reset_peak_memory()

        # Create prompt
        base_text = "The quick brown fox jumps over the lazy dog. "
        prompt = base_text * (prompt_len // 8 + 1)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        actual_len = input_ids.shape[1]

        # Run inference
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_length,
                do_sample=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        after_inference = get_memory_snapshot()

        snapshot = {
            "prompt_length": actual_len,
            "gpu_allocated_mb": after_inference.gpu_allocated_mb,
            "gpu_reserved_mb": after_inference.gpu_reserved_mb,
            "gpu_peak_mb": after_inference.gpu_peak_mb,
            "cpu_rss_mb": after_inference.cpu_rss_mb,
        }
        inference_snapshots.append(snapshot)

        logger.info(
            f"  Prompt={actual_len}: GPU alloc={after_inference.gpu_allocated_mb:.0f}MB, "
            f"peak={after_inference.gpu_peak_mb:.0f}MB, CPU={after_inference.cpu_rss_mb:.0f}MB"
        )

    # 4. Disk size
    disk_size = get_model_size_mb(model_path)

    # Cleanup
    del model
    clear_gpu_memory()

    return {
        "model_name": model_name,
        "disk_size_mb": round(disk_size, 2),
        "baseline_gpu_mb": baseline.gpu_allocated_mb,
        "baseline_cpu_mb": baseline.cpu_rss_mb,
        "model_load_gpu_mb": after_load.gpu_allocated_mb,
        "model_load_cpu_mb": after_load.cpu_rss_mb,
        "model_gpu_footprint_mb": round(
            after_load.gpu_allocated_mb - baseline.gpu_allocated_mb, 2
        ),
        "inference_snapshots": inference_snapshots,
    }


def run_memory_benchmark(
    model_configs: list[dict[str, str]],
    prompt_lengths: list[int] | None = None,
    generation_length: int = 128,
    results_dir: str = "results",
) -> list[dict[str, Any]]:
    """
    Run memory benchmarks for multiple model variants.

    Args:
        model_configs: List of dicts with 'name' and 'path' keys.
        prompt_lengths: Prompt lengths to test.
        generation_length: Tokens to generate.
        results_dir: Output directory.

    Returns:
        List of memory measurement results.
    """
    all_results: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []

    for config in model_configs:
        name = config["name"]
        path = config["path"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Memory Benchmark: {name}")
        logger.info(f"{'='*60}")

        result = measure_model_memory(
            model_path=path,
            model_name=name,
            prompt_lengths=prompt_lengths,
            generation_length=generation_length,
        )
        all_results.append(result)

        # Flatten for CSV
        for snap in result["inference_snapshots"]:
            flat_results.append({
                "model": name,
                "disk_size_mb": result["disk_size_mb"],
                "model_gpu_footprint_mb": result["model_gpu_footprint_mb"],
                "model_load_cpu_mb": result["model_load_cpu_mb"],
                "prompt_length": snap["prompt_length"],
                "gpu_allocated_mb": snap["gpu_allocated_mb"],
                "gpu_reserved_mb": snap["gpu_reserved_mb"],
                "gpu_peak_mb": snap["gpu_peak_mb"],
                "cpu_rss_mb": snap["cpu_rss_mb"],
            })

    # Save
    save_results_csv(flat_results, f"{results_dir}/memory.csv")
    save_results_json(all_results, f"{results_dir}/memory.json")
    logger.info(f"Memory results saved to {results_dir}/")

    return all_results
