"""
Throughput benchmarking for LLM inference.
===========================================
Measures tokens/sec, prompt processing throughput, and
generation throughput across varying configurations.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.utils import (
    clear_gpu_memory,
    compute_stats,
    save_results_csv,
    save_results_json,
    setup_logging,
)

logger = setup_logging("benchmark.throughput")


def _create_prompt(tokenizer: PreTrainedTokenizerBase, target_length: int) -> str:
    """Create a prompt of approximately the target token length."""
    base_text = "The quick brown fox jumps over the lazy dog. "
    repeated = base_text * (target_length // 8 + 1)
    tokens = tokenizer.encode(repeated, add_special_tokens=False)[:target_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def measure_throughput(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, float]:
    """
    Measure prompt and generation throughput for a single run.

    Returns:
        Dict with prompt_tokens_per_sec, gen_tokens_per_sec,
        total_tokens_per_sec, and timing details.
    """
    prompt_len = input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- Prefill (prompt processing) ---
    prefill_start = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prefill_time = time.perf_counter() - prefill_start

    # --- Full generation (prefill + decoding) ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    gen_start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.perf_counter() - gen_start
    tokens_generated = outputs.shape[1] - prompt_len
    decode_time = max(total_time - prefill_time, 1e-6)

    return {
        "prompt_tokens_per_sec": prompt_len / prefill_time if prefill_time > 0 else 0,
        "gen_tokens_per_sec": tokens_generated / decode_time if decode_time > 0 else 0,
        "total_tokens_per_sec": (prompt_len + tokens_generated) / total_time if total_time > 0 else 0,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "tokens_generated": tokens_generated,
    }


def run_throughput_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str = "model",
    prompt_lengths: list[int] | None = None,
    generation_lengths: list[int] | None = None,
    num_warmup: int = 3,
    num_iterations: int = 10,
    results_dir: str = "results",
) -> list[dict[str, Any]]:
    """
    Run throughput benchmarks across prompt × generation configurations.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        model_name: Name for this model variant.
        prompt_lengths: Prompt token counts to test.
        generation_lengths: Generation token counts to test.
        num_warmup: Warmup iterations.
        num_iterations: Measurement iterations.
        results_dir: Output directory.

    Returns:
        List of result dictionaries.
    """
    prompt_lengths = prompt_lengths or [32, 128, 512, 1024, 2048]
    generation_lengths = generation_lengths or [64, 128, 256, 512]
    device = next(model.parameters()).device
    all_results: list[dict[str, Any]] = []

    logger.info(f"Running throughput benchmark for: {model_name}")

    for prompt_len in prompt_lengths:
        prompt = _create_prompt(tokenizer, prompt_len)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        actual_prompt_len = input_ids.shape[1]

        for gen_len in generation_lengths:
            logger.info(f"  Benchmarking prompt={actual_prompt_len}, gen={gen_len}...")

            prompt_tps_list: list[float] = []
            gen_tps_list: list[float] = []
            total_tps_list: list[float] = []

            # Warmup
            for _ in range(num_warmup):
                measure_throughput(model, input_ids, attention_mask, gen_len)
                clear_gpu_memory()

            # Measure
            for _ in range(num_iterations):
                metrics = measure_throughput(model, input_ids, attention_mask, gen_len)
                prompt_tps_list.append(metrics["prompt_tokens_per_sec"])
                gen_tps_list.append(metrics["gen_tokens_per_sec"])
                total_tps_list.append(metrics["total_tokens_per_sec"])

            result = {
                "model": model_name,
                "prompt_length": actual_prompt_len,
                "generation_length": gen_len,
                "prompt_throughput_mean": round(compute_stats(prompt_tps_list)["mean"], 2),
                "prompt_throughput_std": round(compute_stats(prompt_tps_list)["std"], 2),
                "gen_throughput_mean": round(compute_stats(gen_tps_list)["mean"], 2),
                "gen_throughput_std": round(compute_stats(gen_tps_list)["std"], 2),
                "total_throughput_mean": round(compute_stats(total_tps_list)["mean"], 2),
                "total_throughput_std": round(compute_stats(total_tps_list)["std"], 2),
            }
            all_results.append(result)

            logger.info(
                f"    → Prompt: {result['prompt_throughput_mean']:.0f} tok/s, "
                f"Gen: {result['gen_throughput_mean']:.0f} tok/s, "
                f"Total: {result['total_throughput_mean']:.0f} tok/s"
            )

    csv_path = save_results_csv(all_results, f"{results_dir}/throughput.csv")
    json_path = save_results_json(all_results, f"{results_dir}/throughput.json")
    logger.info(f"Results saved: {csv_path}, {json_path}")

    return all_results
