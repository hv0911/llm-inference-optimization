"""
Latency benchmarking for LLM inference.
========================================
Measures total latency, Time-To-First-Token (TTFT), and per-token latency
across configurable prompt and generation lengths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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

logger = setup_logging("benchmark.latency")


@dataclass
class LatencyResult:
    """Result of a single latency measurement."""
    model_name: str
    prompt_length: int
    generation_length: int
    total_latency_ms: float
    ttft_ms: float
    per_token_ms: float
    tokens_generated: int


def _create_prompt(tokenizer: PreTrainedTokenizerBase, target_length: int) -> str:
    """Create a prompt of approximately the target token length."""
    # Use a repeating pattern to fill the prompt
    base_text = "The quick brown fox jumps over the lazy dog. "
    repeated = base_text * (target_length // 8 + 1)

    # Tokenize and truncate to exact length
    tokens = tokenizer.encode(repeated, add_special_tokens=False)[:target_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def measure_ttft(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """
    Measure Time-To-First-Token by running a single forward pass.

    This simulates the prefill phase — processing the entire prompt
    to produce the first output token.

    Returns:
        TTFT in milliseconds.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000


def measure_generation_latency(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[float, float, int]:
    """
    Measure total generation latency and TTFT.

    Returns:
        Tuple of (total_latency_ms, ttft_ms, tokens_generated).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure TTFT (prefill)
    ttft_ms = measure_ttft(model, input_ids, attention_mask)

    # Measure total generation (prefill + decode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_ms = (time.perf_counter() - start) * 1000
    tokens_generated = outputs.shape[1] - input_ids.shape[1]

    return total_ms, ttft_ms, tokens_generated


def run_latency_benchmark(
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
    Run comprehensive latency benchmarks.

    Measures latency across a matrix of prompt × generation lengths.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        model_name: Name for this model variant (e.g., "fp16", "awq").
        prompt_lengths: List of prompt token counts to test.
        generation_lengths: List of generation token counts to test.
        num_warmup: Warmup iterations (discarded).
        num_iterations: Measurement iterations.
        results_dir: Output directory.

    Returns:
        List of result dictionaries.
    """
    prompt_lengths = prompt_lengths or [32, 128, 512, 1024, 2048]
    generation_lengths = generation_lengths or [64, 128, 256, 512]
    device = next(model.parameters()).device
    all_results: list[dict[str, Any]] = []

    logger.info(f"Running latency benchmark for: {model_name}")
    logger.info(f"  Prompt lengths: {prompt_lengths}")
    logger.info(f"  Generation lengths: {generation_lengths}")
    logger.info(f"  Warmup: {num_warmup}, Iterations: {num_iterations}")

    for prompt_len in prompt_lengths:
        # Create prompt of target length
        prompt = _create_prompt(tokenizer, prompt_len)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        actual_prompt_len = input_ids.shape[1]

        for gen_len in generation_lengths:
            logger.info(f"  Benchmarking prompt={actual_prompt_len}, gen={gen_len}...")

            latencies_ms: list[float] = []
            ttfts_ms: list[float] = []
            tokens_list: list[int] = []

            # Warmup
            for _ in range(num_warmup):
                measure_generation_latency(model, input_ids, attention_mask, gen_len)
                clear_gpu_memory()

            # Measure
            for i in range(num_iterations):
                total_ms, ttft_ms, tokens = measure_generation_latency(
                    model, input_ids, attention_mask, gen_len
                )
                latencies_ms.append(total_ms)
                ttfts_ms.append(ttft_ms)
                tokens_list.append(tokens)

            # Compute stats
            latency_stats = compute_stats(latencies_ms)
            ttft_stats = compute_stats(ttfts_ms)
            avg_tokens = sum(tokens_list) / len(tokens_list)
            per_token = latency_stats["mean"] / avg_tokens if avg_tokens > 0 else 0

            result = {
                "model": model_name,
                "prompt_length": actual_prompt_len,
                "generation_length": gen_len,
                "tokens_generated": avg_tokens,
                "total_latency_mean_ms": round(latency_stats["mean"], 2),
                "total_latency_std_ms": round(latency_stats["std"], 2),
                "total_latency_p50_ms": round(latency_stats["p50"], 2),
                "total_latency_p95_ms": round(latency_stats["p95"], 2),
                "total_latency_p99_ms": round(latency_stats["p99"], 2),
                "ttft_mean_ms": round(ttft_stats["mean"], 2),
                "ttft_std_ms": round(ttft_stats["std"], 2),
                "ttft_p50_ms": round(ttft_stats["p50"], 2),
                "ttft_p95_ms": round(ttft_stats["p95"], 2),
                "per_token_ms": round(per_token, 2),
            }
            all_results.append(result)

            logger.info(
                f"    → Latency: {latency_stats['mean']:.1f}ms ± {latency_stats['std']:.1f}ms, "
                f"TTFT: {ttft_stats['mean']:.1f}ms, Per-token: {per_token:.1f}ms"
            )

    # Save results
    csv_path = save_results_csv(all_results, f"{results_dir}/latency.csv")
    json_path = save_results_json(all_results, f"{results_dir}/latency.json")
    logger.info(f"Results saved: {csv_path}, {json_path}")

    return all_results
