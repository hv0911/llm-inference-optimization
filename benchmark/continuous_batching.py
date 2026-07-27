"""
Continuous batching benchmark.
===============================
Demonstrates how vLLM's continuous batching improves GPU utilization
by measuring throughput at varying concurrency levels.

Requires a running vLLM server.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.utils import (
    compute_stats,
    save_results_csv,
    save_results_json,
    setup_logging,
)

logger = setup_logging("benchmark.continuous_batching")


async def send_request(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict[str, Any]:
    """
    Send a single completion request to the vLLM server.

    Args:
        client: AsyncOpenAI client instance.
        model: Model name on the server.
        prompt: Input prompt.
        max_tokens: Max tokens to generate.
        request_id: Request identifier.

    Returns:
        Dict with timing and token count.
    """
    start = time.perf_counter()

    try:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - start
        tokens = response.usage.completion_tokens if response.usage else 0

        return {
            "request_id": request_id,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_generated": tokens,
            "tokens_per_sec": round(tokens / elapsed, 2) if elapsed > 0 else 0,
            "status": "success",
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.warning(f"Request {request_id} failed: {e}")
        return {
            "request_id": request_id,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_generated": 0,
            "tokens_per_sec": 0,
            "status": f"error: {e}",
        }


async def run_concurrent_requests(
    base_url: str,
    model: str,
    num_requests: int,
    max_tokens: int = 128,
    prompt: str = "Explain the concept of attention in transformers.",
) -> list[dict[str, Any]]:
    """
    Launch multiple concurrent requests against a vLLM server.

    Args:
        base_url: vLLM server URL (e.g., http://localhost:8000/v1).
        model: Model identifier.
        num_requests: Number of concurrent requests.
        max_tokens: Max tokens per request.
        prompt: Shared prompt for all requests.

    Returns:
        List of per-request results.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("openai package required: pip install openai")
        return []

    client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
    )

    tasks = [
        send_request(client, model, prompt, max_tokens, i)
        for i in range(num_requests)
    ]

    results = await asyncio.gather(*tasks)
    await client.close()

    return list(results)


def run_continuous_batching_benchmark(
    base_url: str = "http://localhost:8000",
    model: str = "Qwen/Qwen3-0.6B",
    concurrency_levels: list[int] | None = None,
    max_tokens: int = 128,
    results_dir: str = "results",
) -> list[dict[str, Any]]:
    """
    Benchmark continuous batching at various concurrency levels.

    Continuous batching allows vLLM to process new requests while existing
    requests are still generating tokens. This avoids the "convoy effect"
    of static batching where all requests must finish before new ones start.

    At higher concurrency, continuous batching enables:
    - Better GPU utilization (more compute per cycle)
    - Higher aggregate throughput
    - More predictable per-request latency

    Args:
        base_url: vLLM server URL.
        model: Model name.
        concurrency_levels: List of concurrent request counts to test.
        max_tokens: Max tokens per request.
        results_dir: Output directory.

    Returns:
        Aggregated results per concurrency level.
    """
    concurrency_levels = concurrency_levels or [1, 8, 16, 32, 64]
    aggregated_results: list[dict[str, Any]] = []

    logger.info("=" * 60)
    logger.info("Continuous Batching Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Server: {base_url}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Concurrency levels: {concurrency_levels}")

    prompt = (
        "Explain how continuous batching in large language model serving "
        "improves GPU utilization compared to static batching. Include "
        "details about the iteration-level scheduling approach."
    )

    for concurrency in concurrency_levels:
        logger.info(f"\n  Testing concurrency={concurrency}...")

        batch_start = time.perf_counter()
        results = asyncio.run(
            run_concurrent_requests(
                base_url=base_url,
                model=model,
                num_requests=concurrency,
                max_tokens=max_tokens,
                prompt=prompt,
            )
        )
        batch_elapsed = time.perf_counter() - batch_start

        if not results:
            logger.warning(f"  No results for concurrency={concurrency} — is vLLM running?")
            continue

        # Aggregate
        successful = [r for r in results if r["status"] == "success"]
        if not successful:
            continue

        latencies = [r["latency_ms"] for r in successful]
        total_tokens = sum(r["tokens_generated"] for r in successful)
        latency_stats = compute_stats(latencies)

        agg = {
            "concurrency": concurrency,
            "total_requests": len(results),
            "successful_requests": len(successful),
            "total_tokens_generated": total_tokens,
            "batch_elapsed_s": round(batch_elapsed, 2),
            "aggregate_throughput_tps": round(total_tokens / batch_elapsed, 2),
            "avg_latency_ms": round(latency_stats["mean"], 2),
            "p50_latency_ms": round(latency_stats["p50"], 2),
            "p95_latency_ms": round(latency_stats["p95"], 2),
            "p99_latency_ms": round(latency_stats["p99"], 2),
        }
        aggregated_results.append(agg)

        logger.info(
            f"    → Throughput: {agg['aggregate_throughput_tps']:.0f} tok/s, "
            f"Avg latency: {agg['avg_latency_ms']:.0f}ms, "
            f"P95: {agg['p95_latency_ms']:.0f}ms"
        )

    # Save
    if aggregated_results:
        save_results_csv(aggregated_results, f"{results_dir}/continuous_batching.csv")
        save_results_json(aggregated_results, f"{results_dir}/continuous_batching.json")
        logger.info(f"\nResults saved to {results_dir}/continuous_batching.*")

    return aggregated_results
