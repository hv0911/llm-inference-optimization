"""
Prefix caching benchmark.
===========================
Demonstrates vLLM's prefix caching (APC - Automatic Prefix Caching)
by comparing TTFT and latency with/without KV cache reuse.

Prefix caching stores computed KV cache blocks for common prompt prefixes
(e.g., system prompts). When multiple requests share the same prefix,
the KV cache is reused instead of recomputed, reducing TTFT significantly.

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

logger = setup_logging("benchmark.prefix_caching")


async def send_chat_request(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict[str, Any]:
    """Send a chat completion request with a system prompt."""
    start = time.perf_counter()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - start
        tokens = response.usage.completion_tokens if response.usage else 0

        return {
            "request_id": request_id,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_generated": tokens,
            "status": "success",
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_generated": 0,
            "status": f"error: {e}",
        }


async def run_prefix_caching_test(
    base_url: str,
    model: str,
    system_prompt: str,
    num_requests: int,
    max_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Run sequential requests with a shared system prompt."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("openai package required: pip install openai")
        return []

    client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
    )

    user_prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does backpropagation work?",
        "What is gradient descent?",
        "Explain the transformer architecture.",
        "What is attention in deep learning?",
        "How does BERT work?",
        "What is transfer learning?",
        "Explain batch normalization.",
        "What is regularization?",
    ]

    results = []
    for i in range(num_requests):
        user_prompt = user_prompts[i % len(user_prompts)]
        result = await send_chat_request(
            client, model, system_prompt, user_prompt, max_tokens, i
        )
        results.append(result)

    await client.close()
    return results


def run_prefix_caching_benchmark(
    base_url: str = "http://localhost:8000",
    model: str = "Qwen/Qwen3-0.6B",
    num_requests: int = 50,
    max_tokens: int = 64,
    results_dir: str = "results",
) -> dict[str, Any]:
    """
    Benchmark prefix caching with a shared system prompt.

    How prefix caching works:
    1. First request: Full KV cache is computed for the system prompt (cold)
    2. Subsequent requests: KV cache blocks for the shared prefix are
       looked up by hash and reused (warm), skipping redundant computation
    3. Only the unique user message tokens need new KV computation

    This benchmark measures the "cold start" (first request) vs "warm" latency
    for subsequent requests that share the same system prompt prefix.

    Args:
        base_url: vLLM server URL.
        model: Model name.
        num_requests: Total requests to send.
        max_tokens: Max tokens per response.
        results_dir: Output directory.

    Returns:
        Comparison results.
    """
    logger.info("=" * 60)
    logger.info("Prefix Caching Benchmark")
    logger.info("=" * 60)

    # Long system prompt that gets cached
    system_prompt = (
        "You are an advanced AI assistant specialized in machine learning and "
        "artificial intelligence. You have deep expertise in neural networks, "
        "deep learning architectures, natural language processing, computer vision, "
        "reinforcement learning, and optimization techniques. When answering questions, "
        "you provide detailed, technically accurate explanations with examples. "
        "You reference relevant research papers and established methodologies. "
        "You explain complex concepts step by step, starting from fundamentals "
        "and building up to advanced topics. Your responses are structured, "
        "clear, and educational. You use analogies to make abstract concepts "
        "more intuitive and accessible to learners at all levels. "
        "Always cite specific architectures, algorithms, and techniques "
        "when discussing implementation details."
    )

    logger.info(f"  System prompt length: ~{len(system_prompt.split())} words")
    logger.info(f"  Requests: {num_requests}")

    # Run the test
    results = asyncio.run(
        run_prefix_caching_test(
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            num_requests=num_requests,
            max_tokens=max_tokens,
        )
    )

    if not results:
        logger.warning("No results — is the vLLM server running?")
        return {}

    successful = [r for r in results if r["status"] == "success"]
    if len(successful) < 2:
        logger.warning("Need at least 2 successful requests for comparison")
        return {}

    # Separate cold (first request) vs warm (subsequent)
    cold_latency = successful[0]["latency_ms"]
    warm_latencies = [r["latency_ms"] for r in successful[1:]]
    warm_stats = compute_stats(warm_latencies)

    comparison = {
        "cold_start_latency_ms": cold_latency,
        "warm_latency_mean_ms": round(warm_stats["mean"], 2),
        "warm_latency_p50_ms": round(warm_stats["p50"], 2),
        "warm_latency_p95_ms": round(warm_stats["p95"], 2),
        "speedup_vs_cold": round(cold_latency / warm_stats["mean"], 2) if warm_stats["mean"] > 0 else 0,
        "total_requests": len(results),
        "successful_requests": len(successful),
    }

    logger.info("\nResults:")
    logger.info(f"  Cold start (1st request): {cold_latency:.0f}ms")
    logger.info(f"  Warm average: {warm_stats['mean']:.0f}ms")
    logger.info(f"  Warm P50: {warm_stats['p50']:.0f}ms")
    logger.info(f"  Warm P95: {warm_stats['p95']:.0f}ms")
    logger.info(f"  Speedup: {comparison['speedup_vs_cold']:.2f}x")

    # Save
    save_results_csv(
        [{"type": "cold", "latency_ms": cold_latency}]
        + [{"type": "warm", "latency_ms": lat} for lat in warm_latencies],
        f"{results_dir}/prefix_caching.csv"
    )
    save_results_json(comparison, f"{results_dir}/prefix_caching.json")
    logger.info(f"Results saved to {results_dir}/prefix_caching.*")

    return comparison
