#!/usr/bin/env python3
"""
OpenAI-compatible client for vLLM server.
===========================================
Demonstrates: basic completion, streaming, chat, and concurrent requests.

Usage:
    python serving/openai_client.py
    python serving/openai_client.py --base-url http://localhost:8000/v1
    python serving/openai_client.py --mode stream
    python serving/openai_client.py --mode concurrent --num-requests 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client examples for vLLM.",
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (auto-detected if not set)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "basic", "chat", "stream", "concurrent"],
                        help="Demo mode to run")
    parser.add_argument("--num-requests", type=int, default=5,
                        help="Number of concurrent requests")
    return parser.parse_args()


def get_model_name(client, model_override: str | None = None) -> str:
    """Auto-detect model name from the server."""
    if model_override:
        return model_override
    models = client.models.list()
    return models.data[0].id if models.data else "unknown"


# ============================================================================
# Demo: Basic Completion
# ============================================================================
def demo_basic_completion(client, model: str) -> None:
    """Basic text completion."""
    print("\n" + "=" * 60)
    print("  Demo: Basic Completion")
    print("=" * 60)

    response = client.completions.create(
        model=model,
        prompt="The future of AI inference optimization is",
        max_tokens=100,
        temperature=0.7,
    )

    print(f"\nPrompt: 'The future of AI inference optimization is'")
    print(f"Output: {response.choices[0].text}")
    print(f"Tokens: {response.usage.completion_tokens}")


# ============================================================================
# Demo: Chat Completion
# ============================================================================
def demo_chat_completion(client, model: str) -> None:
    """Chat-style completion with system prompt."""
    print("\n" + "=" * 60)
    print("  Demo: Chat Completion")
    print("=" * 60)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant specializing in machine learning."
            },
            {
                "role": "user",
                "content": "What is quantization in the context of LLMs?"
            },
        ],
        max_tokens=200,
        temperature=0.7,
    )

    print(f"\nUser: What is quantization in the context of LLMs?")
    print(f"Assistant: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.completion_tokens}")


# ============================================================================
# Demo: Streaming
# ============================================================================
def demo_streaming(client, model: str) -> None:
    """Streaming chat completion."""
    print("\n" + "=" * 60)
    print("  Demo: Streaming Response")
    print("=" * 60)
    print("\nUser: Explain PagedAttention in 3 sentences.")
    print("Assistant: ", end="", flush=True)

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Explain PagedAttention in 3 sentences."},
        ],
        max_tokens=150,
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# ============================================================================
# Demo: Concurrent Requests
# ============================================================================
async def demo_concurrent(base_url: str, model: str, num_requests: int) -> None:
    """Send multiple concurrent requests to demonstrate batching."""
    from openai import AsyncOpenAI

    print("\n" + "=" * 60)
    print(f"  Demo: {num_requests} Concurrent Requests")
    print("=" * 60)

    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")

    prompts = [
        "What is gradient descent?",
        "Explain batch normalization.",
        "What is dropout in neural networks?",
        "How does attention work?",
        "What is transfer learning?",
    ]

    async def send_one(i: int) -> dict:
        prompt = prompts[i % len(prompts)]
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - start
        return {
            "request_id": i,
            "latency_ms": round(elapsed * 1000, 1),
            "tokens": response.usage.completion_tokens,
        }

    start = time.perf_counter()
    results = await asyncio.gather(*[send_one(i) for i in range(num_requests)])
    total = time.perf_counter() - start

    await client.close()

    print(f"\nCompleted {num_requests} requests in {total:.2f}s")
    for r in results:
        print(f"  Request {r['request_id']}: {r['latency_ms']}ms, {r['tokens']} tokens")

    total_tokens = sum(r["tokens"] for r in results)
    print(f"\nAggregate: {total_tokens} tokens in {total:.2f}s = {total_tokens/total:.0f} tok/s")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package required")
        print("  pip install openai")
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # Auto-detect model
    try:
        model = get_model_name(client, args.model)
        print(f"Connected to vLLM server at {args.base_url}")
        print(f"Model: {model}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM server at {args.base_url}")
        print(f"  {e}")
        print(f"\nMake sure vLLM is running:")
        print(f"  bash serving/serve_vllm.sh")
        sys.exit(1)

    modes = [args.mode] if args.mode != "all" else ["basic", "chat", "stream", "concurrent"]

    if "basic" in modes:
        demo_basic_completion(client, model)

    if "chat" in modes:
        demo_chat_completion(client, model)

    if "stream" in modes:
        demo_streaming(client, model)

    if "concurrent" in modes:
        asyncio.run(demo_concurrent(args.base_url, model, args.num_requests))

    print("\n" + "=" * 60)
    print("  All demos complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
