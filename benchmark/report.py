"""
Benchmark report generator.
=============================
Generates a comprehensive Markdown report from benchmark CSV results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils import detect_gpu, setup_logging

logger = setup_logging("benchmark.report")


def _section_divider() -> str:
    return "\n---\n"


def _format_table(df: pd.DataFrame, float_fmt: str = ".2f") -> str:
    """Convert DataFrame to Markdown table."""
    lines = []
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---:" for _ in df.columns) + " |"
    lines.append(headers)
    lines.append(sep)

    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:{float_fmt}}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_benchmark_report(
    results_dir: str = "results",
    output_path: str | None = None,
) -> Path:
    """
    Generate a comprehensive Markdown benchmark report.

    Reads all available CSV results and produces a formatted report
    with tables, analysis, and plot references.

    Args:
        results_dir: Directory containing benchmark CSVs.
        output_path: Output file path (default: results/BENCHMARK_REPORT.md).

    Returns:
        Path to the generated report.
    """
    results_path = Path(results_dir)
    output = Path(output_path) if output_path else results_path / "BENCHMARK_REPORT.md"

    gpu = detect_gpu()
    report_parts: list[str] = []

    # --- Header ---
    report_parts.append("# 📊 Benchmark Report\n")
    report_parts.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # --- Hardware ---
    report_parts.append("## Hardware Configuration\n")
    if gpu.available:
        report_parts.append(f"- **GPU**: {gpu.device_name}")
        report_parts.append(f"- **VRAM**: {gpu.total_memory_mb:.0f} MB")
        report_parts.append(f"- **CUDA**: {gpu.cuda_version}")
        report_parts.append(f"- **Compute Capability**: {gpu.compute_capability}")
    else:
        report_parts.append("- **Device**: CPU")
    report_parts.append("")

    # --- Latency ---
    latency_csv = results_path / "latency.csv"
    if latency_csv.exists():
        report_parts.append(_section_divider())
        report_parts.append("## Latency Results\n")

        df = pd.read_csv(latency_csv)

        # Summary table (mean latency at gen_len=128)
        summary = df[df["generation_length"] == 128][
            ["model", "prompt_length", "total_latency_mean_ms", "ttft_mean_ms", "per_token_ms"]
        ].copy()
        summary.columns = ["Model", "Prompt Len", "Latency (ms)", "TTFT (ms)", "Per Token (ms)"]
        report_parts.append("### Summary (generation_length=128)\n")
        report_parts.append(_format_table(summary))
        report_parts.append("")

        # Full results
        report_parts.append("\n<details><summary>📋 Full Latency Results</summary>\n")
        report_parts.append(_format_table(df))
        report_parts.append("\n</details>\n")

        report_parts.append("\n![Latency Comparison](plots/latency_comparison.png)\n")
        report_parts.append("![TTFT Comparison](plots/ttft_comparison.png)\n")

    # --- Throughput ---
    throughput_csv = results_path / "throughput.csv"
    if throughput_csv.exists():
        report_parts.append(_section_divider())
        report_parts.append("## Throughput Results\n")

        df = pd.read_csv(throughput_csv)
        summary = df[df["prompt_length"] == 128][
            ["model", "generation_length", "gen_throughput_mean", "total_throughput_mean"]
        ].copy()
        summary.columns = ["Model", "Gen Len", "Gen tok/s", "Total tok/s"]
        report_parts.append("### Summary (prompt_length=128)\n")
        report_parts.append(_format_table(summary))
        report_parts.append("")
        report_parts.append("\n![Throughput Comparison](plots/throughput_comparison.png)\n")

    # --- Memory ---
    memory_csv = results_path / "memory.csv"
    if memory_csv.exists():
        report_parts.append(_section_divider())
        report_parts.append("## Memory Results\n")

        df = pd.read_csv(memory_csv)
        # Model-level summary
        summary = df.groupby("model").agg({
            "disk_size_mb": "first",
            "model_gpu_footprint_mb": "first",
            "gpu_peak_mb": "max",
            "cpu_rss_mb": "max",
        }).reset_index()
        summary.columns = ["Model", "Disk (MB)", "GPU Footprint (MB)", "Peak GPU (MB)", "Peak CPU (MB)"]
        report_parts.append(_format_table(summary))
        report_parts.append("")
        report_parts.append("\n![Memory Comparison](plots/memory_comparison.png)\n")

    # --- Continuous Batching ---
    cb_csv = results_path / "continuous_batching.csv"
    if cb_csv.exists():
        report_parts.append(_section_divider())
        report_parts.append("## Continuous Batching Results\n")
        report_parts.append(
            "Continuous batching allows vLLM to interleave request processing "
            "at the iteration level. As concurrency increases, GPU utilization "
            "improves and aggregate throughput scales near-linearly.\n"
        )
        df = pd.read_csv(cb_csv)
        report_parts.append(_format_table(df[[
            "concurrency", "aggregate_throughput_tps",
            "avg_latency_ms", "p95_latency_ms",
        ]]))
        report_parts.append("")
        report_parts.append("\n![Continuous Batching](plots/continuous_batching.png)\n")

    # --- Prefix Caching ---
    pc_csv = results_path / "prefix_caching.csv"
    if pc_csv.exists():
        report_parts.append(_section_divider())
        report_parts.append("## Prefix Caching Results\n")
        report_parts.append(
            "Prefix caching reuses KV cache blocks for shared prompt prefixes. "
            "The first request (cold) computes the full cache; subsequent requests "
            "(warm) skip the shared prefix computation.\n"
        )
        df = pd.read_csv(pc_csv)
        cold = df[df["type"] == "cold"]["latency_ms"].mean()
        warm = df[df["type"] == "warm"]["latency_ms"].mean()
        report_parts.append(f"- **Cold start**: {cold:.0f}ms")
        report_parts.append(f"- **Warm average**: {warm:.0f}ms")
        if warm > 0:
            report_parts.append(f"- **Speedup**: {cold/warm:.2f}x")
        report_parts.append("")
        report_parts.append("\n![Prefix Caching](plots/prefix_caching.png)\n")

    # --- Evaluation ---
    eval_csv = results_path / "evaluation.csv"
    if eval_csv.exists():
        report_parts.append(_section_divider())
        report_parts.append("## Evaluation Results\n")
        report_parts.append(
            "Evaluation using `lm-evaluation-harness` on standard NLP benchmarks.\n"
        )
        df = pd.read_csv(eval_csv)
        report_parts.append(_format_table(df, float_fmt=".4f"))
        report_parts.append("")

    # --- Footer ---
    report_parts.append(_section_divider())
    report_parts.append("## Methodology\n")
    report_parts.append("- All benchmarks use **greedy decoding** (temperature=0) for reproducibility")
    report_parts.append("- Latency/throughput: 3 warmup + 10 measured iterations per configuration")
    report_parts.append("- Memory: Measured using `torch.cuda.memory_allocated()` and `psutil`")
    report_parts.append("- Continuous batching: Async requests via OpenAI Python SDK")
    report_parts.append("- Prefix caching: Sequential requests with shared system prompt")
    report_parts.append("")

    # Write
    output.parent.mkdir(parents=True, exist_ok=True)
    report_content = "\n".join(report_parts)

    with open(output, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Benchmark report generated: {output}")
    return output
