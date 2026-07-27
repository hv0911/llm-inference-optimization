"""
Publication-quality plot generation from benchmark results.
=============================================================
Generates comparison charts for latency, throughput, memory,
continuous batching scaling, and prefix caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import setup_logging

logger = setup_logging("benchmark.plot")

# ============================================================================
# Style Configuration
# ============================================================================
STYLE_CONFIG = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
}

MODEL_COLORS = {
    "fp16": "#58a6ff",
    "base": "#58a6ff",
    "awq": "#3fb950",
    "gptq": "#d29922",
}

MODEL_LABELS = {
    "fp16": "FP16 (Baseline)",
    "base": "FP16 (Baseline)",
    "awq": "AWQ (W4A16)",
    "gptq": "GPTQ (W4A16)",
}


def _apply_style():
    """Apply the dark publication style."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette([MODEL_COLORS.get("fp16", "#58a6ff"),
                     MODEL_COLORS.get("awq", "#3fb950"),
                     MODEL_COLORS.get("gptq", "#d29922")])


def _save_plot(fig: plt.Figure, filepath: str | Path, dpi: int = 150) -> Path:
    """Save a figure and close it."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {filepath}")
    return filepath


# ============================================================================
# Latency Plots
# ============================================================================
def plot_latency_comparison(
    csv_path: str | Path,
    output_dir: str | Path = "results/plots",
    generation_length: int = 128,
) -> Path:
    """
    Plot latency comparison across models at a fixed generation length.

    Creates a grouped bar chart showing mean latency per prompt length,
    with error bars for standard deviation.
    """
    _apply_style()
    df = pd.read_csv(csv_path)
    df_filtered = df[df["generation_length"] == generation_length]

    fig, ax = plt.subplots(figsize=(12, 6))

    models = df_filtered["model"].unique()
    prompt_lengths = sorted(df_filtered["prompt_length"].unique())
    x = np.arange(len(prompt_lengths))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = df_filtered[df_filtered["model"] == model]
        means = [
            model_data[model_data["prompt_length"] == pl]["total_latency_mean_ms"].values[0]
            for pl in prompt_lengths
        ]
        stds = [
            model_data[model_data["prompt_length"] == pl]["total_latency_std_ms"].values[0]
            for pl in prompt_lengths
        ]
        color = MODEL_COLORS.get(model, "#8b949e")
        label = MODEL_LABELS.get(model, model)

        ax.bar(x + i * width, means, width, yerr=stds,
               label=label, color=color, alpha=0.9,
               capsize=3, error_kw={"linewidth": 1})

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title(f"Inference Latency Comparison (gen_len={generation_length})")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(prompt_lengths)
    ax.legend(framealpha=0.8, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    return _save_plot(fig, Path(output_dir) / "latency_comparison.png")


def plot_ttft_comparison(
    csv_path: str | Path,
    output_dir: str | Path = "results/plots",
) -> Path:
    """Plot TTFT comparison across models and prompt lengths."""
    _apply_style()
    df = pd.read_csv(csv_path)

    # Take one generation length for TTFT (it shouldn't vary much)
    gen_len = df["generation_length"].min()
    df_filtered = df[df["generation_length"] == gen_len]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in df_filtered["model"].unique():
        model_data = df_filtered[df_filtered["model"] == model].sort_values("prompt_length")
        color = MODEL_COLORS.get(model, "#8b949e")
        label = MODEL_LABELS.get(model, model)

        ax.plot(
            model_data["prompt_length"],
            model_data["ttft_mean_ms"],
            marker="o", linewidth=2, markersize=6,
            label=label, color=color,
        )

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Time to First Token (ms)")
    ax.set_title("TTFT vs Prompt Length")
    ax.legend(framealpha=0.8, edgecolor="#30363d")
    ax.grid(alpha=0.3)

    return _save_plot(fig, Path(output_dir) / "ttft_comparison.png")


# ============================================================================
# Throughput Plots
# ============================================================================
def plot_throughput_comparison(
    csv_path: str | Path,
    output_dir: str | Path = "results/plots",
    prompt_length: int = 128,
) -> Path:
    """Plot generation throughput (tokens/sec) across models."""
    _apply_style()
    df = pd.read_csv(csv_path)
    df_filtered = df[df["prompt_length"] == prompt_length]

    fig, ax = plt.subplots(figsize=(10, 6))

    models = df_filtered["model"].unique()
    gen_lengths = sorted(df_filtered["generation_length"].unique())
    x = np.arange(len(gen_lengths))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = df_filtered[df_filtered["model"] == model]
        values = [
            model_data[model_data["generation_length"] == gl]["gen_throughput_mean"].values[0]
            for gl in gen_lengths
        ]
        color = MODEL_COLORS.get(model, "#8b949e")
        label = MODEL_LABELS.get(model, model)

        ax.bar(x + i * width, values, width,
               label=label, color=color, alpha=0.9)

    ax.set_xlabel("Generation Length (tokens)")
    ax.set_ylabel("Generation Throughput (tokens/sec)")
    ax.set_title(f"Generation Throughput Comparison (prompt_len={prompt_length})")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(gen_lengths)
    ax.legend(framealpha=0.8, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    return _save_plot(fig, Path(output_dir) / "throughput_comparison.png")


# ============================================================================
# Memory Plots
# ============================================================================
def plot_memory_comparison(
    csv_path: str | Path,
    output_dir: str | Path = "results/plots",
) -> Path:
    """Plot GPU memory usage comparison across models."""
    _apply_style()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Model footprint (bar chart)
    ax1 = axes[0]
    models = df["model"].unique()
    footprints = [
        df[df["model"] == m]["model_gpu_footprint_mb"].iloc[0] for m in models
    ]
    disk_sizes = [
        df[df["model"] == m]["disk_size_mb"].iloc[0] for m in models
    ]
    colors = [MODEL_COLORS.get(m, "#8b949e") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    x = np.arange(len(models))
    ax1.bar(x - 0.15, footprints, 0.3, label="GPU Footprint", color=colors, alpha=0.9)
    ax1.bar(x + 0.15, disk_sizes, 0.3, label="Disk Size", color=colors, alpha=0.5)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Size (MB)")
    ax1.set_title("Model Size Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.legend(framealpha=0.8, edgecolor="#30363d")
    ax1.grid(axis="y", alpha=0.3)

    # Right: Peak memory during inference
    ax2 = axes[1]
    for model in models:
        model_data = df[df["model"] == model].sort_values("prompt_length")
        color = MODEL_COLORS.get(model, "#8b949e")
        label = MODEL_LABELS.get(model, model)

        ax2.plot(
            model_data["prompt_length"],
            model_data["gpu_peak_mb"],
            marker="s", linewidth=2, markersize=6,
            label=label, color=color,
        )

    ax2.set_xlabel("Prompt Length (tokens)")
    ax2.set_ylabel("Peak GPU Memory (MB)")
    ax2.set_title("Peak GPU Memory vs Prompt Length")
    ax2.legend(framealpha=0.8, edgecolor="#30363d")
    ax2.grid(alpha=0.3)

    fig.suptitle("Memory Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_plot(fig, Path(output_dir) / "memory_comparison.png")


# ============================================================================
# Continuous Batching Plot
# ============================================================================
def plot_continuous_batching(
    csv_path: str | Path,
    output_dir: str | Path = "results/plots",
) -> Path:
    """Plot throughput scaling with concurrency level."""
    _apply_style()
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Aggregate throughput vs concurrency
    ax1.plot(
        df["concurrency"], df["aggregate_throughput_tps"],
        marker="o", linewidth=2.5, color="#58a6ff", markersize=8,
    )
    ax1.fill_between(
        df["concurrency"], df["aggregate_throughput_tps"],
        alpha=0.15, color="#58a6ff",
    )
    ax1.set_xlabel("Concurrent Requests")
    ax1.set_ylabel("Aggregate Throughput (tokens/sec)")
    ax1.set_title("Throughput Scaling with Continuous Batching")
    ax1.grid(alpha=0.3)

    # Right: Latency distribution
    ax2.plot(df["concurrency"], df["avg_latency_ms"],
             marker="o", label="Mean", color="#3fb950", linewidth=2)
    ax2.plot(df["concurrency"], df["p95_latency_ms"],
             marker="^", label="P95", color="#d29922", linewidth=2)
    ax2.plot(df["concurrency"], df["p99_latency_ms"],
             marker="s", label="P99", color="#f85149", linewidth=2)
    ax2.set_xlabel("Concurrent Requests")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency vs Concurrency")
    ax2.legend(framealpha=0.8, edgecolor="#30363d")
    ax2.grid(alpha=0.3)

    fig.suptitle("Continuous Batching Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_plot(fig, Path(output_dir) / "continuous_batching.png")


# ============================================================================
# Prefix Caching Plot
# ============================================================================
def plot_prefix_caching(
    csv_path: str | Path,
    output_dir: str | Path = "results/plots",
) -> Path:
    """Plot cold vs warm latency for prefix caching."""
    _apply_style()
    df = pd.read_csv(csv_path)

    cold = df[df["type"] == "cold"]["latency_ms"].values
    warm = df[df["type"] == "warm"]["latency_ms"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar comparison
    categories = ["Cold Start\n(1st request)", "Warm Average\n(cached prefix)"]
    values = [cold.mean(), warm.mean()]
    colors = ["#f85149", "#3fb950"]

    ax1.bar(categories, values, color=colors, alpha=0.9, width=0.5)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Prefix Caching: Cold vs Warm")
    ax1.grid(axis="y", alpha=0.3)

    # Add speedup annotation
    if warm.mean() > 0:
        speedup = cold.mean() / warm.mean()
        ax1.annotate(
            f"{speedup:.1f}x faster",
            xy=(1, warm.mean()),
            xytext=(1, warm.mean() + (cold.mean() - warm.mean()) * 0.3),
            fontsize=12, fontweight="bold", color="#3fb950",
            ha="center",
        )

    # Right: Warm latency distribution
    ax2.hist(warm, bins=20, color="#3fb950", alpha=0.7, edgecolor="#30363d")
    ax2.axvline(warm.mean(), color="#58a6ff", linestyle="--",
                linewidth=2, label=f"Mean: {warm.mean():.0f}ms")
    ax2.axvline(cold.mean(), color="#f85149", linestyle="--",
                linewidth=2, label=f"Cold: {cold.mean():.0f}ms")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Warm Latency Distribution")
    ax2.legend(framealpha=0.8, edgecolor="#30363d")

    fig.suptitle("Prefix Caching Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_plot(fig, Path(output_dir) / "prefix_caching.png")


# ============================================================================
# Generate All Plots
# ============================================================================
def generate_all_plots(results_dir: str = "results") -> list[Path]:
    """Generate all benchmark plots from CSV results."""
    results_path = Path(results_dir)
    output_dir = results_path / "plots"
    generated: list[Path] = []

    logger.info("Generating benchmark plots...")

    if (results_path / "latency.csv").exists():
        generated.append(plot_latency_comparison(results_path / "latency.csv", output_dir))
        generated.append(plot_ttft_comparison(results_path / "latency.csv", output_dir))

    if (results_path / "throughput.csv").exists():
        generated.append(plot_throughput_comparison(results_path / "throughput.csv", output_dir))

    if (results_path / "memory.csv").exists():
        generated.append(plot_memory_comparison(results_path / "memory.csv", output_dir))

    if (results_path / "continuous_batching.csv").exists():
        generated.append(plot_continuous_batching(results_path / "continuous_batching.csv", output_dir))

    if (results_path / "prefix_caching.csv").exists():
        generated.append(plot_prefix_caching(results_path / "prefix_caching.csv", output_dir))

    logger.info(f"Generated {len(generated)} plots in {output_dir}/")
    return generated
