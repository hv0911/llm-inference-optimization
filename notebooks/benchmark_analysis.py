#!/usr/bin/env python3
"""
Benchmark Analysis Script
===========================
Loads benchmark CSVs, generates comparison tables, creates plots,
and performs statistical analysis.

This script can be converted to a Jupyter notebook using:
    jupytext --to notebook notebooks/benchmark_analysis.py

Usage:
    python notebooks/benchmark_analysis.py
    python notebooks/benchmark_analysis.py --results-dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np


def load_all_results(results_dir: str = "results") -> dict[str, pd.DataFrame]:
    """Load all available benchmark CSVs."""
    results_path = Path(results_dir)
    data = {}

    csv_files = [
        "latency.csv",
        "throughput.csv",
        "memory.csv",
        "continuous_batching.csv",
        "prefix_caching.csv",
        "evaluation.csv",
    ]

    for filename in csv_files:
        filepath = results_path / filename
        if filepath.exists():
            data[filename.replace(".csv", "")] = pd.read_csv(filepath)
            print(f"  ✓ Loaded {filename} ({len(data[filename.replace('.csv', '')])} rows)")
        else:
            print(f"  ✗ Not found: {filename}")

    return data


def analyze_latency(df: pd.DataFrame) -> None:
    """Analyze latency results."""
    print("\n" + "=" * 60)
    print("LATENCY ANALYSIS")
    print("=" * 60)

    # Summary by model
    summary = df.groupby("model").agg({
        "total_latency_mean_ms": ["mean", "std"],
        "ttft_mean_ms": ["mean", "std"],
        "per_token_ms": "mean",
    }).round(2)
    print("\nOverall Summary:")
    print(summary)

    # Speedup analysis
    models = df["model"].unique()
    if "base" in models:
        print("\nSpeedup vs FP16 Baseline:")
        base_data = df[df["model"] == "base"]
        for model in models:
            if model == "base":
                continue
            model_data = df[df["model"] == model]
            if not model_data.empty and not base_data.empty:
                avg_base = base_data["total_latency_mean_ms"].mean()
                avg_model = model_data["total_latency_mean_ms"].mean()
                speedup = avg_base / avg_model if avg_model > 0 else 0
                print(f"  {model}: {speedup:.2f}x faster")


def analyze_throughput(df: pd.DataFrame) -> None:
    """Analyze throughput results."""
    print("\n" + "=" * 60)
    print("THROUGHPUT ANALYSIS")
    print("=" * 60)

    summary = df.groupby("model").agg({
        "prompt_throughput_mean": "mean",
        "gen_throughput_mean": "mean",
        "total_throughput_mean": "mean",
    }).round(2)
    print("\nAverage Throughput (tokens/sec):")
    print(summary)


def analyze_memory(df: pd.DataFrame) -> None:
    """Analyze memory results."""
    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS")
    print("=" * 60)

    summary = df.groupby("model").agg({
        "disk_size_mb": "first",
        "model_gpu_footprint_mb": "first",
        "gpu_peak_mb": "max",
        "cpu_rss_mb": "max",
    }).round(2)
    print("\nMemory Summary:")
    print(summary)

    # Compression ratio
    models = df["model"].unique()
    if "base" in models:
        base_size = df[df["model"] == "base"]["disk_size_mb"].iloc[0]
        print(f"\nCompression Ratios (vs FP16 base={base_size:.0f}MB):")
        for model in models:
            if model == "base":
                continue
            model_size = df[df["model"] == model]["disk_size_mb"].iloc[0]
            ratio = base_size / model_size if model_size > 0 else 0
            print(f"  {model}: {ratio:.2f}x compression ({model_size:.0f}MB)")


def analyze_evaluation(df: pd.DataFrame) -> None:
    """Analyze evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION ANALYSIS")
    print("=" * 60)

    print("\nBenchmark Scores:")
    print(df.to_string(index=False, float_format="%.4f"))

    # Quality degradation
    models = df["model"].values
    if "base" in models:
        base_row = df[df["model"] == "base"].iloc[0]
        print("\nQuality Degradation vs FP16:")
        for _, row in df.iterrows():
            if row["model"] == "base":
                continue
            for col in df.columns:
                if col == "model":
                    continue
                base_val = base_row[col]
                model_val = row[col]
                if isinstance(base_val, (int, float)) and isinstance(model_val, (int, float)):
                    diff = model_val - base_val
                    pct = (diff / base_val * 100) if base_val > 0 else 0
                    direction = "↑" if diff > 0 else "↓"
                    print(f"  {row['model']}/{col}: {diff:+.4f} ({pct:+.2f}%) {direction}")


def analyze_continuous_batching(df: pd.DataFrame) -> None:
    """Analyze continuous batching results."""
    print("\n" + "=" * 60)
    print("CONTINUOUS BATCHING ANALYSIS")
    print("=" * 60)

    print("\nThroughput Scaling:")
    for _, row in df.iterrows():
        print(
            f"  Concurrency {int(row['concurrency']):3d}: "
            f"{row['aggregate_throughput_tps']:8.0f} tok/s, "
            f"avg latency {row['avg_latency_ms']:8.0f}ms"
        )

    # Scaling efficiency
    if len(df) > 1:
        base_throughput = df.iloc[0]["aggregate_throughput_tps"]
        max_throughput = df["aggregate_throughput_tps"].max()
        max_concurrency = df.loc[df["aggregate_throughput_tps"].idxmax(), "concurrency"]
        print(f"\nScaling: {base_throughput:.0f} → {max_throughput:.0f} tok/s "
              f"({max_throughput/base_throughput:.1f}x at concurrency={int(max_concurrency)})")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to results directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  Benchmark Analysis")
    print("=" * 60)
    print(f"\nLoading results from: {args.results_dir}")

    data = load_all_results(args.results_dir)

    if not data:
        print("\nNo benchmark results found. Run benchmarks first:")
        print("  python scripts/benchmark.py")
        return

    if "latency" in data:
        analyze_latency(data["latency"])

    if "throughput" in data:
        analyze_throughput(data["throughput"])

    if "memory" in data:
        analyze_memory(data["memory"])

    if "evaluation" in data:
        analyze_evaluation(data["evaluation"])

    if "continuous_batching" in data:
        analyze_continuous_batching(data["continuous_batching"])

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    try:
        from benchmark.plot import generate_all_plots
        plots = generate_all_plots(args.results_dir)
        print(f"  Generated {len(plots)} plots")
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    print("\n" + "=" * 60)
    print("  Analysis Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
