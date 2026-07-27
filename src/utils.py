"""
Shared utilities for the LLM optimization pipeline.
====================================================
GPU detection, memory tracking, timing, logging, results I/O,
and markdown report generation.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import psutil
import torch

# ============================================================================
# Logging
# ============================================================================
def setup_logging(
    name: str = "llm-opt",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure structured logging with console and optional file output.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# GPU Detection
# ============================================================================
@dataclass
class GPUInfo:
    """Information about available GPU hardware."""
    available: bool
    device_name: str
    device_count: int
    cuda_version: str
    total_memory_mb: float
    free_memory_mb: float
    compute_capability: tuple[int, int] | None


def detect_gpu() -> GPUInfo:
    """
    Detect GPU hardware and return detailed info.

    Returns:
        GPUInfo with hardware details. Falls back gracefully if no GPU.
    """
    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            device_name="CPU",
            device_count=0,
            cuda_version="N/A",
            total_memory_mb=0.0,
            free_memory_mb=0.0,
            compute_capability=None,
        )

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_mem / (1024 ** 2)
    free_mem = torch.cuda.mem_get_info(device)[0] / (1024 ** 2)

    return GPUInfo(
        available=True,
        device_name=props.name,
        device_count=torch.cuda.device_count(),
        cuda_version=torch.version.cuda or "Unknown",
        total_memory_mb=total_mem,
        free_memory_mb=free_mem,
        compute_capability=(props.major, props.minor),
    )


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_dtype(dtype_str: str = "auto") -> torch.dtype:
    """
    Resolve dtype string to torch dtype.

    Args:
        dtype_str: One of 'auto', 'float16', 'bfloat16', 'float32'.

    Returns:
        Appropriate torch dtype. 'auto' selects float16 for CUDA, float32 for CPU.
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if dtype_str == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    return dtype_map.get(dtype_str, torch.float16)


# ============================================================================
# Memory Tracking
# ============================================================================
@dataclass
class MemorySnapshot:
    """Snapshot of current memory usage."""
    timestamp: str
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_mb: float
    cpu_rss_mb: float
    cpu_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_memory_snapshot(label: str = "") -> MemorySnapshot:
    """
    Capture current GPU and CPU memory usage.

    Args:
        label: Optional label for this snapshot.

    Returns:
        MemorySnapshot with current readings.
    """
    process = psutil.Process(os.getpid())

    gpu_allocated = 0.0
    gpu_reserved = 0.0
    gpu_peak = 0.0

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return MemorySnapshot(
        timestamp=datetime.now().isoformat(),
        gpu_allocated_mb=round(gpu_allocated, 2),
        gpu_reserved_mb=round(gpu_reserved, 2),
        gpu_peak_mb=round(gpu_peak, 2),
        cpu_rss_mb=round(process.memory_info().rss / (1024 ** 2), 2),
        cpu_percent=round(process.memory_percent(), 2),
    )


def reset_peak_memory() -> None:
    """Reset the peak GPU memory tracking counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def clear_gpu_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Timing
# ============================================================================
@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    elapsed_seconds: float
    start_time: str
    end_time: str

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_seconds * 1000


@contextmanager
def timer(name: str = "operation") -> Generator[dict[str, Any], None, None]:
    """
    Context manager for timing code blocks.

    Usage:
        with timer("inference") as t:
            output = model.generate(...)
        print(f"Took {t['elapsed_ms']:.2f}ms")
    """
    result: dict[str, Any] = {}
    start = time.perf_counter()
    start_time = datetime.now().isoformat()

    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        end_time = datetime.now().isoformat()
        result.update({
            "name": name,
            "elapsed_seconds": elapsed,
            "elapsed_ms": elapsed * 1000,
            "start_time": start_time,
            "end_time": end_time,
        })


def timed(func):
    """Decorator to time function execution and log the result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("llm-opt")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ============================================================================
# Statistics
# ============================================================================
def compute_stats(values: list[float]) -> dict[str, float]:
    """
    Compute descriptive statistics for a list of values.

    Args:
        values: List of numeric measurements.

    Returns:
        Dict with mean, std, min, max, median, p50, p95, p99.
    """
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# ============================================================================
# Results I/O
# ============================================================================
def save_results_csv(
    data: list[dict[str, Any]],
    filepath: str | Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """
    Save benchmark results to CSV.

    Args:
        data: List of dictionaries to save.
        filepath: Output CSV path.
        fieldnames: Column names. Auto-detected if None.

    Returns:
        Path to saved CSV file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not fieldnames and data:
        fieldnames = list(data[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(data)

    return filepath


def save_results_json(
    data: Any,
    filepath: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save results to JSON.

    Args:
        data: Data to serialize.
        filepath: Output JSON path.
        indent: JSON indentation.

    Returns:
        Path to saved JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

    return filepath


def load_results_csv(filepath: str | Path) -> pd.DataFrame:
    """Load benchmark results from CSV into a DataFrame."""
    return pd.read_csv(filepath)


def load_results_json(filepath: str | Path) -> Any:
    """Load results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Markdown Generation
# ============================================================================
def dataframe_to_markdown(
    df: pd.DataFrame,
    title: str = "",
    float_format: str = ".4f",
) -> str:
    """
    Convert a DataFrame to a formatted markdown table.

    Args:
        df: DataFrame to convert.
        title: Optional table title.
        float_format: Format string for floats.

    Returns:
        Markdown string with table.
    """
    lines: list[str] = []

    if title:
        lines.append(f"### {title}\n")

    # Header
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines.append(headers)
    lines.append(separator)

    # Rows
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:{float_format}}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_comparison_table(
    results: dict[str, dict[str, float]],
    metric_name: str = "Metric",
) -> str:
    """
    Generate a markdown comparison table across model variants.

    Args:
        results: Dict of {model_name: {metric: value}}.
        metric_name: Name for the metric column.

    Returns:
        Markdown table string.
    """
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    lines: list[str] = []
    header = f"| {metric_name} | " + " | ".join(models) + " |"
    separator = "| --- | " + " | ".join("---" for _ in models) + " |"
    lines.append(header)
    lines.append(separator)

    for metric in metrics:
        row_vals = []
        for model in models:
            val = results[model].get(metric, "N/A")
            if isinstance(val, float):
                row_vals.append(f"{val:.4f}")
            else:
                row_vals.append(str(val))
        lines.append(f"| {metric} | " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


# ============================================================================
# Model Size Utilities
# ============================================================================
def get_model_size_mb(model_path: str | Path) -> float:
    """
    Calculate total size of model files on disk.

    Args:
        model_path: Path to model directory.

    Returns:
        Total size in megabytes.
    """
    total = 0
    model_path = Path(model_path)

    if not model_path.exists():
        return 0.0

    for f in model_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size

    return total / (1024 ** 2)


def get_param_count(model: torch.nn.Module) -> dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dict with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


# ============================================================================
# Text Generation Helper
# ============================================================================
def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> tuple[str, float]:
    """
    Generate text with timing.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to use sampling.

    Returns:
        Tuple of (generated_text, elapsed_seconds).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Decode only the generated tokens (exclude prompt)
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, elapsed
