"""
Tests for utility functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    compute_stats,
    dataframe_to_markdown,
    generate_comparison_table,
    get_model_size_mb,
    get_torch_dtype,
    load_results_csv,
    load_results_json,
    save_results_csv,
    save_results_json,
    setup_logging,
    timer,
)


class TestLogging:
    """Test logging setup."""

    def test_setup_logging(self):
        logger = setup_logging("test-logger")
        assert logger.name == "test-logger"

    def test_logging_with_file(self, tmp_path: Path):
        log_file = str(tmp_path / "test.log")
        logger = setup_logging("test-file-logger", log_file=log_file)
        logger.info("test message")
        assert Path(log_file).exists()


class TestGPUUtils:
    """Test GPU utility functions."""

    def test_get_torch_dtype_auto(self):
        import torch
        dtype = get_torch_dtype("auto")
        assert dtype in (torch.float16, torch.float32)

    def test_get_torch_dtype_explicit(self):
        import torch
        assert get_torch_dtype("float16") == torch.float16
        assert get_torch_dtype("float32") == torch.float32
        assert get_torch_dtype("bfloat16") == torch.bfloat16

    def test_get_torch_dtype_aliases(self):
        import torch
        assert get_torch_dtype("fp16") == torch.float16
        assert get_torch_dtype("fp32") == torch.float32
        assert get_torch_dtype("bf16") == torch.bfloat16


class TestTimer:
    """Test timing utilities."""

    def test_timer_context_manager(self):
        import time

        with timer("test-op") as t:
            time.sleep(0.05)

        assert t["name"] == "test-op"
        assert t["elapsed_seconds"] >= 0.04  # Allow slight timing variance
        assert t["elapsed_ms"] >= 40
        assert "start_time" in t
        assert "end_time" in t


class TestStatistics:
    """Test statistical computation."""

    def test_compute_stats_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_stats(values)

        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
        assert stats["p50"] == 3.0
        assert "p95" in stats
        assert "p99" in stats
        assert "std" in stats

    def test_compute_stats_single_value(self):
        stats = compute_stats([42.0])
        assert stats["mean"] == 42.0
        assert stats["std"] == 0.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0


class TestResultsIO:
    """Test CSV and JSON I/O."""

    def test_save_load_csv(self, tmp_path: Path, sample_csv_data: list[dict]):
        filepath = tmp_path / "test_results.csv"
        saved = save_results_csv(sample_csv_data, filepath)

        assert saved.exists()
        df = load_results_csv(filepath)
        assert len(df) == 3
        assert "model" in df.columns
        assert df.iloc[0]["model"] == "base"

    def test_save_load_json(self, tmp_path: Path):
        data = {"metric": 42.5, "name": "test"}
        filepath = tmp_path / "test_results.json"
        saved = save_results_json(data, filepath)

        assert saved.exists()
        loaded = load_results_json(filepath)
        assert loaded["metric"] == 42.5
        assert loaded["name"] == "test"

    def test_csv_creates_parent_dirs(self, tmp_path: Path):
        filepath = tmp_path / "nested" / "dir" / "results.csv"
        save_results_csv([{"a": 1}], filepath)
        assert filepath.exists()

    def test_json_creates_parent_dirs(self, tmp_path: Path):
        filepath = tmp_path / "nested" / "dir" / "results.json"
        save_results_json({"a": 1}, filepath)
        assert filepath.exists()


class TestMarkdownGeneration:
    """Test markdown table generation."""

    def test_dataframe_to_markdown(self):
        df = pd.DataFrame({
            "Model": ["base", "awq"],
            "Latency": [150.5, 120.8],
        })
        md = dataframe_to_markdown(df, title="Test Table")

        assert "### Test Table" in md
        assert "| Model |" in md
        assert "base" in md
        assert "awq" in md

    def test_comparison_table(self):
        results = {
            "base": {"hellaswag": 0.35, "arc": 0.52},
            "awq": {"hellaswag": 0.34, "arc": 0.51},
        }
        md = generate_comparison_table(results, metric_name="Task")

        assert "| Task |" in md
        assert "base" in md
        assert "awq" in md
        assert "hellaswag" in md


class TestModelSize:
    """Test model size calculation."""

    def test_model_size_nonexistent(self):
        size = get_model_size_mb("/nonexistent/path")
        assert size == 0.0

    def test_model_size_calculation(self, tmp_path: Path):
        # Create some test files
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1024 * 1024)  # 1 MB
        (tmp_path / "config.json").write_text("{}")

        size = get_model_size_mb(tmp_path)
        assert size >= 1.0  # At least 1 MB
