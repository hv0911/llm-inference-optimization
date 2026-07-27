"""
Tests for benchmark utilities and result processing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils import save_results_csv


class TestBenchmarkCSVFormat:
    """Test that benchmark CSV outputs have correct structure."""

    def test_latency_csv_columns(self, tmp_path: Path, sample_csv_data: list[dict]):
        """Verify latency CSV has required columns."""
        filepath = tmp_path / "latency.csv"
        save_results_csv(sample_csv_data, filepath)

        df = pd.read_csv(filepath)
        required_cols = ["model", "prompt_length", "generation_length"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_data_types(self, tmp_path: Path, sample_csv_data: list[dict]):
        """Verify numeric columns are numeric."""
        filepath = tmp_path / "test.csv"
        save_results_csv(sample_csv_data, filepath)

        df = pd.read_csv(filepath)
        assert pd.api.types.is_numeric_dtype(df["total_latency_mean_ms"])
        assert pd.api.types.is_numeric_dtype(df["ttft_mean_ms"])

    def test_multiple_models(self, tmp_path: Path, sample_csv_data: list[dict]):
        """Verify data contains multiple model variants."""
        filepath = tmp_path / "test.csv"
        save_results_csv(sample_csv_data, filepath)

        df = pd.read_csv(filepath)
        models = df["model"].unique()
        assert len(models) >= 2


class TestBenchmarkReportData:
    """Test benchmark report data processing."""

    def test_eval_csv_structure(self, tmp_path: Path, sample_eval_data: list[dict]):
        """Verify evaluation CSV structure."""
        filepath = tmp_path / "evaluation.csv"
        save_results_csv(sample_eval_data, filepath)

        df = pd.read_csv(filepath)
        assert "model" in df.columns
        assert "average" in df.columns

    def test_eval_scores_range(self, sample_eval_data: list[dict]):
        """Verify evaluation scores are in valid range [0, 1]."""
        for result in sample_eval_data:
            for key, value in result.items():
                if key != "model" and isinstance(value, float):
                    assert 0 <= value <= 1, f"Score {key}={value} out of range"

    def test_memory_csv_structure(self, tmp_path: Path):
        """Test memory results CSV structure."""
        data = [
            {
                "model": "base",
                "disk_size_mb": 1200.0,
                "model_gpu_footprint_mb": 1100.0,
                "prompt_length": 128,
                "gpu_allocated_mb": 1150.0,
                "gpu_reserved_mb": 1200.0,
                "gpu_peak_mb": 1250.0,
                "cpu_rss_mb": 800.0,
            },
        ]
        filepath = tmp_path / "memory.csv"
        save_results_csv(data, filepath)

        df = pd.read_csv(filepath)
        assert "gpu_peak_mb" in df.columns
        assert df.iloc[0]["gpu_peak_mb"] > 0
