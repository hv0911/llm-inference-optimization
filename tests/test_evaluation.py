"""
Tests for evaluation utilities.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.lm_eval_runner import DEFAULT_TASKS, generate_eval_markdown
from src.utils import save_results_csv


class TestEvalConfig:
    """Test evaluation configuration."""

    def test_default_tasks(self):
        """Verify default tasks list."""
        assert "hellaswag" in DEFAULT_TASKS
        assert "arc_easy" in DEFAULT_TASKS
        assert "arc_challenge" in DEFAULT_TASKS
        assert "boolq" in DEFAULT_TASKS
        assert "piqa" in DEFAULT_TASKS
        assert "winogrande" in DEFAULT_TASKS

    def test_task_count(self):
        """Verify expected number of default tasks."""
        assert len(DEFAULT_TASKS) == 6


class TestEvalMarkdown:
    """Test evaluation markdown generation."""

    def test_generate_markdown(self, tmp_path: Path, sample_eval_data: list[dict]):
        """Test markdown table generation from eval results."""
        csv_path = tmp_path / "evaluation.csv"
        save_results_csv(sample_eval_data, csv_path)

        md = generate_eval_markdown(csv_path)

        assert "## Evaluation Results" in md
        assert "base" in md
        assert "awq" in md
        assert "hellaswag" in md

    def test_markdown_has_table_structure(self, tmp_path: Path, sample_eval_data: list[dict]):
        """Verify generated markdown has proper table delimiters."""
        csv_path = tmp_path / "evaluation.csv"
        save_results_csv(sample_eval_data, csv_path)

        md = generate_eval_markdown(csv_path)

        lines = md.strip().split("\n")
        # Should have header, separator, and data rows
        table_lines = [l for l in lines if l.startswith("|")]
        assert len(table_lines) >= 3  # header + separator + at least 1 data row
