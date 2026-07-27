"""
Shared test fixtures and configuration.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_csv_data() -> list[dict]:
    """Sample benchmark data for testing I/O."""
    return [
        {
            "model": "base",
            "prompt_length": 128,
            "generation_length": 64,
            "total_latency_mean_ms": 150.5,
            "ttft_mean_ms": 25.3,
            "per_token_ms": 1.95,
        },
        {
            "model": "awq",
            "prompt_length": 128,
            "generation_length": 64,
            "total_latency_mean_ms": 120.8,
            "ttft_mean_ms": 20.1,
            "per_token_ms": 1.57,
        },
        {
            "model": "gptq",
            "prompt_length": 128,
            "generation_length": 64,
            "total_latency_mean_ms": 125.2,
            "ttft_mean_ms": 21.5,
            "per_token_ms": 1.63,
        },
    ]


@pytest.fixture
def sample_eval_data() -> list[dict]:
    """Sample evaluation results."""
    return [
        {"model": "base", "hellaswag": 0.3521, "arc_easy": 0.5234, "average": 0.4378},
        {"model": "awq", "hellaswag": 0.3480, "arc_easy": 0.5180, "average": 0.4330},
        {"model": "gptq", "hellaswag": 0.3495, "arc_easy": 0.5200, "average": 0.4348},
    ]


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cpu")])
    model.generate.return_value = MagicMock(shape=(1, 192))
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(128))
    tokenizer.decode.return_value = "The quick brown fox"
    tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    return tokenizer
