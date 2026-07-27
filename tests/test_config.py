"""
Tests for configuration loading and validation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AWQConfig,
    AppConfig,
    BenchmarkConfig,
    GPTQConfig,
    ModelConfig,
    QuantizationConfig,
    load_config,
    load_config_with_overrides,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_defaults(self):
        config = ModelConfig()
        assert config.model_id == "Qwen/Qwen3-0.6B"
        assert config.torch_dtype == "auto"
        assert config.device_map == "auto"

    def test_custom_values(self):
        config = ModelConfig(model_id="custom/model", torch_dtype="float16")
        assert config.model_id == "custom/model"
        assert config.torch_dtype == "float16"

    def test_path_properties(self):
        config = ModelConfig(base_dir="models/base")
        assert config.base_path.name == "base"


class TestQuantizationConfig:
    """Test quantization config dataclasses."""

    def test_awq_defaults(self):
        config = AWQConfig()
        assert config.scheme == "W4A16"
        assert config.bits == 4
        assert config.group_size == 128
        assert "lm_head" in config.ignore

    def test_gptq_defaults(self):
        config = GPTQConfig()
        assert config.scheme == "W4A16"
        assert config.bits == 4

    def test_quantization_config(self):
        config = QuantizationConfig()
        assert config.num_calibration_samples == 256
        assert config.max_seq_length == 2048
        assert config.awq.scheme == "W4A16"
        assert config.gptq.scheme == "W4A16"


class TestBenchmarkConfig:
    """Test benchmark configuration."""

    def test_defaults(self):
        config = BenchmarkConfig()
        assert 32 in config.prompt_lengths
        assert 2048 in config.prompt_lengths
        assert config.num_warmup == 3
        assert config.num_iterations == 10

    def test_continuous_batching(self):
        config = BenchmarkConfig()
        assert 1 in config.continuous_batching.concurrency_levels
        assert 64 in config.continuous_batching.concurrency_levels


class TestAppConfig:
    """Test top-level AppConfig."""

    def test_defaults(self):
        config = AppConfig()
        assert config.model.model_id == "Qwen/Qwen3-0.6B"
        assert config.quantization.awq.bits == 4
        assert config.benchmark.num_iterations == 10
        assert config.serving.port == 8000


class TestLoadConfig:
    """Test config loading from YAML."""

    def test_load_missing_file(self):
        """Loading from non-existent file should return defaults."""
        config = load_config("/non/existent/path.yaml")
        assert isinstance(config, AppConfig)
        assert config.model.model_id == "Qwen/Qwen3-0.6B"

    def test_load_from_yaml(self, tmp_path: Path):
        """Load config from a YAML file."""
        yaml_content = {
            "model": {
                "model_id": "custom/model",
                "base_dir": "custom/base",
            },
            "benchmark": {
                "num_iterations": 5,
            },
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(config_file)
        assert config.model.model_id == "custom/model"
        assert config.benchmark.num_iterations == 5

    def test_load_with_overrides(self, tmp_path: Path):
        """Test CLI override functionality."""
        yaml_content = {"model": {"model_id": "original/model"}}
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config_with_overrides(
            config_file,
            **{"model.model_id": "overridden/model"}
        )
        assert config.model.model_id == "overridden/model"

    def test_env_var_expansion(self, tmp_path: Path):
        """Test ${VAR} expansion in config values."""
        os.environ["TEST_MODEL_ID"] = "env/model"

        yaml_content = {"model": {"model_id": "${TEST_MODEL_ID}"}}
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(config_file)
        assert config.model.model_id == "env/model"

        del os.environ["TEST_MODEL_ID"]
