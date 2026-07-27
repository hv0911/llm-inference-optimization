"""
Configuration management using dataclasses and YAML.
=====================================================
Provides typed configuration objects loaded from YAML files
with CLI override support and environment variable expansion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ============================================================================
# Project paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


# ============================================================================
# Configuration dataclasses
# ============================================================================
@dataclass
class ModelConfig:
    """Model download and loading configuration."""
    model_id: str = "Qwen/Qwen3-0.6B"
    base_dir: str = "models/base"
    awq_dir: str = "models/awq"
    gptq_dir: str = "models/gptq"
    torch_dtype: str = "auto"
    device_map: str = "auto"

    @property
    def base_path(self) -> Path:
        return PROJECT_ROOT / self.base_dir

    @property
    def awq_path(self) -> Path:
        return PROJECT_ROOT / self.awq_dir

    @property
    def gptq_path(self) -> Path:
        return PROJECT_ROOT / self.gptq_dir


@dataclass
class AWQConfig:
    """AWQ quantization parameters."""
    scheme: str = "W4A16"
    bits: int = 4
    group_size: int = 128
    targets: str = "Linear"
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])


@dataclass
class GPTQConfig:
    """GPTQ quantization parameters."""
    scheme: str = "W4A16"
    bits: int = 4
    group_size: int = 128
    targets: str = "Linear"
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])


@dataclass
class QuantizationConfig:
    """Quantization pipeline configuration."""
    num_calibration_samples: int = 256
    max_seq_length: int = 2048
    calibration_dataset: str = "wikitext"
    calibration_dataset_config: str = "wikitext-2-raw-v1"
    awq: AWQConfig = field(default_factory=AWQConfig)
    gptq: GPTQConfig = field(default_factory=GPTQConfig)


@dataclass
class ContinuousBatchingConfig:
    """Continuous batching benchmark configuration."""
    concurrency_levels: list[int] = field(
        default_factory=lambda: [1, 8, 16, 32, 64]
    )
    total_requests: int = 128
    vllm_base_url: str = "http://localhost:8000"


@dataclass
class PrefixCachingConfig:
    """Prefix caching benchmark configuration."""
    shared_prefix_length: int = 1024
    num_requests: int = 50


@dataclass
class BenchmarkConfig:
    """Benchmark pipeline configuration."""
    prompt_lengths: list[int] = field(
        default_factory=lambda: [32, 128, 512, 1024, 2048]
    )
    generation_lengths: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )
    num_warmup: int = 3
    num_iterations: int = 10
    results_dir: str = "results"
    continuous_batching: ContinuousBatchingConfig = field(
        default_factory=ContinuousBatchingConfig
    )
    prefix_caching: PrefixCachingConfig = field(
        default_factory=PrefixCachingConfig
    )

    @property
    def results_path(self) -> Path:
        return PROJECT_ROOT / self.results_dir


@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration."""
    tasks: list[str] = field(default_factory=lambda: [
        "hellaswag", "arc_easy", "arc_challenge",
        "boolq", "piqa", "winogrande",
    ])
    num_fewshot: int = 0
    batch_size: str = "auto"
    device: str = "cuda:0"


@dataclass
class ServingConfig:
    """vLLM serving configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    enable_prefix_caching: bool = False


@dataclass
class UploadConfig:
    """HuggingFace upload configuration."""
    hf_username: str = "YOUR_HF_USERNAME"
    private: bool = False
    license: str = "apache-2.0"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)


# ============================================================================
# Configuration loading
# ============================================================================
def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR} references in config values."""
    if isinstance(data, str):
        # Expand ${VAR} patterns
        while "${" in data:
            start = data.index("${")
            end = data.index("}", start)
            var_name = data[start + 2:end]
            var_value = os.environ.get(var_name, data[start:end + 1])
            data = data[:start] + var_value + data[end + 1:]
        return data
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dictionary to a nested dataclass instance."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Handle nested dataclasses
        if isinstance(value, dict):
            # Resolve the actual type from string annotation
            type_map = {
                "ModelConfig": ModelConfig,
                "QuantizationConfig": QuantizationConfig,
                "AWQConfig": AWQConfig,
                "GPTQConfig": GPTQConfig,
                "BenchmarkConfig": BenchmarkConfig,
                "ContinuousBatchingConfig": ContinuousBatchingConfig,
                "PrefixCachingConfig": PrefixCachingConfig,
                "EvaluationConfig": EvaluationConfig,
                "ServingConfig": ServingConfig,
                "UploadConfig": UploadConfig,
            }
            # Get type name from the annotation string
            type_name = field_type if isinstance(field_type, str) else field_type.__name__
            if type_name in type_map:
                value = _dict_to_dataclass(type_map[type_name], value)

        kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to config/default.yaml.

    Returns:
        AppConfig instance with all settings loaded.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        # Return defaults if no config file exists
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # Expand environment variables
    config_data = _expand_env_vars(raw_config)

    # Convert to dataclass
    return _dict_to_dataclass(AppConfig, config_data)


def load_config_with_overrides(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> AppConfig:
    """
    Load configuration and apply CLI overrides.

    Overrides use dot-notation keys like:
        model.model_id="Qwen/Qwen3-0.6B"
        quantization.awq.bits=8

    Args:
        config_path: Path to YAML config file.
        **overrides: Dot-notation key=value overrides.

    Returns:
        AppConfig with overrides applied.
    """
    config = load_config(config_path)

    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        obj = config
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], value)

    return config
