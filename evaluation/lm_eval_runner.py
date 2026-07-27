"""
lm-evaluation-harness runner.
===============================
Wraps the lm-evaluation-harness library to evaluate model quality
across standard NLP benchmarks.

Benchmarks:
- HellaSwag (commonsense reasoning)
- ARC Easy/Challenge (science reasoning)
- BoolQ (boolean question answering)
- PIQA (physical intuition)
- Winogrande (coreference resolution)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import save_results_csv, save_results_json, setup_logging

logger = setup_logging("evaluation.lm_eval")

# Standard evaluation tasks
DEFAULT_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "piqa",
    "winogrande",
]


def run_lm_eval(
    model_path: str,
    model_name: str = "model",
    tasks: list[str] | None = None,
    num_fewshot: int = 0,
    batch_size: str = "auto",
    device: str = "cuda:0",
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Run lm-evaluation-harness on a model.

    Uses the programmatic API to evaluate the model on standard
    NLP benchmarks and return structured results.

    Args:
        model_path: Path to model directory or HuggingFace model ID.
        model_name: Human-readable name for results labeling.
        tasks: List of evaluation tasks.
        num_fewshot: Number of few-shot examples.
        batch_size: Batch size ("auto" or integer string).
        device: Device to run on.
        limit: Max examples per task (for quick testing).

    Returns:
        Dict with task scores and metadata.
    """
    import lm_eval

    tasks = tasks or DEFAULT_TASKS
    task_str = ",".join(tasks)

    logger.info(f"Running evaluation for: {model_name}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Tasks: {task_str}")
    logger.info(f"  Few-shot: {num_fewshot}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")

    # Build model args
    model_args = f"pretrained={model_path},trust_remote_code=True"

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )

    # Extract scores
    scores: dict[str, float] = {}
    for task_name, task_result in results.get("results", {}).items():
        # Get the primary metric (usually acc or acc_norm)
        if "acc_norm,none" in task_result:
            scores[task_name] = task_result["acc_norm,none"]
        elif "acc,none" in task_result:
            scores[task_name] = task_result["acc,none"]
        elif "acc_norm" in task_result:
            scores[task_name] = task_result["acc_norm"]
        elif "acc" in task_result:
            scores[task_name] = task_result["acc"]

    # Average
    if scores:
        scores["average"] = sum(scores.values()) / len(scores)

    logger.info(f"\nResults for {model_name}:")
    for task, score in scores.items():
        logger.info(f"  {task}: {score:.4f}")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "scores": scores,
        "num_fewshot": num_fewshot,
        "raw_results": results.get("results", {}),
    }


def evaluate_all_models(
    model_configs: list[dict[str, str]],
    tasks: list[str] | None = None,
    num_fewshot: int = 0,
    batch_size: str = "auto",
    device: str = "cuda:0",
    results_dir: str = "results",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate multiple model variants and produce comparison table.

    Args:
        model_configs: List of dicts with 'name' and 'path' keys.
        tasks: Evaluation tasks.
        num_fewshot: Few-shot examples.
        batch_size: Batch size.
        device: Device string.
        results_dir: Output directory.
        limit: Max examples per task (for testing).

    Returns:
        DataFrame with comparison results.
    """
    tasks = tasks or DEFAULT_TASKS
    all_results: list[dict[str, Any]] = []
    comparison_data: list[dict[str, Any]] = []

    for config in model_configs:
        name = config["name"]
        path = config["path"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"{'='*60}")

        if not Path(path).exists():
            logger.warning(f"Model not found: {path} — skipping")
            continue

        result = run_lm_eval(
            model_path=path,
            model_name=name,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
        )
        all_results.append(result)

        # Build comparison row
        row = {"model": name}
        row.update(result["scores"])
        comparison_data.append(row)

    # Create comparison DataFrame
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        save_results_csv(comparison_data, f"{results_dir}/evaluation.csv")
        save_results_json(all_results, f"{results_dir}/evaluation.json")
        logger.info(f"\nEvaluation results saved to {results_dir}/evaluation.*")

        # Print comparison table
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Comparison")
        logger.info("=" * 60)
        logger.info(df.to_string(index=False, float_format="%.4f"))

        return df

    return pd.DataFrame()


def generate_eval_markdown(
    csv_path: str | Path,
) -> str:
    """Generate a Markdown comparison table from evaluation CSV."""
    df = pd.read_csv(csv_path)

    lines = ["## Evaluation Results\n"]
    lines.append("| Benchmark | " + " | ".join(df["model"].values) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in df["model"]) + " |")

    for col in df.columns:
        if col == "model":
            continue
        values = []
        for val in df[col]:
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append(f"| {col} | " + " | ".join(values) + " |")

    return "\n".join(lines)
