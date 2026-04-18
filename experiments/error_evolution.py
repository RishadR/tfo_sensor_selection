from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from tfo_sensor_selection.config import DatasetConfig, DatasetName, load_metadata
from tfo_sensor_selection.data.transforms import BaseTransform
from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.training.pipeline import run_pipeline


EvolutionType = Literal["wavelength", "detector_distance"]


def _build_group_to_features(cfg: DatasetConfig, evolution_type: EvolutionType) -> dict[int, list[str]]:
    if evolution_type == "wavelength":
        values = cfg.wavelength
    else:
        values = cfg.detector_distances

    grouped: dict[int, list[str]] = {}
    for idx, feature_name in enumerate(cfg.features):
        key = int(values[idx])
        grouped.setdefault(key, []).append(feature_name)

    return grouped


def _resolve_sequence(available_group_values: list[int], requested_sequence: list[int]) -> list[int]:
    available = set(available_group_values)
    sequence: list[int] = []
    seen: set[int] = set()
    for value in requested_sequence:
        value = int(value)
        if value not in available:
            raise ValueError(f"Invalid sequence value {value}. Expected one of: {sorted(available)}")
        if value in seen:
            raise ValueError(f"Duplicate sequence value detected: {value}")
        seen.add(value)
        sequence.append(value)

    if len(sequence) == 0:
        raise ValueError("sequence must contain at least one value")
    return sequence


def _flatten_features(sequence_prefix: list[int], grouped: dict[int, list[str]]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for group_value in sequence_prefix:
        for feature_name in grouped[group_value]:
            if feature_name not in seen:
                selected.append(feature_name)
                seen.add(feature_name)
    return selected


def _compute_error_evolution_step(
    step_idx: int,
    resolved_sequence: list[int],
    grouped: dict[int, list[str]],
    dataset_name: DatasetName,
    evolution_type: EvolutionType,
    model_name: ModelName,
    seeds: list[int],
    n_val_groups: int,
    n_test_groups: int,
    n_trials: int,
    transforms: list[BaseTransform] | None,
) -> dict[str, Any]:
    selected_values = resolved_sequence[: step_idx + 1]
    selected_features = _flatten_features(selected_values, grouped)
    rows: list[dict[str, Any]] = []

    for seed_value in seeds:
        pipeline_result = run_pipeline(
            dataset_name=dataset_name,
            model_name=model_name,
            n_val_groups=n_val_groups,
            n_test_groups=n_test_groups,
            n_trials=n_trials,
            custom_features=selected_features,
            transforms=transforms,
            seed=seed_value,
            record_results=False,
        )

        train_mae = float(pipeline_result.maes["train"])
        val_mae = float(pipeline_result.maes["val"])
        test_mae = float(pipeline_result.maes["test"])

        rows.append(
            {
                "evolution_type": str(evolution_type),
                "dataset_name": str(dataset_name),
                "current_sequence": str(selected_values),
                "seed": int(seed_value),
                "model_name": str(model_name),
                "n_val_groups": int(n_val_groups),
                "n_test_groups": int(n_test_groups),
                "n_trials": int(n_trials),
                "train_error": train_mae,
                "val_error": val_mae,
                "test_error": test_mae,
            }
        )

    return {
        "step_index": int(step_idx + 1),
        "rows": rows,
    }


def record_error_evolution_result(
    results_df: pd.DataFrame,
    output_path: str | Path = "results/error_evolution.csv",
    append: bool = False,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if append and out.exists() and out.stat().st_size > 0:
        existing_df = pd.read_csv(out)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_csv(out, index=False)
    else:
        results_df.to_csv(out, index=False)


def run_error_evolution(
    dataset_name: DatasetName,
    evolution_type: EvolutionType,
    sequence: list[int],
    model_name: ModelName,
    method_name: str,
    seeds: list[int] | None = None,
    n_jobs_steps: int = 1,
    n_val_groups: int = 1,
    n_test_groups: int = 1,
    n_trials: int = 20,
    transforms: list[BaseTransform] | None = None,
    record_csv: bool = True,
    output_path: str | Path = "results/error_evolution.csv",
) -> None:
    if seeds is None:
        seeds = [0, 1, 2]
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one value")
    if not all(isinstance(seed_value, int) for seed_value in seeds):
        raise ValueError("seeds must be a list of integers")
    if n_jobs_steps < 1:
        raise ValueError("n_jobs_steps must be >= 1")

    cfg = load_metadata(dataset_name)
    grouped = _build_group_to_features(cfg, evolution_type=evolution_type)
    resolved_sequence = _resolve_sequence(list(grouped.keys()), sequence)
    step_count = len(resolved_sequence)

    all_rows: list[dict[str, Any]] = []

    worker_count = min(n_jobs_steps, step_count)
    if worker_count == 1:
        for step_idx in range(step_count):
            row = _compute_error_evolution_step(
                step_idx=step_idx,
                resolved_sequence=resolved_sequence,
                grouped=grouped,
                dataset_name=dataset_name,
                model_name=model_name,
                seeds=seeds,
                n_val_groups=n_val_groups,
                n_test_groups=n_test_groups,
                n_trials=n_trials,
                transforms=transforms,
                evolution_type=evolution_type,
            )
            all_rows.extend(row["rows"])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _compute_error_evolution_step,
                    step_idx,
                    resolved_sequence,
                    grouped,
                    dataset_name,
                    evolution_type,
                    model_name,
                    seeds,
                    n_val_groups,
                    n_test_groups,
                    n_trials,
                    transforms,
                ): step_idx
                for step_idx in range(step_count)
            }
            for future in as_completed(futures):
                result = future.result()
                all_rows.extend(result["rows"])

    results_df = pd.DataFrame(all_rows)
    results_df["method_name"] = method_name
    results_df["timestamp"] = pd.Timestamp.now()
    results_df = results_df[
        [
            "timestamp",
            "method_name",
            "evolution_type",
            "dataset_name",
            "current_sequence",
            "seed",
            "model_name",
            "n_val_groups",
            "n_test_groups",
            "n_trials",
            "train_error",
            "val_error",
            "test_error",
        ]
    ]

    if record_csv:
        record_error_evolution_result(results_df=results_df, output_path=output_path, append=True)

    return None


if __name__ == "__main__":
    run_error_evolution(
        dataset_name="invivo",
        # dataset_name="simulation",
        # evolution_type="detector_distance",
        evolution_type="wavelength",
        model_name="mlp",
        method_name="PFI",
        n_jobs_steps=5,
        # sequence=[690, 735, 910, 850, 810],
        sequence=[735, 810, 850],
        # sequence=[15, 30, 100, 75, 45],
        # sequence = [70, 90, 50, 30, 10],
        seeds=[0, 1, 2, 3, 4],
        n_val_groups=1,
        n_test_groups=1,
        n_trials=30,
    )
