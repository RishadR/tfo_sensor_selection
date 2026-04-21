from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from tfo_sensor_selection.config import DatasetConfig, DatasetName, load_metadata
from tfo_sensor_selection.data.transforms import BaseTransform
from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.training.pipeline import run_pipeline


EvolutionType = Literal["wavelength", "detector_distance"]

SEQUENCES_PATH = Path("data/sequences.yaml")
DATASET_KEY_MAP: dict[DatasetName, str] = {
    "invivo": "invivo",
    "simulation": "simulation",
}


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


def _build_feature_to_group(cfg: DatasetConfig, evolution_type: EvolutionType) -> dict[str, int]:
    if evolution_type == "wavelength":
        values = cfg.wavelength
    else:
        values = cfg.detector_distances
    return {feature_name: int(values[idx]) for idx, feature_name in enumerate(cfg.features)}


def _load_sequences_for_combo(
    dataset_name: DatasetName,
    evolution_type: EvolutionType,
) -> dict[str, list[int]]:
    path = SEQUENCES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sequence config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("sequences.yaml must parse to a top-level mapping")

    dataset_key = DATASET_KEY_MAP[dataset_name]
    dataset_entry = raw.get(dataset_key)
    if not isinstance(dataset_entry, dict):
        raise ValueError(f"Missing dataset section '{dataset_key}' in sequences.yaml")

    evolution_entry = dataset_entry.get(evolution_type)
    if not isinstance(evolution_entry, dict):
        raise ValueError(
            f"Missing evolution section '{evolution_type}' for dataset '{dataset_key}' in sequences.yaml"
        )

    sequences: dict[str, list[int]] = {}
    for strategy_name, values in evolution_entry.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"Sequence for strategy '{strategy_name}' must be a non-empty list in sequences.yaml"
            )
        sequences[str(strategy_name)] = [int(v) for v in values]
    if len(sequences) == 0:
        raise ValueError(
            f"No sequences found for dataset='{dataset_name}' and evolution_type='{evolution_type}'"
        )
    return sequences


def _build_unique_stages(
    available_group_values: list[int],
    sequences_by_strategy: dict[str, list[int]],
) -> dict[frozenset[int], set[str]]:
    stage_to_strategies: dict[frozenset[int], set[str]] = {}
    for strategy_name, sequence in sequences_by_strategy.items():
        resolved = _resolve_sequence(available_group_values, sequence)
        for end_idx in range(len(resolved)):
            stage = frozenset(resolved[: end_idx + 1])
            stage_to_strategies.setdefault(stage, set()).add(strategy_name)
    return stage_to_strategies


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


def _normalize_custom_stages(
    available_group_values: list[int],
    custom_stages: list[list[int]],
) -> list[frozenset[int]]:
    if len(custom_stages) == 0:
        raise ValueError("custom_stages must contain at least one stage when provided")

    normalized_stages: list[frozenset[int]] = []
    for stage_values in custom_stages:
        if not isinstance(stage_values, list):
            raise ValueError("Each custom stage must be a list of integers")
        resolved_stage = _resolve_sequence(available_group_values, stage_values)
        normalized_stages.append(frozenset(resolved_stage))

    return normalized_stages


def _stage_features(
    stage_values: frozenset[int],
    feature_to_group: dict[str, int],
    feature_order: list[str],
) -> list[str]:
    return [feature_name for feature_name in feature_order if feature_to_group[feature_name] in stage_values]


def _compute_error_evolution_stage(
    stage_values: frozenset[int],
    selected_features: list[str],
    strategy_names: list[str],
    dataset_name: DatasetName,
    evolution_type: EvolutionType,
    model_name: ModelName,
    seeds: list[int],
    n_val_groups: int,
    n_test_groups: int,
    n_trials: int,
    n_jobs: int,
    transforms: list[BaseTransform] | None,
    device: str = "cpu",
) -> dict[str, Any]:
    selected_values = sorted(int(v) for v in stage_values)
    rows: list[dict[str, Any]] = []

    for seed_value in seeds:
        pipeline_result = run_pipeline(
            dataset_name=dataset_name,
            model_name=model_name,
            n_val_groups=n_val_groups,
            n_test_groups=n_test_groups,
            n_trials=n_trials,
            n_jobs=n_jobs,
            custom_features=selected_features,
            transforms=transforms,
            seed=seed_value,
            record_results=False,
            device=device,
        )

        train_mae = float(pipeline_result.maes["train"])
        val_mae = float(pipeline_result.maes["val"])
        test_mae = float(pipeline_result.maes["test"])

        rows.append(
            {
                "evolution_type": str(evolution_type),
                "dataset_name": str(dataset_name),
                "current_sequence": str(selected_values),
                "selection_strategies": "|".join(sorted(strategy_names)),
                "stage_size": int(len(selected_values)),
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
        "stage_size": int(len(selected_values)),
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


def _parse_stage_values(raw_sequence: Any) -> frozenset[int]:
    if isinstance(raw_sequence, str):
        parsed = ast.literal_eval(raw_sequence)
    else:
        parsed = raw_sequence

    if not isinstance(parsed, list):
        raise ValueError(f"Invalid current_sequence value: {raw_sequence!r}")

    return frozenset(int(value) for value in parsed)


def _load_completed_stages(
    output_path: str | Path,
    dataset_name: DatasetName,
    evolution_type: EvolutionType,
) -> set[frozenset[int]]:
    out = Path(output_path)
    if not out.exists() or out.stat().st_size == 0:
        return set()

    existing_df = pd.read_csv(out)
    required_columns = {"dataset_name", "evolution_type", "current_sequence"}
    missing_columns = required_columns - set(existing_df.columns)
    if missing_columns:
        raise KeyError(
            f"Existing results file is missing required columns: {sorted(missing_columns)}"
        )

    combo_df = existing_df[
        (existing_df["dataset_name"] == str(dataset_name))
        & (existing_df["evolution_type"] == str(evolution_type))
    ]

    completed: set[frozenset[int]] = set()
    for raw_sequence in combo_df["current_sequence"].dropna().unique().tolist():
        completed.add(_parse_stage_values(raw_sequence))

    return completed


def run_error_evolution(
    dataset_name: DatasetName,
    evolution_type: EvolutionType,
    model_name: ModelName,
    custom_stages: list[list[int]] | None = None,
    seeds: list[int] | None = None,
    n_jobs_steps: int = 1,
    n_val_groups: int = 1,
    n_test_groups: int = 1,
    n_trials: int = 20,
    n_jobs: int = 20,
    transforms: list[BaseTransform] | None = None,
    device: str = "cpu",
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
    feature_to_group = _build_feature_to_group(cfg, evolution_type=evolution_type)

    if custom_stages is None:
        sequences_by_strategy = _load_sequences_for_combo(
            dataset_name=dataset_name,
            evolution_type=evolution_type,
        )
        stage_to_strategies = _build_unique_stages(list(grouped.keys()), sequences_by_strategy)
        stages = sorted(stage_to_strategies.keys(), key=lambda stage: (len(stage), tuple(sorted(stage))))
    else:
        stages = _normalize_custom_stages(list(grouped.keys()), custom_stages)
        stage_to_strategies = {stage: {"custom"} for stage in stages}

    completed_stages = _load_completed_stages(
        output_path=output_path,
        dataset_name=dataset_name,
        evolution_type=evolution_type,
    )
    stages = [stage for stage in stages if stage not in completed_stages]
    step_count = len(stages)

    if step_count == 0:
        return None

    all_rows: list[dict[str, Any]] = []

    worker_count = min(n_jobs_steps, step_count)
    if worker_count == 1:
        for stage in stages:
            row = _compute_error_evolution_stage(
                stage_values=stage,
                selected_features=_stage_features(stage, feature_to_group, cfg.features),
                strategy_names=sorted(stage_to_strategies[stage]),
                dataset_name=dataset_name,
                model_name=model_name,
                seeds=seeds,
                n_val_groups=n_val_groups,
                n_test_groups=n_test_groups,
                n_trials=n_trials,
                n_jobs=n_jobs,
                transforms=transforms,
                evolution_type=evolution_type,
                device=device,
            )
            all_rows.extend(row["rows"])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _compute_error_evolution_stage,
                    stage,
                    _stage_features(stage, feature_to_group, cfg.features),
                    sorted(stage_to_strategies[stage]),
                    dataset_name,
                    evolution_type,
                    model_name,
                    seeds,
                    n_val_groups,
                    n_test_groups,
                    n_trials,
                    n_jobs,
                    transforms,
                    device=device,
                ): stage
                for stage in stages
            }
            for future in as_completed(futures):
                result = future.result()
                all_rows.extend(result["rows"])

    results_df = pd.DataFrame(all_rows)
    results_df["timestamp"] = pd.Timestamp.now()
    results_df = results_df[
        [
            "timestamp",
            "evolution_type",
            "dataset_name",
            "current_sequence",
            "selection_strategies",
            "stage_size",
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
        # custom_stages=[[100]],
        evolution_type="wavelength",
        model_name="mlp",
        n_jobs_steps=5,
        seeds=[0, 1, 2, 3, 4],
        n_val_groups=1,
        n_test_groups=1,
        n_trials=30,
        n_jobs=30,
        device="mps",
    )
