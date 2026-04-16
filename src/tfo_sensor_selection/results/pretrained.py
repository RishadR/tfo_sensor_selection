from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfo_sensor_selection.config import DatasetConfig, DatasetName, load_metadata
from tfo_sensor_selection.data.loader import load_dataset
from tfo_sensor_selection.data.splitter import holdout_split
from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.results.model_recorder import load_trained_model_artifact


@dataclass(frozen=True)
class PretrainedSeedContext:
    dataset: DatasetName
    model_name: ModelName
    seed: int
    n_val_groups: int
    n_test_groups: int
    config: DatasetConfig
    feature_names: list[str]
    label_name: str
    transform: Any
    model: Any
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    maes: dict[str, float]


def _to_xy(df: pd.DataFrame, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[features].to_numpy(dtype=float)
    y = df[label].to_numpy(dtype=float)
    return x, y


def discover_pretrained_model_seeds(
    dataset_name: DatasetName,
    model_name: ModelName,
    artifacts_dir: str | Path,
) -> list[int]:
    base = Path(artifacts_dir)
    pattern = re.compile(
        rf"^{re.escape(str(dataset_name))}_{re.escape(str(model_name))}_seed(\d+)\.pkl$"
    )
    seeds: list[int] = []
    for path in base.glob(f"{dataset_name}_{model_name}_seed*.pkl"):
        match = pattern.match(path.name)
        if match:
            seeds.append(int(match.group(1)))
    if not seeds:
        raise FileNotFoundError(
            f"No trained model artifacts found for dataset='{dataset_name}', "
            f"model='{model_name}' in '{base}'"
        )
    return sorted(seeds)


def _resolve_seed_values(available_seed_values: list[int], requested_seed_values: list[int] | None) -> list[int]:
    if requested_seed_values is None:
        return list(available_seed_values)

    if len(requested_seed_values) == 0:
        raise ValueError("seed_values must contain at least one value")

    available = set(available_seed_values)
    resolved: list[int] = []
    seen: set[int] = set()
    for seed_value in requested_seed_values:
        seed_value = int(seed_value)
        if seed_value not in available:
            raise ValueError(
                f"Invalid seed value {seed_value}. Expected one of: {sorted(available)}"
            )
        if seed_value in seen:
            raise ValueError(f"Duplicate seed value detected: {seed_value}")
        seen.add(seed_value)
        resolved.append(seed_value)
    return resolved


def _artifact_path_for_seed(
    seed_value: int,
    dataset_name: DatasetName,
    model_name: ModelName,
    artifacts_dir: str | Path,
) -> Path:
    return Path(artifacts_dir) / f"{dataset_name}_{model_name}_seed{seed_value}.pkl"


def _load_seed_context(
    seed_value: int,
    dataset_name: DatasetName,
    model_name: ModelName,
    artifacts_dir: str | Path,
    cfg: DatasetConfig,
    df: pd.DataFrame,
) -> PretrainedSeedContext:
    artifact_path = _artifact_path_for_seed(
        seed_value=seed_value,
        dataset_name=dataset_name,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
    )
    artifact = load_trained_model_artifact(artifact_path)

    if artifact.dataset != str(dataset_name):
        raise ValueError(
            f"Artifact dataset mismatch for seed {seed_value}: "
            f"expected {dataset_name}, found {artifact.dataset}"
        )
    if artifact.model_name != model_name:
        raise ValueError(
            f"Artifact model mismatch for seed {seed_value}: "
            f"expected {model_name}, found {artifact.model_name}"
        )
    if artifact.seed != int(seed_value):
        raise ValueError(
            f"Artifact seed mismatch for seed {seed_value}: "
            f"expected {seed_value}, found {artifact.seed}"
        )

    train_df, val_df, test_df = holdout_split(
        df=df,
        grouping_col=cfg.grouping_col,
        n_val_groups=artifact.n_val_groups,
        n_test_groups=artifact.n_test_groups,
        val_start_idx=seed_value,
        ignored_group=cfg.ignored_group,
    )

    feature_names = list(artifact.feature_names)
    label_name = str(artifact.label_name)

    x_train_raw, y_train = _to_xy(train_df, feature_names, label_name)
    x_val_raw, y_val = _to_xy(val_df, feature_names, label_name)
    x_test_raw, y_test = _to_xy(test_df, feature_names, label_name)

    transform = artifact.transform
    x_train = transform.transform(x_train_raw)
    x_val = transform.transform(x_val_raw)
    x_test = transform.transform(x_test_raw)

    return PretrainedSeedContext(
        dataset=dataset_name,
        model_name=model_name,
        seed=seed_value,
        n_val_groups=artifact.n_val_groups,
        n_test_groups=artifact.n_test_groups,
        config=cfg,
        feature_names=feature_names,
        label_name=label_name,
        transform=transform,
        model=artifact.model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        maes=dict(artifact.maes),
    )


def load_pretrained_seed_contexts(
    dataset_name: DatasetName,
    model_name: ModelName,
    seed_values: list[int] | None = None,
    artifacts_dir: str | Path = "models/trained",
) -> dict[int, PretrainedSeedContext]:
    available_seed_values = discover_pretrained_model_seeds(
        dataset_name=dataset_name,
        model_name=model_name,
        artifacts_dir=artifacts_dir,
    )
    resolved_seed_values = _resolve_seed_values(available_seed_values, seed_values)

    cfg = load_metadata(dataset_name)
    df = load_dataset(cfg)

    contexts: dict[int, PretrainedSeedContext] = {}
    for seed_value in resolved_seed_values:
        contexts[seed_value] = _load_seed_context(
            seed_value=seed_value,
            dataset_name=dataset_name,
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            cfg=cfg,
            df=df,
        )
    return contexts


__all__ = [
    "PretrainedSeedContext",
    "discover_pretrained_model_seeds",
    "load_pretrained_seed_contexts",
]