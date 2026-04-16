from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfo_sensor_selection.config import DatasetConfig, DatasetName, load_metadata
from tfo_sensor_selection.data.loader import load_dataset
from tfo_sensor_selection.data.splitter import holdout_split
from tfo_sensor_selection.data.transforms import BaseTransform, TransformPipeline
from tfo_sensor_selection.models import ModelName, build_model
from tfo_sensor_selection.models.base import BaseModel
from tfo_sensor_selection.results.recorder import record_model_metrics
from tfo_sensor_selection.training.hparam_search import SearchSpaceFn, search_best_params
from tfo_sensor_selection.training.trainer import compute_mae


@dataclass
class PipelineResult:
    dataset: DatasetName
    model_name: ModelName
    seed: int
    n_val_groups: int
    n_test_groups: int
    model: BaseModel
    best_params: dict[str, Any]
    config: DatasetConfig
    feature_names: list[str]
    label_name: str
    transform: TransformPipeline
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


def _to_xy(df, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[features].to_numpy(dtype=float)
    y = df[label].to_numpy(dtype=float)
    return x, y


def run_pipeline(
    dataset_name: DatasetName,
    model_name: ModelName,
    n_val_groups: int,
    n_test_groups: int,
    n_trials: int = 50,
    n_jobs: int = 10,
    custom_features: list[str] | None = None,
    transforms: list[BaseTransform] | None = None,
    seed: int = 42,
    record_results: bool = True,
    search_space_fn: SearchSpaceFn | None = None,
) -> PipelineResult:
    """
    Run the full training pipeline: load data, split, transform, train model, and evaluate.

    Args:
        dataset_name: The name of the dataset to load and train on
        model_name: The name of the model to train
        n_val_groups: The number of groups to hold out for validation
        n_test_groups: The number of groups to hold out for testing
        n_trials: The number of hyperparameter search trials to perform
        n_jobs: The number of parallel jobs to use for hyperparameter search in Optuna
        custom_features: A list of custom features to use instead of the default dataset features from the cfg
        ignore if you don't want to override the default features
        transforms: A list of data transformations to apply to the features
        seed: The random seed for reproducibility
        record_results: Whether to record the results of the pipeline run to a CSV file
        search_space_fn: Optional Optuna search-space callback that overrides the model defaults

    Returns:
        A PipelineResult object containing the trained model, data splits, and evaluation metrics
    """

    cfg = load_metadata(dataset_name)
    df = load_dataset(cfg)

    train_df, val_df, test_df = holdout_split(
        df=df,
        grouping_col=cfg.grouping_col,
        n_val_groups=n_val_groups,
        n_test_groups=n_test_groups,
        val_start_idx=seed,
        ignored_group=cfg.ignored_group,
    )

    label_name = cfg.labels[0]
    if custom_features is not None:
        features = custom_features
    else:
        features = cfg.features
    x_train_raw, y_train = _to_xy(train_df, features, label_name)
    x_val_raw, y_val = _to_xy(val_df, features, label_name)
    x_test_raw, y_test = _to_xy(test_df, features, label_name)

    transform = TransformPipeline(transforms)
    x_train = transform.fit_transform(x_train_raw)
    x_val = transform.transform(x_val_raw)
    x_test = transform.transform(x_test_raw)

    best_params, _ = search_best_params(
        model_name=model_name,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        n_trials=n_trials,
        n_jobs=n_jobs,
        seed=seed,
        search_space_fn=search_space_fn,
    )

    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    model = build_model(model_name, seed=seed)
    model.set_params(**best_params)
    model.fit(x_train_val, y_train_val)

    maes = {
        "train": compute_mae(model, x_train, y_train),
        "val": compute_mae(model, x_val, y_val),
        "test": compute_mae(model, x_test, y_test),
    }

    result = PipelineResult(
        dataset=dataset_name,
        model_name=model_name,
        seed=seed,
        n_val_groups=n_val_groups,
        n_test_groups=n_test_groups,
        model=model,
        best_params=best_params,
        config=cfg,
        feature_names=features,
        label_name=label_name,
        transform=transform,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        maes=maes,
    )

    if record_results:
        record_model_metrics(
            {
                "dataset": result.dataset,
                "model": result.model_name,
                "seed": result.seed,
                "n_val_groups": result.n_val_groups,
                "n_test_groups": result.n_test_groups,
                "best_params": result.best_params,
                "maes": result.maes,
            }
        )

    return result


