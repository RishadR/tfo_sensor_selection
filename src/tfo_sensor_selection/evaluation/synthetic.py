from __future__ import annotations

import numpy as np
import pandas as pd

from tfo_sensor_selection.synthetic.base import ConditionalSyntheticGenerator
from tfo_sensor_selection.synthetic.conditional import generate_missing_features


def prepare_synthetic_test_input(
    x_test: np.ndarray,
    feature_names: list[str],
    synthetic_x: np.ndarray | None = None,
    target_features: list[str] | None = None,
    synthetic_values: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x_test, dtype=float)

    if synthetic_x is not None:
        sx = np.asarray(synthetic_x, dtype=float)
        if sx.shape != x.shape:
            raise ValueError(f"synthetic_x shape {sx.shape} must match x_test shape {x.shape}")
        return sx.copy()

    if target_features is None or synthetic_values is None:
        raise ValueError("Provide either synthetic_x, or both target_features and synthetic_values")

    modified = x.copy()
    indices = [feature_names.index(name) for name in target_features]

    vals = np.asarray(synthetic_values, dtype=float)
    if vals.ndim == 1 and len(indices) == 1:
        vals = vals.reshape(-1, 1)

    if vals.shape[0] != modified.shape[0] or vals.shape[1] != len(indices):
        raise ValueError(
            f"synthetic_values shape {vals.shape} must be (n_samples, n_target_features)="
            f"({modified.shape[0]}, {len(indices)})"
        )

    modified[:, indices] = vals
    return modified


def prepare_conditional_synthetic_features(
    generator: ConditionalSyntheticGenerator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_feature_names: list[str],
    label_name: str,
    known_feature_names: list[str],
) -> np.ndarray:
    conditional_column_names = [label_name] + [f for f in known_feature_names if f != label_name]
    all_column_names = [label_name] + [f for f in all_feature_names if f != label_name]
    target_column_names = [f for f in all_feature_names if f not in conditional_column_names]

    completed = generate_missing_features(
        generator=generator,
        train_df=train_df,
        source_df=test_df[all_column_names].copy(),
        all_column_names=all_column_names,
        conditional_column_names=conditional_column_names,
        target_column_names=target_column_names,
    )
    return completed[all_feature_names].to_numpy(dtype=float)
