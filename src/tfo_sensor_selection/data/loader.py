from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfo_sensor_selection.config import DatasetConfig

_SPLIT_LABELS = list("abcde")


def _assign_validation_idx(df: pd.DataFrame, n_splits: int = 5, boundary: int = 60) -> pd.DataFrame:
    n = len(df)
    labels = _SPLIT_LABELS[:n_splits]
    n_boundaries = n_splits - 1
    group_size = (n - n_boundaries * boundary) // n_splits

    validation_idx = np.empty(n, dtype=object)
    pos = 0
    for i, label in enumerate(labels):
        end = pos + group_size if i < n_splits - 1 else n
        validation_idx[pos:end] = label
        pos = end
        if i < n_splits - 1:
            validation_idx[pos : pos + boundary] = "z"
            pos += boundary

    df = df.copy()
    df["validation_idx"] = validation_idx
    return df


def load_dataset(dataset_config: DatasetConfig) -> pd.DataFrame:
    path = Path(dataset_config.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)
    # Load a smaller set 
    # if dataset_config.name == "simulation":
    #     df = df.iloc[::4, :].reset_index(drop=True)
    required_cols = set(dataset_config.features + dataset_config.labels + [dataset_config.grouping_col])
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Dataset {path.name} is missing required columns: {missing}")

    df = _assign_validation_idx(df)
    return df
