from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error

from tfo_sensor_selection.models.base import BaseModel


def compute_absolute_errors(model: BaseModel, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    return np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))


def compute_mae(model: BaseModel, x: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(x)
    return float(mean_absolute_error(y, pred))
