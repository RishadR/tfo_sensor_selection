from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from tfo_sensor_selection.models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        self._random_state = random_state
        self._model = RandomForestRegressor(random_state=random_state, **kwargs)

    def suggest_params(self, trial: Any) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }

    def set_params(self, **params: Any) -> "RandomForestModel":
        params = {**params, "random_state": self._random_state}
        self._model = RandomForestRegressor(**params)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self._model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=float)


class GradientBoostingModel(BaseModel):
    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        self._random_state = random_state
        self._model = GradientBoostingRegressor(random_state=random_state, **kwargs)

    def suggest_params(self, trial: Any) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

    def set_params(self, **params: Any) -> "GradientBoostingModel":
        params = {**params, "random_state": self._random_state}
        self._model = GradientBoostingRegressor(**params)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostingModel":
        self._model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=float)
