from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error

from tfo_sensor_selection.models import ModelName, build_model
from tfo_sensor_selection.models.base import BaseModel


SearchSpaceFn = Callable[[Any, BaseModel], dict[str, Any]]


def search_best_params(
    model_name: ModelName,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
    n_jobs: int = 10,
    search_space_fn: SearchSpaceFn | None = None,
) -> tuple[dict[str, Any], float]:

    def objective(trial: Any) -> float:
        model = build_model(model_name, seed=seed)
        params = model.suggest_params(trial) if search_space_fn is None else search_space_fn(trial, model)
        model.set_params(**params)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        return float(mean_absolute_error(y_val, pred))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)

    return dict(study.best_params), float(study.best_value)
