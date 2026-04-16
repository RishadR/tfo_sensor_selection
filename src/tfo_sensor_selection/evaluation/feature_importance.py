from __future__ import annotations

from typing import Any

import numpy as np

from tfo_sensor_selection.evaluation.synthetic import prepare_synthetic_test_input
from tfo_sensor_selection.training.trainer import compute_mae


def evaluate_synthetic_impact(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    synthetic_cases: dict[str, dict[str, Any]],
) -> dict[str, float]:
    baseline_mae = compute_mae(model, x_test, y_test)
    output: dict[str, float] = {"baseline_mae": baseline_mae}

    for case_name, case_kwargs in synthetic_cases.items():
        x_case = prepare_synthetic_test_input(
            x_test=x_test,
            feature_names=feature_names,
            synthetic_x=case_kwargs.get("synthetic_x"),
            target_features=case_kwargs.get("target_features"),
            synthetic_values=case_kwargs.get("synthetic_values"),
        )
        mae = compute_mae(model, x_case, y_test)
        output[f"{case_name}_mae"] = mae
        output[f"{case_name}_mae_delta"] = mae - baseline_mae

    return output
