from __future__ import annotations

from typing import Any

from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.training.hparam_search import SearchSpaceFn
from tfo_sensor_selection.training.pipeline import PipelineResult, run_pipeline


ParamSpec = dict[str, Any]


def make_search_space(param_specs: dict[str, ParamSpec]) -> SearchSpaceFn:
    def search_space(trial: Any, _model: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in param_specs.items():
            kind = spec["type"]
            if kind == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                    log=spec.get("log", False),
                )
                continue

            if kind == "float":
                step = spec.get("step")
                log = spec.get("log", False)
                if step is not None and log:
                    raise ValueError(f"Parameter '{name}' cannot use both step and log")
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    step=step,
                    log=log,
                )
                continue

            if kind == "categorical":
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
                continue

            raise ValueError(f"Unsupported parameter type '{kind}' for '{name}'")

        return params

    return search_space


def run_invivo_range_check(
    model_name: ModelName,
    param_specs: dict[str, ParamSpec] | None = None,
    *,
    n_trials: int = 20,
    seed: int = 42,
    record_results: bool = False,
    search_space_fn: SearchSpaceFn | None = None,
) -> PipelineResult:
    if param_specs is not None and search_space_fn is not None:
        raise ValueError("Pass either param_specs or search_space_fn, not both")

    effective_search_space = search_space_fn
    if effective_search_space is None and param_specs is not None:
        effective_search_space = make_search_space(param_specs)

    return run_pipeline(
        dataset_name="invivo",
        model_name=model_name,
        n_val_groups=1,
        n_test_groups=1,
        n_trials=n_trials,
        seed=seed,
        record_results=record_results,
        search_space_fn=effective_search_space,
    )


def example_random_forest() -> PipelineResult:
    return run_invivo_range_check(
        model_name="random_forest",
        n_trials=30,
        seed=42,
        param_specs={
            "n_estimators": {"type": "int", "low": 20, "high": 200, "step": 20},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 4},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        },
    )


def example_gradient_boosting() -> PipelineResult:
    return run_invivo_range_check(
        model_name="gradient_boosting",
        n_trials=30,
        seed=42,
        param_specs={
            "n_estimators": {"type": "int", "low": 20, "high": 200, "step": 20},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.2, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 6},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
        },
    )


def example_raw_callback() -> PipelineResult:
    def search_space(trial: Any, _model: Any) -> dict[str, Any]:
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "epochs": trial.suggest_int("epochs", 60, 180, step=20),
        }

    return run_invivo_range_check(
        model_name="neural_network",
        n_trials=20,
        seed=42,
        search_space_fn=search_space,
    )


def _print_result(name: str, result: PipelineResult) -> None:
    print(f"Experiment: {name}")
    print(f"  Best params: {result.best_params}")
    print(f"  Train MAE: {result.maes['train']:.6f}")
    print(f"  Val MAE: {result.maes['val']:.6f}")
    print(f"  Test MAE: {result.maes['test']:.6f}")


def main() -> None:
    baseline = run_pipeline(
        dataset_name="invivo",
        model_name="random_forest",
        n_val_groups=1,
        n_test_groups=1,
        n_trials=10,
        seed=42,
        record_results=False,
    )
    _print_result("baseline_random_forest", baseline)
    _print_result("example_random_forest", example_random_forest())
    _print_result("example_gradient_boosting", example_gradient_boosting())
    _print_result("example_raw_callback", example_raw_callback())


if __name__ == "__main__":
    main()
