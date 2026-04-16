from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.results import save_trained_model_artifact
from tfo_sensor_selection.training.hparam_search import SearchSpaceFn
from tfo_sensor_selection.training.pipeline import PipelineResult, run_pipeline


@dataclass(frozen=True)
class SavedExperimentRun:
    result: PipelineResult
    artifact_path: Path


def run_experiment(
    model_name: ModelName,
    n_trials: int = 20,
    seed: int = 42,
    record_results: bool = True,
    search_space_fn: SearchSpaceFn | None = None,
) -> PipelineResult:
    return run_pipeline(
        dataset_name="invivo",
        model_name=model_name,
        n_val_groups=1,
        n_test_groups=1,
        n_trials=n_trials,
        seed=seed,
        record_results=record_results,
        search_space_fn=search_space_fn,
    )


def run_seed_sweep(
    model_name: ModelName,
    seeds: list[int] = [0, 1, 2],
    n_trials: int = 20,
    record_results: bool = True,
    search_space_fn: SearchSpaceFn | None = None,
    artifact_dir: str | Path = "models/trained",
) -> list[SavedExperimentRun]:
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")

    saved_runs: list[SavedExperimentRun] = []
    for seed in seeds:
        result = run_experiment(
            model_name=model_name,
            n_trials=n_trials,
            seed=seed,
            record_results=record_results,
            search_space_fn=search_space_fn,
        )
        artifact_path = save_trained_model_artifact(result=result, output_dir=artifact_dir)
        saved_runs.append(SavedExperimentRun(result=result, artifact_path=artifact_path))

    return saved_runs

def main(seeds: list[int] = [0, 1, 2]):
    saved_runs = run_seed_sweep(model_name="mlp", seeds=seeds)
    print("Experiment: invivo_model_training")
    for run in saved_runs:
        result = run.result
        print(f"Seed: {result.seed}")
        print(f"Model: {result.model_name}")
        print(f"Best params: {result.best_params}")
        print(f"Artifact: {run.artifact_path}")
        print(f"Train MAE: {result.maes['train']:.6f}")
        print(f"Val MAE: {result.maes['val']:.6f}")
        print(f"Test MAE: {result.maes['test']:.6f}")



if __name__ == "__main__":
    main(seeds=[0, 1, 2])