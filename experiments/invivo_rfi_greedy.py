from __future__ import annotations

from pathlib import Path

from tfo_sensor_selection.feature_selection.rfi_greedy import (
    RFISplit,
    build_distance_groups,
    build_wavelength_groups,
    run_greedy_group_selection,
)
from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.results import load_pretrained_seed_contexts
from tfo_sensor_selection.results.rfi_recorder import record_rfi_rows

def run_invivo_rfi_greedy(
    model_name: ModelName = "mlp",
    generator_name: str = "arf",
    generator_model_path: Path | None = Path("models/invivo_arf.pkl"),  # Reuse fitted model
    rfi_split: RFISplit = "test",
    seed: list[int] | None = None,
    synthesize_count: int = 1,
    n_job_synthesize: int = 1,
    record_rfi_csv: bool = True,
    rfi_output_path: str | Path = "results/invivo_rfi_greedy_results.csv",
    artifacts_dir: str | Path = "models/trained",
) -> dict[str, object]:
    if seed is not None and (
        not isinstance(seed, list) or len(seed) == 0 or not all(isinstance(s, int) for s in seed)
    ):
        raise ValueError("seed must be None or a non-empty list[int]")

    seed_contexts = load_pretrained_seed_contexts(
        dataset_name="invivo",
        model_name=model_name,
        seed_values=seed,
        artifacts_dir=artifacts_dir,
    )
    loaded_seed_values = sorted(seed_contexts)
    model_contexts = [seed_contexts[seed_value] for seed_value in loaded_seed_values]

    pipeline_result = model_contexts[0]
    generator_seed = loaded_seed_values[0]

    wavelength_groups = build_wavelength_groups(pipeline_result.config)
    distance_groups = build_distance_groups(pipeline_result.config)

    wavelength_run = run_greedy_group_selection(
        model_contexts=model_contexts,
        grouping_type="wavelength",
        group_to_features=wavelength_groups,
        generator_name=generator_name,
        generator_model_path=generator_model_path,
        seed=generator_seed,
        rfi_mode="mean",
        rfi_split=rfi_split,
        synthesize_count=synthesize_count,
        n_jobs_synthesize=n_job_synthesize,
    )
    distance_run = run_greedy_group_selection(
        model_contexts=model_contexts,
        grouping_type="detector_distance",
        group_to_features=distance_groups,
        generator_name=generator_name,
        generator_model_path=generator_model_path,
        seed=generator_seed,
        rfi_mode="mean",
        rfi_split=rfi_split,
        synthesize_count=synthesize_count,
        n_jobs_synthesize=n_job_synthesize,
    )

    if record_rfi_csv:
        rows = list(wavelength_run["rows"])  + list(distance_run["rows"])
        record_rfi_rows(rows=rows, output_path=rfi_output_path)

    return {
        "dataset": pipeline_result.dataset,
        "model_name": pipeline_result.model_name,
        "generator_name": generator_name,
        "rfi_mode": "mean",
        "rfi_split": rfi_split,
        "seed": loaded_seed_values,
        "train_mae": float(sum(ctx.maes["train"] for ctx in model_contexts) / len(model_contexts)),
        "val_mae": float(sum(ctx.maes["val"] for ctx in model_contexts) / len(model_contexts)),
        "test_mae": float(sum(ctx.maes["test"] for ctx in model_contexts) / len(model_contexts)),
        "baseline_split_mae": float(wavelength_run["baseline_split_mae"]),
        "wavelength_order": wavelength_run["selected_order"],
        "distance_order": distance_run["selected_order"],
    }


if __name__ == "__main__":
    result = run_invivo_rfi_greedy(
        model_name="mlp",
        generator_name="arf",
        rfi_split="test",
        synthesize_count=50,
        n_job_synthesize=8,
    ) 
    print("Experiment: invivo_rfi_greedy")
    print(f"Seed: {result['seed']}")
    print(f"Generator: {result['generator_name']}")
    print(f"RFI Split: {result['rfi_split']}")
    print(f"Train MAE: {result['train_mae']:.6f}")
    print(f"Val MAE: {result['val_mae']:.6f}")
    print(f"Test MAE: {result['test_mae']:.6f}")
    print(f"Selected-Split Baseline MAE: {result['baseline_split_mae']:.6f}")
    print(f"Wavelength order: {result['wavelength_order']}")
    print(f"Distance order: {result['distance_order']}")
