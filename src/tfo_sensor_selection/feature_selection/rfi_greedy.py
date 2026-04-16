from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from tfo_sensor_selection.config import DatasetConfig
from tfo_sensor_selection.synthetic.conditional import generate_missing_features
from tfo_sensor_selection.synthetic.factory import build_generator
from tfo_sensor_selection.training.trainer import compute_absolute_errors


GroupingType = Literal["wavelength", "detector_distance"]
RFIMode = Literal["mean", "per_sample"]
RFISplit = Literal["train", "val", "test", "all"]


def _validate_feature_alignment(cfg: DatasetConfig) -> None:
    n_features = len(cfg.features)
    if len(cfg.wavelength) != n_features:
        raise ValueError(
            "Metadata mismatch: wavelength list length must equal number of features "
            f"({len(cfg.wavelength)} != {n_features})"
        )
    if len(cfg.detector_distances) != n_features:
        raise ValueError(
            "Metadata mismatch: detector_distances list length must equal number of features "
            f"({len(cfg.detector_distances)} != {n_features})"
        )


def build_wavelength_groups(cfg: DatasetConfig) -> dict[int, list[str]]:
    _validate_feature_alignment(cfg)
    groups: dict[int, list[str]] = {}
    for idx, feature_name in enumerate(cfg.features):
        key = int(cfg.wavelength[idx])
        groups.setdefault(key, []).append(feature_name)
    return {key: groups[key] for key in sorted(groups)}


def build_distance_groups(cfg: DatasetConfig) -> dict[int, list[str]]:
    _validate_feature_alignment(cfg)
    groups: dict[int, list[str]] = {}
    for idx, feature_name in enumerate(cfg.features):
        key = int(cfg.detector_distances[idx])
        groups.setdefault(key, []).append(feature_name)
    return {key: groups[key] for key in sorted(groups)}


def _flatten_selected_features(selected_group_values: list[int], groups: dict[int, list[str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group_value in selected_group_values:
        for feature_name in groups[group_value]:
            if feature_name not in seen:
                seen.add(feature_name)
                ordered.append(feature_name)
    return ordered


def _get_split_payload(
    model_context: Any,
    rfi_split: RFISplit,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    if rfi_split == "train":
        return (
            model_context.train_df,
            model_context.x_train,
            model_context.y_train,
            float(model_context.maes["train"]),
        )
    if rfi_split == "val":
        return (
            model_context.val_df,
            model_context.x_val,
            model_context.y_val,
            float(model_context.maes["val"]),
        )
    if rfi_split == "test":
        return (
            model_context.test_df,
            model_context.x_test,
            model_context.y_test,
            float(model_context.maes["test"]),
            )
    if rfi_split == "all":
        return (
            pd.concat([model_context.train_df, model_context.val_df, model_context.test_df], ignore_index=True),
            np.concatenate([model_context.x_train, model_context.x_val, model_context.x_test], axis=0),
            np.concatenate([model_context.y_train, model_context.y_val, model_context.y_test], axis=0),
            float(model_context.maes["all"]),
        )
    else:
        raise ValueError(f"Invalid rfi_split value: {rfi_split}")


def _synthesize_one(
    synth_idx: int,
    run_contexts: list[dict[str, Any]],
    source_columns: list[str],
    feature_names: list[str],
    conditioned_feature_names: list[str],
    target_feature_names: list[str],
) -> tuple[float, float]:
    """Run one round of synthesis across all run_contexts and return (mean_rfi, synth_mae)."""
    per_ctx_rfi: list[float] = []
    per_ctx_synth_mae: list[float] = []
    for ctx in run_contexts:
        model_context = ctx["model_context"]
        completed_df = generate_missing_features(
            generator=ctx["generator"],
            train_df=model_context.train_df,
            source_df=ctx["split_df"][source_columns].copy(),
            all_column_names=source_columns,
            conditional_column_names=conditioned_feature_names,
            target_column_names=target_feature_names,
            fit=False,
        )
        x_synth_raw = completed_df[feature_names].to_numpy(dtype=float)
        x_synth = model_context.transform.transform(x_synth_raw)
        synthesized_absolute_errors = compute_absolute_errors(model_context.model, x_synth, ctx["y_split"])
        synth_split_mae = float(np.mean(synthesized_absolute_errors))
        mean_rfi = synth_split_mae - ctx["baseline_split_mae"]
        per_ctx_rfi.append(mean_rfi)
        per_ctx_synth_mae.append(synth_split_mae)
    return float(np.mean(per_ctx_rfi)), float(np.mean(per_ctx_synth_mae))


def run_greedy_group_selection(
    model_contexts: Sequence[Any],
    grouping_type: GroupingType,
    group_to_features: dict[int, list[str]],
    generator_name: str,
    seed: int,
    generator_params: dict = {},
    generator_model_path: Path | None = None,
    rfi_mode: RFIMode = "mean",
    rfi_split: RFISplit = "test",
    synthesize_count: int = 1,
    n_jobs_synthesize: int = 1,
) -> dict[str, Any]:
    """
    Run greedy RFI-based group selection.  
    
    Args:
        model_contexts: List of model evaluation contexts used to aggregate RFI
        grouping_type: The type of grouping to perform ("wavelength" or "detector_distance")
        group_to_features: A mapping from group values to the list of feature names in that group
        generator_name: The name of the generator to use for synthesizing missing features
        seed: The random seed for reproducibility
        generator_params: Additional parameters to pass to the generator 
        (e.g. ARF parameters - check ARFGenerator.__init__ for details)
        generator_model_path: Optional path to a prefit generator model. If provided,
        the generator is loaded from disk and not fit in this function.
        rfi_mode: Compute mean RFI score across samples ("mean") or record per-sample RFI scores ("per_sample")
        rfi_split: The data split to evaluate RFI/generate fake data on ("train", "val", or "test"). Note that
        we both train and generate on the same split to avoid distribution shift between training and evaluation of the generator, which could confound RFI results.
        synthesize_count: Number of times to redraw synthetic data and compute RFI for each candidate (default 1).
                         Results are aggregated as mean and variance across redraws.
        n_jobs_synthesize: Number of parallel jobs to use for synthesizing data and computing RFI across multiple 
        model contexts. Only applicable if synthesize_count > 1.
    """
    
    if len(model_contexts) == 0:
        raise ValueError("model_contexts must contain at least one model context")
    if rfi_mode != "mean":
        raise ValueError("run_greedy_group_selection only supports rfi_mode='mean'")

    ref_result = model_contexts[0]
    label_name = ref_result.label_name
    feature_names = list(ref_result.feature_names)
    for model_context in model_contexts[1:]:
        if model_context.label_name != label_name:
            raise ValueError("All model contexts must share the same label_name")
        if list(model_context.feature_names) != feature_names:
            raise ValueError("All model contexts must share the same feature_names")

    run_contexts: list[dict[str, Any]] = []
    for model_context in model_contexts:
        split_df, x_split, y_split, baseline_split_mae = _get_split_payload(
            model_context=model_context,
            rfi_split=rfi_split,
        )
        baseline_absolute_errors = compute_absolute_errors(model_context.model, x_split, y_split)
        generator = build_generator(
            generator_name=generator_name,
            seed=seed,
            extra_params=generator_params,
            model_path=generator_model_path,
        )
        if generator_model_path is None:
            generator.fit(
                train_df=split_df,
                fit_columns=feature_names,
            )
        run_contexts.append(
            {
                "model_context": model_context,
                "split_df": split_df,
                "x_split": x_split,
                "y_split": y_split,
                "baseline_split_mae": float(baseline_split_mae),
                "baseline_absolute_errors": baseline_absolute_errors,
                "generator": generator,
            }
        )

    baseline_split_mae = float(np.mean([ctx["baseline_split_mae"] for ctx in run_contexts]))
    source_columns = [label_name] + feature_names

    remaining_group_values = sorted(group_to_features)
    selected_group_values: list[int] = []
    round_rows: list[dict[str, Any]] = []

    model_names = {ctx["model_context"].model_name for ctx in run_contexts}
    run_model_name = ref_result.model_name if len(model_names) == 1 else "multi_model"

    for round_index in range(1, len(remaining_group_values) + 1):
        conditioned_feature_names = _flatten_selected_features(selected_group_values, group_to_features)
        candidate_rows: list[dict[str, Any]] = []

        for candidate_group_value in remaining_group_values:
            target_feature_names = list(group_to_features[candidate_group_value])

            # Collect RFI and synthesized errors across multiple synthesizations and models
            per_synthesize_rfi: list[float] = []
            per_synthesize_synth_mae: list[float] = []

            worker_count = min(n_jobs_synthesize, synthesize_count)
            if worker_count <= 1:
                for synth_idx in range(synthesize_count):
                    rfi, synth_mae = _synthesize_one(
                        synth_idx,
                        run_contexts,
                        source_columns,
                        feature_names,
                        conditioned_feature_names,
                        target_feature_names,
                    )
                    per_synthesize_rfi.append(rfi)
                    per_synthesize_synth_mae.append(synth_mae)
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(
                            _synthesize_one,
                            synth_idx,
                            run_contexts,
                            source_columns,
                            feature_names,
                            conditioned_feature_names,
                            target_feature_names,
                        ): synth_idx
                        for synth_idx in range(synthesize_count)
                    }
                    for future in as_completed(futures):
                        rfi, synth_mae = future.result()
                        per_synthesize_rfi.append(rfi)
                        per_synthesize_synth_mae.append(synth_mae)

            # Compute aggregate statistics across synthesizations
            rfi_mean = float(np.mean(per_synthesize_rfi))
            rfi_variance = float(np.var(per_synthesize_rfi))
            synth_split_mae_mean = float(np.mean(per_synthesize_synth_mae))

            train_mae = float(np.mean([ctx["model_context"].maes["train"] for ctx in run_contexts]))
            val_mae = float(np.mean([ctx["model_context"].maes["val"] for ctx in run_contexts]))
            test_mae = float(np.mean([ctx["model_context"].maes["test"] for ctx in run_contexts]))

            candidate_rows.append(
                {
                    "dataset": ref_result.dataset,
                    "model_name": run_model_name,
                    "seed": int(seed),
                    "generator_name": generator_name,
                    "grouping_type": grouping_type,
                    "rfi_mode": rfi_mode,
                    "rfi_split": rfi_split,
                    "round_index": int(round_index),
                    "candidate_group_value": int(candidate_group_value),
                    "candidate_group_features": target_feature_names,
                    "conditioned_group_values": list(selected_group_values),
                    "conditioned_features": list(conditioned_feature_names),
                    "synthesized_split_mae": synth_split_mae_mean,
                    "baseline_split_mae": float(baseline_split_mae),
                    "rfi_mean": rfi_mean,
                    "rfi_variance": rfi_variance,
                    "per_sample_rfi": None,
                    "baseline_absolute_errors": None,
                    "synthesized_absolute_errors": None,
                    "selected_group_value": None,
                    "selected_in_round": False,
                    "selected_order": [],
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "test_mae": test_mae,
                }
            )

        # Largest RFI wins. Tie-breaker: smallest group value for deterministic behavior.
        selected_row = sorted(candidate_rows, key=lambda row: (-float(row["rfi_mean"]), int(row["candidate_group_value"])))[0]
        selected_group_value = int(selected_row["candidate_group_value"])
        selected_group_values.append(selected_group_value)
        remaining_group_values = [value for value in remaining_group_values if value != selected_group_value]

        for row in candidate_rows:
            row["selected_group_value"] = selected_group_value
            row["selected_in_round"] = int(row["candidate_group_value"]) == selected_group_value
            row["selected_order"] = list(selected_group_values)

        round_rows.extend(candidate_rows)

    return {
        "dataset": ref_result.dataset,
        "model_name": run_model_name,
        "seed": int(seed),
        "generator_name": generator_name,
        "grouping_type": grouping_type,
        "rfi_mode": rfi_mode,
        "rfi_split": rfi_split,
        "baseline_split_mae": float(baseline_split_mae),
        "selected_order": selected_group_values,
        "rows": round_rows,
    }
