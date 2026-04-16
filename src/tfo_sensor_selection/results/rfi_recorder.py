from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RFI_COLUMNS = [
    "timestamp",
    "dataset",
    "model_name",
    "seed",
    "generator_name",
    "grouping_type",
    "rfi_mode",
    "rfi_split",
    "round_index",
    "candidate_group_value",
    "candidate_group_features",
    "conditioned_group_values",
    "conditioned_features",
    "synthesized_split_mae",
    "baseline_split_mae",
    "rfi_mean",
    "rfi_variance",
    "selected_group_value",
    "selected_in_round",
    "selected_order",
    "train_mae",
    "val_mae",
    "test_mae",
]

RFI_PER_SAMPLE_COLUMNS = [
    "timestamp",
    "dataset",
    "model_name",
    "seed",
    "generator_name",
    "grouping_type",
    "rfi_mode",
    "rfi_split",
    "round_index",
    "candidate_group_value",
    "selected_group_value",
    "selected_in_round",
    "sample_index",
    "baseline_absolute_error",
    "synthesized_absolute_error",
    "per_sample_rfi",
]


def _serialize(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return value


def record_rfi_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path = "results/rfi_greedy_results.csv",
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = {"timestamp": timestamp}
        for key in RFI_COLUMNS:
            if key == "timestamp":
                continue
            normalized[key] = _serialize(row.get(key))
        normalized_rows.append(normalized)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RFI_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)


def record_rfi_per_sample_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path = "results/rfi_greedy_per_sample.csv",
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    expanded_rows: list[dict[str, Any]] = []
    for row in rows:
        per_sample_rfi = row.get("per_sample_rfi")
        baseline_absolute_errors = row.get("baseline_absolute_errors")
        synthesized_absolute_errors = row.get("synthesized_absolute_errors")
        if per_sample_rfi is None or baseline_absolute_errors is None or synthesized_absolute_errors is None:
            continue

        for sample_index, value in enumerate(per_sample_rfi):
            expanded_rows.append(
                {
                    "timestamp": timestamp,
                    "dataset": row.get("dataset"),
                    "model_name": row.get("model_name"),
                    "seed": row.get("seed"),
                    "generator_name": row.get("generator_name"),
                    "grouping_type": row.get("grouping_type"),
                    "rfi_mode": row.get("rfi_mode"),
                    "rfi_split": row.get("rfi_split"),
                    "round_index": row.get("round_index"),
                    "candidate_group_value": row.get("candidate_group_value"),
                    "selected_group_value": row.get("selected_group_value"),
                    "selected_in_round": row.get("selected_in_round"),
                    "sample_index": sample_index,
                    "baseline_absolute_error": baseline_absolute_errors[sample_index],
                    "synthesized_absolute_error": synthesized_absolute_errors[sample_index],
                    "per_sample_rfi": value,
                    "rfi_mean": row.get("rfi_mean"),
                    "rfi_variance": row.get("rfi_variance"),
                }
            )

    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RFI_PER_SAMPLE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(expanded_rows)
