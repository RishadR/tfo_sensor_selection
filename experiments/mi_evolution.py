from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import infomeasure as im
import numpy as np
import yaml

from tfo_sensor_selection.config import DatasetConfig, DatasetName, load_metadata
from tfo_sensor_selection.data.loader import load_dataset


EvolutionType = Literal["wavelength", "detector_distance"]
SEQUENCE_DATASET_TO_METADATA_DATASET: dict[str, DatasetName] = {
	"invivo": "invivo",
	"simulated": "simulation",
}


def _build_group_to_features(cfg: DatasetConfig, evolution_type: EvolutionType) -> dict[int, list[str]]:
	if evolution_type == "wavelength":
		values = cfg.wavelength
	else:
		values = cfg.detector_distances

	grouped: dict[int, list[str]] = {}
	for idx, feature_name in enumerate(cfg.features):
		key = int(values[idx])
		grouped.setdefault(key, []).append(feature_name)

	return grouped


def _resolve_sequence(available_group_values: list[int], requested_sequence: list[int]) -> list[int]:
	available = set(available_group_values)
	sequence: list[int] = []
	seen: set[int] = set()
	for value in requested_sequence:
		value = int(value)
		if value not in available:
			raise ValueError(
				f"Invalid sequence value {value}. Expected one of: {sorted(available)}"
			)
		if value in seen:
			raise ValueError(f"Duplicate sequence value detected: {value}")
		seen.add(value)
		sequence.append(value)

	if len(sequence) == 0:
		raise ValueError("sequence must contain at least one value")
	return sequence


def _flatten_features(sequence_prefix: list[int], grouped: dict[int, list[str]]) -> list[str]:
	selected: list[str] = []
	seen: set[str] = set()
	for group_value in sequence_prefix:
		for feature_name in grouped[group_value]:
			if feature_name not in seen:
				selected.append(feature_name)
				seen.add(feature_name)
	return selected


def _load_sequence_config(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle)

	if not isinstance(payload, dict):
		raise ValueError("Expected sequences.yaml to contain a top-level mapping")
	return payload


def _experiment_id(index: int) -> str:
	return f"mi_evolution_{index:04d}"


def _compute_mutual_information_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | list[float]]:
	estimator = im.estimator(
		x,
		y,
		measure="mutual_information",
		approach="ksg",
		normalize=False,
	)
	test_result = estimator.statistical_test(n_tests=20, method="bootstrap")
	ci_95 = test_result.confidence_interval(95)

	return {
		"mi": float(estimator.result()),
		"p_value": float(test_result.p_value),
		"t_score": float(test_result.t_score),
		"confidence_interval_95_lower": float(ci_95[0]),
		"confidence_interval_95_upper": float(ci_95[1]),
	}


def write_mi_evolution_results(
	results: list[dict[str, Any]],
	output_path: str | Path = "results/mi_evolution.yaml",
) -> None:
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	experiments: list[dict[str, Any]] = []
	for index, result in enumerate(results, start=1):
		experiments.append(
			{
				"experiment_id": _experiment_id(index),
				"timestamp": datetime.now(tz=timezone.utc).isoformat(),
				**result,
			}
		)

	with out.open("w", encoding="utf-8") as handle:
		yaml.safe_dump(
			{"experiments": experiments},
			handle,
			sort_keys=False,
			allow_unicode=False,
		)


def run_mi_evolution(
	dataset_name: DatasetName,
	evolution_type: EvolutionType,
	sequence: list[int],
	method: str,
) -> dict[str, Any]:
	cfg = load_metadata(dataset_name)
	grouped = _build_group_to_features(cfg, evolution_type=evolution_type)
	resolved_sequence = _resolve_sequence(list(grouped.keys()), sequence)

	df = load_dataset(cfg)
	label_name = cfg.labels[0]
	y = df[label_name].to_numpy(dtype=float)

	step_rows: list[dict[str, Any]] = []
	for step_idx in range(len(resolved_sequence)):
		selected_values = resolved_sequence[: step_idx + 1]
		selected_features = _flatten_features(selected_values, grouped)
		x = df[selected_features].to_numpy(dtype=float)
		mi_stats = _compute_mutual_information_stats(x, y)

		step_rows.append(
			{
				"step_index": step_idx + 1,
				"included_values": selected_values,
				**mi_stats,
			}
		)

	result: dict[str, Any] = {
		"dataset": dataset_name,
		"evolution_type": evolution_type,
		"method": method,
		"sequence": resolved_sequence,
		"label_name": label_name,
		"steps": step_rows,
	}

	return result


def run_all_mi_evolution(
	sequences_path: str | Path = "data/sequences.yaml",
	output_path: str | Path = "results/mi_evolution.yaml",
) -> list[dict[str, Any]]:
	root = Path(__file__).resolve().parents[1]
	sequence_config = _load_sequence_config(root / sequences_path)
	results: list[dict[str, Any]] = []

	for sequence_dataset_name, grouping_config in sequence_config.items():
		dataset_name = SEQUENCE_DATASET_TO_METADATA_DATASET[sequence_dataset_name]

		for evolution_type, method_config in grouping_config.items():
			for method, sequence in method_config.items():
				results.append(
					run_mi_evolution(
						dataset_name=dataset_name,
						evolution_type=evolution_type,
						sequence=[int(value) for value in sequence],
						method=str(method),
					)
				)

	write_mi_evolution_results(results=results, output_path=root / output_path)
	for index, result in enumerate(results, start=1):
		result["experiment_id"] = _experiment_id(index)
	return results


if __name__ == "__main__":
	outputs = run_all_mi_evolution()
	print("Experiment: mi_evolution")
	print(f"Generated experiments: {len(outputs)}")
	for output in outputs:
		print(
			f"{output['experiment_id']}: dataset={output['dataset']}, "
			f"evolution_type={output['evolution_type']}, method={output['method']}"
		)