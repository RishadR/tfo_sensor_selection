import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from tfo_sensor_selection.config import DatasetConfig, DatasetName, load_metadata
from tfo_sensor_selection.data.loader import load_dataset
from tfo_sensor_selection.data.splitter import holdout_split
from tfo_sensor_selection.models import ModelName
from tfo_sensor_selection.results import load_trained_model_artifact
from tfo_sensor_selection.training.trainer import compute_mae


EvolutionType = Literal["wavelength", "detector_distance"]


@dataclass(frozen=True)
class SeedEvaluationContext:
	seed: int
	n_val_groups: int
	n_test_groups: int
	feature_names: list[str]
	label_name: str
	transform: Any
	model: Any
	train_df: pd.DataFrame
	val_df: pd.DataFrame
	test_df: pd.DataFrame
	y_train: np.ndarray
	y_val: np.ndarray
	y_test: np.ndarray


def _to_xy(df: pd.DataFrame, features: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
	x = df[features].to_numpy(dtype=float)
	y = df[label].to_numpy(dtype=float)
	return x, y


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


def _mask_uninformative_features(
	df: pd.DataFrame,
	all_features: list[str],
	chosen_features: list[str],
	feature_means: dict[str, float],
) -> pd.DataFrame:
	"""Replace every feature not in chosen_features with its training-set mean."""
	df = df.copy()
	chosen_set = set(chosen_features)
	for feature in all_features:
		if feature not in chosen_set:
			df[feature] = feature_means[feature]
	return df


def record_error_evolution_alt_result(
	results_df: pd.DataFrame,
	output_path: str | Path,
) -> None:
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(out, index=False)


def _discover_seeds_from_artifacts(
	dataset_name: DatasetName,
	model_name: ModelName,
	artifacts_dir: str | Path,
) -> list[int]:
	base = Path(artifacts_dir)
	pattern = re.compile(
		rf"^{re.escape(str(dataset_name))}_{re.escape(str(model_name))}_seed(\d+)\.pkl$"
	)
	seeds = []
	for path in base.glob(f"{dataset_name}_{model_name}_seed*.pkl"):
		match = pattern.match(path.name)
		if match:
			seeds.append(int(match.group(1)))
	if not seeds:
		raise FileNotFoundError(
			f"No trained model artifacts found for dataset='{dataset_name}', "
			f"model='{model_name}' in '{base}'"
		)
	return sorted(seeds)


def _artifact_path_for_seed(
	seed_value: int,
	dataset_name: DatasetName,
	model_name: ModelName,
	artifacts_dir: str | Path,
) -> Path:
	return Path(artifacts_dir) / f"{dataset_name}_{model_name}_seed{seed_value}.pkl"


def _load_seed_context(
	seed_value: int,
	dataset_name: DatasetName,
	model_name: ModelName,
	artifacts_dir: str | Path,
	cfg: DatasetConfig,
	df: pd.DataFrame,
) -> SeedEvaluationContext:
	artifact_path = _artifact_path_for_seed(
		seed_value=seed_value,
		dataset_name=dataset_name,
		model_name=model_name,
		artifacts_dir=artifacts_dir,
	)
	artifact = load_trained_model_artifact(artifact_path)

	if artifact.dataset != str(dataset_name):
		raise ValueError(
			f"Artifact dataset mismatch for seed {seed_value}: "
			f"expected {dataset_name}, found {artifact.dataset}"
		)
	if artifact.model_name != model_name:
		raise ValueError(
			f"Artifact model mismatch for seed {seed_value}: "
			f"expected {model_name}, found {artifact.model_name}"
		)
	if artifact.seed != int(seed_value):
		raise ValueError(
			f"Artifact seed mismatch for seed {seed_value}: "
			f"expected {seed_value}, found {artifact.seed}"
		)

	n_val_groups = artifact.n_val_groups
	n_test_groups = artifact.n_test_groups

	train_df, val_df, test_df = holdout_split(
		df=df,
		grouping_col=cfg.grouping_col,
		n_val_groups=n_val_groups,
		n_test_groups=n_test_groups,
		val_start_idx=seed_value,
		ignored_group=cfg.ignored_group,
	)

	feature_names = list(artifact.feature_names)
	label_name = str(artifact.label_name)

	_, y_train = _to_xy(train_df, feature_names, label_name)
	_, y_val = _to_xy(val_df, feature_names, label_name)
	_, y_test = _to_xy(test_df, feature_names, label_name)

	return SeedEvaluationContext(
		seed=seed_value,
		n_val_groups=n_val_groups,
		n_test_groups=n_test_groups,
		feature_names=feature_names,
		label_name=label_name,
		transform=artifact.transform,
		model=artifact.model,
		train_df=train_df,
		val_df=val_df,
		test_df=test_df,
		y_train=y_train,
		y_val=y_val,
		y_test=y_test,
	)


def load_pretrained_seed_contexts(
	dataset_name: DatasetName,
	model_name: ModelName,
	artifacts_dir: str | Path = "models/trained",
) -> dict[int, SeedEvaluationContext]:
	seeds = _discover_seeds_from_artifacts(
		dataset_name=dataset_name,
		model_name=model_name,
		artifacts_dir=artifacts_dir,
	)
	cfg = load_metadata(dataset_name)
	df = load_dataset(cfg)
	contexts: dict[int, SeedEvaluationContext] = {}
	for seed_value in seeds:
		contexts[seed_value] = _load_seed_context(
			seed_value=seed_value,
			dataset_name=dataset_name,
			model_name=model_name,
			artifacts_dir=artifacts_dir,
			cfg=cfg,
			df=df,
		)
	return contexts


def run_error_evolution_alt(
	dataset_name: DatasetName,
	evolution_type: EvolutionType,
	sequence: list[int],
	model_name: ModelName,
	method_name: str,
	selection_strategy: str,
	n_trials: int = 20,
	record_csv: bool = True,
	output_path: str | Path = "results/error_evolution.csv",
	pretrained_results: dict[int, SeedEvaluationContext] | None = None,
	artifacts_dir: str | Path = "models/trained",
) -> pd.DataFrame:
	"""
	Discover pre-trained model artifacts for the given dataset and model, then evaluate
	each stage of the sequence by replacing uninformative features with their training mean.
	Seeds are inferred automatically from the filenames in artifacts_dir.
	"""
	cfg = load_metadata(dataset_name)
	grouped = _build_group_to_features(cfg, evolution_type=evolution_type)
	resolved_sequence = _resolve_sequence(list(grouped.keys()), sequence)

	all_rows: list[dict[str, Any]] = []

	# --- Load one model artifact per seed, or reuse provided contexts ---
	if pretrained_results is not None:
		seed_contexts = pretrained_results
	else:
		seed_contexts = load_pretrained_seed_contexts(
			dataset_name=dataset_name,
			model_name=model_name,
			artifacts_dir=artifacts_dir,
		)
	# --- All models loaded; proceed with stage evaluation ---

	for seed_value in sorted(seed_contexts):
		seed_context = seed_contexts[seed_value]
		all_features = list(seed_context.feature_names)

		# Compute per-feature training means for masking
		feature_means = {f: float(seed_context.train_df[f].mean()) for f in all_features}

		# Evaluate each stage with uninformative features masked
		for step_idx in range(len(resolved_sequence)):
			chosen_values = resolved_sequence[: step_idx + 1]
			chosen_features = _flatten_features(chosen_values, grouped)

			train_masked = _mask_uninformative_features(
				seed_context.train_df, all_features, chosen_features, feature_means
			)
			val_masked = _mask_uninformative_features(
				seed_context.val_df, all_features, chosen_features, feature_means
			)
			test_masked = _mask_uninformative_features(
				seed_context.test_df, all_features, chosen_features, feature_means
			)

			x_train = seed_context.transform.transform(train_masked[all_features].to_numpy(dtype=float))
			x_val = seed_context.transform.transform(val_masked[all_features].to_numpy(dtype=float))
			x_test = seed_context.transform.transform(test_masked[all_features].to_numpy(dtype=float))

			train_mae = compute_mae(seed_context.model, x_train, seed_context.y_train)
			val_mae = compute_mae(seed_context.model, x_val, seed_context.y_val)
			test_mae = compute_mae(seed_context.model, x_test, seed_context.y_test)

			all_rows.append(
				{
					"evolution_type": str(evolution_type),
					"dataset_name": str(dataset_name),
					"current_sequence": str(chosen_values),
					"seed": int(seed_value),
					"model_name": str(model_name),
					"n_val_groups": int(seed_context.n_val_groups),
					"n_test_groups": int(seed_context.n_test_groups),
					"n_trials": int(n_trials),
					"train_error": train_mae,
					"val_error": val_mae,
					"test_error": test_mae,
				}
			)

	results_df = pd.DataFrame(all_rows)
	results_df["method_name"] = method_name
	results_df["selection_strategy"] = selection_strategy
	results_df["timestamp"] = pd.Timestamp.now()
	results_df = results_df[
		[
			"timestamp",
			"method_name",
			"selection_strategy",
			"evolution_type",
			"dataset_name",
			"current_sequence",
			"seed",
			"model_name",
			"n_val_groups",
			"n_test_groups",
			"n_trials",
			"train_error",
			"val_error",
			"test_error",
		]
	]

	if record_csv:
		record_error_evolution_alt_result(results_df=results_df, output_path=output_path)

	return results_df


if __name__ == "__main__":
	SEQUENCES_PATH = Path("data/sequences.yaml")
	DATASET_KEY_MAP: dict[str, DatasetName] = {
		"simulation": "simulation",
		"invivo": "invivo",
	}
	MODEL_NAME = "mlp"
	N_TRIALS = 30
	ARTIFACTS_DIR = Path("models/trained")

	with SEQUENCES_PATH.open("r") as fh:
		sequences_cfg = yaml.safe_load(fh)

	all_results: list[pd.DataFrame] = []

	for yaml_dataset_key, dataset_name in ['simulation', 'invivo']:
		for evolution_type, strategies in sequences_cfg[yaml_dataset_key].items():
			# Load once per (dataset, evolution_type) combo — seeds discovered from disk
			print(f"Loading models for {dataset_name} / {evolution_type} ...")
			shared_models = load_pretrained_seed_contexts(
				dataset_name=dataset_name,
				model_name=MODEL_NAME,
				artifacts_dir=ARTIFACTS_DIR,
			)
			print(f"  Found seeds: {sorted(shared_models)}")

			for strategy_name, sequence in strategies.items():
				print(f"  Evaluating {strategy_name} ...")
				result_df = run_error_evolution_alt(
					dataset_name=dataset_name,
					evolution_type=evolution_type,
					model_name=MODEL_NAME,
					method_name="MeanMask",
					selection_strategy=strategy_name,
					sequence=[int(v) for v in sequence],
					n_trials=N_TRIALS,
					artifacts_dir=ARTIFACTS_DIR,
					record_csv=False,
					pretrained_results=shared_models,
				)
				all_results.append(result_df)

	combined = pd.concat(all_results, ignore_index=True)
	record_error_evolution_alt_result(results_df=combined, output_path="results/error_evolution_alt.csv")
