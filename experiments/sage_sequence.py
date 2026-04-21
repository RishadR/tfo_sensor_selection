from __future__ import annotations

from typing import Literal

import numpy as np
import sage

from tfo_sensor_selection.config import DatasetConfig, DatasetName
from tfo_sensor_selection.results.pretrained import load_pretrained_seed_contexts

EvolutionType = Literal["wavelength", "detector_distance"]
MAX_BACKGROUND_SAMPLES = 1024


def _build_feature_to_group(cfg: DatasetConfig, evolution_type: EvolutionType) -> dict[str, int]:
	values = cfg.wavelength if evolution_type == "wavelength" else cfg.detector_distances
	return {feature_name: int(values[idx]) for idx, feature_name in enumerate(cfg.features)}


def _compute_seed_feature_sage(
	x_train: np.ndarray,
	x_test: np.ndarray,
	y_test: np.ndarray,
	model_predict,
	n_permutations: int,
) -> np.ndarray:
	if len(x_train) > MAX_BACKGROUND_SAMPLES:
		rng = np.random.default_rng(0)
		indices = rng.choice(len(x_train), size=MAX_BACKGROUND_SAMPLES, replace=False)
		background = x_train[indices]
	else:
		background = x_train

	imputer = sage.MarginalImputer(model_predict, background)
	estimator = sage.PermutationEstimator(imputer, loss="mse")
	explanation = estimator(
		x_test,
		y_test,
		n_permutations=n_permutations,
		verbose=False,
		bar=False,
	)
	return np.asarray(explanation.values, dtype=float)


def _aggregate_group_scores(
	feature_scores: np.ndarray,
	feature_names: list[str],
	feature_to_group: dict[str, int],
) -> dict[int, float]:
	out: dict[int, float] = {}
	for idx, feature_name in enumerate(feature_names):
		group_value = feature_to_group[feature_name]
		out[group_value] = out.get(group_value, 0.0) + float(feature_scores[idx])
	return out


def _sorted_group_items(group_scores: dict[int, float]) -> list[tuple[int, float]]:
	return sorted(group_scores.items(), key=lambda item: item[1], reverse=True)


def _compute_rankings_for_setup(
	dataset_name: DatasetName,
	evolution_type: EvolutionType,
	n_permutations: int,
) -> None:
	contexts = load_pretrained_seed_contexts(
		dataset_name=dataset_name,
		model_name="mlp",
		seed_values=None,
		artifacts_dir="models/trained",
	)
	if len(contexts) == 0:
		raise RuntimeError(f"No pretrained contexts found for dataset={dataset_name}")

	first_seed = sorted(contexts.keys())[0]
	cfg = contexts[first_seed].config
	feature_to_group = _build_feature_to_group(cfg, evolution_type)

	all_seed_group_scores: list[dict[int, float]] = []
	all_groups = sorted(set(feature_to_group.values()))

	print("=" * 72)
	print(f"Dataset: {dataset_name} | Evolution: {evolution_type}")
	print(f"Seeds: {sorted(contexts.keys())}")

	for seed in sorted(contexts.keys()):
		ctx = contexts[seed]
		feature_scores = _compute_seed_feature_sage(
			x_train=ctx.x_train,
			x_test=ctx.x_test,
			y_test=ctx.y_test,
			model_predict=ctx.model.predict,
			n_permutations=n_permutations,
		)
		group_scores = _aggregate_group_scores(
			feature_scores=feature_scores,
			feature_names=ctx.feature_names,
			feature_to_group=feature_to_group,
		)
		all_seed_group_scores.append(group_scores)

	mean_scores: dict[int, float] = {}
	std_scores: dict[int, float] = {}
	for group_value in all_groups:
		values = np.array([scores.get(group_value, 0.0) for scores in all_seed_group_scores], dtype=float)
		mean_scores[group_value] = float(values.mean())
		std_scores[group_value] = float(values.std(ddof=0))

	ranked = _sorted_group_items(mean_scores)
	print("Group importance order (highest -> lowest):")
	for rank, (group_value, score) in enumerate(ranked, start=1):
		print(
			f"{rank:>2}. group={group_value}: mean_sum_sage={score:.8f} "
			f"(seed_std={std_scores[group_value]:.8f})"
		)


def main(n_permutations: int = 128) -> None:
	setups: list[tuple[DatasetName, EvolutionType]] = [
		("invivo", "wavelength"),
		("invivo", "detector_distance"),
		("simulation", "wavelength"),
		("simulation", "detector_distance"),
	]
	for dataset_name, evolution_type in setups:
		_compute_rankings_for_setup(
			dataset_name=dataset_name,
			evolution_type=evolution_type,
			n_permutations=n_permutations,
		)


if __name__ == "__main__":
	main()
