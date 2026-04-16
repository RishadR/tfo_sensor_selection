from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from arfpy import arf
from scipy.stats import truncnorm

from tfo_sensor_selection.synthetic.base import ConditionalSyntheticGenerator


class ARFGenerator(ConditionalSyntheticGenerator):
    PERSISTENCE_VERSION = 1

    def __init__(
        self,
        seed: int = 42,
        arf_params: dict | None = None,
        model_path: Path | None = None,
    ) -> None:
        """
        Initialize the ARFGenerator.

        Args:
            seed (int): Random seed for reproducibility.
            arf_params (dict | None): Parameters to pass directly to the ARF model. This includes (with defaults):
             - num_trees=30
             - delta=0
             - max_iters=10
             - early_stop=True
             - verbose=True
             - min_node_size=5
            model_path (Path | None): Optional path to a saved fitted ARFGenerator state.
        """

        self.seed = seed
        self.arf_params = dict(arf_params) if arf_params is not None else {}
        self._fit_columns: list[str] = []
        self._model = None
        self._psi = None
        self._rng = np.random.default_rng(seed)
        self._leaf_stats_cache: pd.DataFrame | None = None
        self._coverage_cache: np.ndarray | None = None
        self._feature_stats_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._tree_leaf_index_cache: dict[object, np.ndarray] = {}
        self._tree_ids_cache: np.ndarray | None = None

        if model_path is not None:
            self._load_model_from_path(model_path)

    def name(self) -> str:
        return "arf"

    def is_fitted(self) -> bool:
        return self._model is not None and self._psi is not None

    def _reset_generation_caches(self) -> None:
        self._leaf_stats_cache = None
        self._coverage_cache = None
        self._feature_stats_cache = {}
        self._tree_leaf_index_cache = {}
        self._tree_ids_cache = None

    def _get_leaf_stats(self) -> pd.DataFrame:
        if self._psi is None:
            raise RuntimeError("Generator must be fit before sampling")

        if self._leaf_stats_cache is None:
            psi_cnt = self._psi["cnt"]
            self._leaf_stats_cache = self._prepare_leaf_stats_table(psi_cnt=psi_cnt)
        return self._leaf_stats_cache

    def _get_tree_leaf_indices(self) -> tuple[np.ndarray | None, dict[object, np.ndarray]]:
        leaf_stats = self._get_leaf_stats()
        if "tree" not in leaf_stats.columns:
            return None, {}

        if self._tree_ids_cache is None or not self._tree_leaf_index_cache:
            tree_values = leaf_stats["tree"].to_numpy()
            tree_ids = np.unique(tree_values)
            self._tree_ids_cache = tree_ids
            self._tree_leaf_index_cache = {
                tree_id: np.flatnonzero(tree_values == tree_id) for tree_id in tree_ids
            }

        return self._tree_ids_cache, self._tree_leaf_index_cache

    def _get_coverage(self) -> np.ndarray:
        if self._coverage_cache is None:
            leaf_stats = self._get_leaf_stats()
            self._coverage_cache = self._resolve_coverage(psi_cnt=leaf_stats, n_leaves=len(leaf_stats))
        return self._coverage_cache

    def _get_feature_stats(
        self,
        feature_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        leaf_stats = self._get_leaf_stats()
        if feature_name not in self._feature_stats_cache:
            self._feature_stats_cache[feature_name] = self._resolve_feature_stats(
                psi_cnt=leaf_stats,
                feature_name=feature_name,
            )
        return self._feature_stats_cache[feature_name]

    def _precompute_feature_stats_cache(self) -> None:
        for feature_name in list(dict.fromkeys(self._fit_columns)):
            self._get_feature_stats(feature_name=feature_name)

    def save_model(self, path: str | Path) -> None:
        if not self.is_fitted():
            raise RuntimeError("Cannot save ARF model before fit")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": self.PERSISTENCE_VERSION,
            "generator": self.name(),
            "seed": int(self.seed),
            "arf_params": dict(self.arf_params),
            "fit_columns": list(self._fit_columns),
            "model": self._model,
            "psi": self._psi,
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load_model(cls, path: Path) -> "ARFGenerator":
        return cls(model_path=path)

    def _load_model_from_path(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError("model_path must be a pathlib.Path")

        path = model_path
        if not path.exists():
            raise FileNotFoundError(f"ARF model file does not exist: {path}")

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("ARF model file payload must be a dict")

        required = {"version", "generator", "seed", "arf_params", "fit_columns", "model", "psi"}
        missing = sorted(required - set(payload.keys()))
        if missing:
            raise KeyError(f"ARF model payload is missing keys: {missing}")

        version = payload["version"]
        if version != self.PERSISTENCE_VERSION:
            raise ValueError(
                f"Unsupported ARF model payload version {version}. "
                f"Expected {self.PERSISTENCE_VERSION}."
            )

        if payload["generator"] != self.name():
            raise ValueError(
                f"Model payload generator type '{payload['generator']}' does not match '{self.name()}'."
            )

        self.seed = int(payload["seed"])
        if not isinstance(payload["arf_params"], dict):
            raise ValueError("ARF model payload field 'arf_params' must be a dict")
        self.arf_params = dict(payload["arf_params"])

        fit_columns = payload["fit_columns"]
        if not isinstance(fit_columns, list):
            raise ValueError("ARF model payload field 'fit_columns' must be a list")
        self._fit_columns = [str(col) for col in fit_columns]

        self._model = payload["model"]
        self._psi = payload["psi"]
        self._rng = np.random.default_rng(self.seed)
        self._reset_generation_caches()
        self._precompute_feature_stats_cache()

        if not self.is_fitted():
            raise ValueError("Loaded ARF model payload did not contain a fitted model and psi state")

    @staticmethod
    def _resolve_coverage(psi_cnt: pd.DataFrame, n_leaves: int) -> np.ndarray:
        coverage_candidates = ["cvg", "coverage", "prob", "weight", "wt", "leaf_weight"]
        for col in coverage_candidates:
            if col in psi_cnt.columns:
                return psi_cnt[col].to_numpy(dtype=float)
        return np.full(n_leaves, 1.0 / max(n_leaves, 1), dtype=float)

    @staticmethod
    def _prepare_leaf_stats_table(psi_cnt: pd.DataFrame) -> pd.DataFrame:
        """Return a per-leaf wide table regardless of arfpy FORDE schema flavor."""
        required_long = {"variable", "mean", "sd", "min", "max"}
        if required_long.issubset(set(psi_cnt.columns)):
            leaf_keys = [c for c in ("tree", "nodeid") if c in psi_cnt.columns]
            if not leaf_keys:
                leaf_keys = ["leaf_id"]
                work = psi_cnt.copy()
                work["leaf_id"] = np.arange(len(work), dtype=int)
            else:
                work = psi_cnt

            out = work[leaf_keys].drop_duplicates().reset_index(drop=True)
            for stat in ("mean", "sd", "min", "max"):
                pivot = work.pivot_table(
                    index=leaf_keys,
                    columns="variable",
                    values=stat,
                    aggfunc="first",
                )
                pivot.columns = [f"{str(c)}_{stat}" for c in pivot.columns]
                out = out.merge(pivot.reset_index(), on=leaf_keys, how="left")

            for cov_col in ("cvg", "coverage", "prob", "weight", "wt", "leaf_weight"):
                if cov_col in work.columns:
                    cov = work.groupby(leaf_keys, as_index=False)[cov_col].first()
                    out = out.merge(cov, on=leaf_keys, how="left")
                    break

            return out

        return psi_cnt

    @staticmethod
    def _resolve_feature_stats(psi_cnt: pd.DataFrame, feature_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean_col = f"{feature_name}_mean"
        std_candidates = [f"{feature_name}_sd", f"{feature_name}_std"]

        std_col = next((c for c in std_candidates if c in psi_cnt.columns), None)
        min_col = f"{feature_name}_min" if f"{feature_name}_min" in psi_cnt.columns else None
        max_col = f"{feature_name}_max" if f"{feature_name}_max" in psi_cnt.columns else None

        means = psi_cnt[mean_col].to_numpy(dtype=float)
        stds = np.clip(psi_cnt[std_col].to_numpy(dtype=float), 1e-8, None)
        mins = psi_cnt[min_col].to_numpy(dtype=float)
        maxs = psi_cnt[max_col].to_numpy(dtype=float)
        return means, stds, mins, maxs

    def fit(
        self,
        train_df: pd.DataFrame,
        fit_columns: list[str],
    ) -> "ARFGenerator":
        self._fit_columns = fit_columns

        table = train_df[self._fit_columns].copy()
        # Create an unconditional root model by fitting the entire dataset
        model = arf.arf(x=table, **self.arf_params)
        psi = model.forde()
        self._model = model
        self._psi = psi
        self._reset_generation_caches()
        self._precompute_feature_stats_cache()
        return self

    def generate_conditional(
        self,
        conditional_df: pd.DataFrame,
        conditional_column_names: list[str],
        target_column_names: list[str],
    ) -> pd.DataFrame:
        if self._model is None or self._psi is None:
            raise RuntimeError("Generator must be fit before sampling")

        if len(conditional_df.index) == 0:
            return pd.DataFrame(columns=target_column_names)

        missing = [col for col in conditional_column_names if col not in conditional_df.columns]
        if missing:
            raise KeyError(f"conditional_df is missing conditional columns: {missing}")

        leaf_stats = self._get_leaf_stats()
        n_rows = len(conditional_df)

        tree_ids, tree_leaf_index_map = self._get_tree_leaf_indices()
        if tree_ids is not None and len(tree_ids) > 0:
            selected_tree_id = self._rng.choice(tree_ids)
            active_leaf_indices = tree_leaf_index_map[selected_tree_id]
        else:
            active_leaf_indices = np.arange(len(leaf_stats))

        n_leaves = len(active_leaf_indices)

        if n_leaves == 0:
            raise RuntimeError("ARF FORDE output has no continuous leaf statistics")

        coverage = self._get_coverage()[active_leaf_indices]
        log_weights = np.log(np.clip(coverage, 1e-10, None))
        log_weights = np.tile(log_weights, (n_rows, 1))

        for col in conditional_column_names:
            means, stds, mins, maxs = self._get_feature_stats(feature_name=col)
            means = means[active_leaf_indices]
            stds = stds[active_leaf_indices]
            mins = mins[active_leaf_indices]
            maxs = maxs[active_leaf_indices]
            observed = conditional_df[col].to_numpy(dtype=float)[:, np.newaxis]

            a = (mins - means) / stds
            b = (maxs - means) / stds
            log_weights += truncnorm.logpdf(observed, a, b, loc=means, scale=stds)

        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        probs = np.exp(log_weights)
        row_sums = probs.sum(axis=1, keepdims=True)
        invalid = ~np.isfinite(row_sums) | (row_sums <= 0)
        if np.any(invalid):
            probs[invalid[:, 0]] = 1.0 / n_leaves
            row_sums = probs.sum(axis=1, keepdims=True)
        probs /= row_sums

        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0
        uniforms = self._rng.random((n_rows, 1))
        sampled_leaf_indices_local = np.argmax(cdf >= uniforms, axis=1)
        sampled_leaf_indices = active_leaf_indices[sampled_leaf_indices_local]

        generated_values = np.empty((n_rows, len(target_column_names)), dtype=float)
        for idx, target in enumerate(target_column_names):
            means, stds, mins, maxs = self._get_feature_stats(feature_name=target)
            means = means[sampled_leaf_indices]
            stds = stds[sampled_leaf_indices]
            mins = mins[sampled_leaf_indices]
            maxs = maxs[sampled_leaf_indices]

            a = (mins - means) / stds
            b = (maxs - means) / stds
            generated_values[:, idx] = truncnorm.rvs(a, b, loc=means, scale=stds, random_state=self._rng)
        return pd.DataFrame(generated_values, index=conditional_df.index, columns=target_column_names)
