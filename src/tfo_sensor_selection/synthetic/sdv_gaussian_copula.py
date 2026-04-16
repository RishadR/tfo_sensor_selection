from __future__ import annotations

import importlib

import pandas as pd

from tfo_sensor_selection.synthetic.base import ConditionalSyntheticGenerator


class SDVGaussianCopulaGenerator(ConditionalSyntheticGenerator):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._synthesizer = None
        self._fit_columns: list[str] = []

    def name(self) -> str:
        return "gaussian_copula"

    def fit(
        self,
        train_df: pd.DataFrame,
        fit_columns: list[str],
    ) -> "SDVGaussianCopulaGenerator":
        sdv_single_table = importlib.import_module("sdv.single_table")
        sdv_metadata_mod = importlib.import_module("sdv.metadata")

        GaussianCopulaSynthesizer = sdv_single_table.GaussianCopulaSynthesizer
        Metadata = sdv_metadata_mod.Metadata

        self._fit_columns = list(fit_columns)
        train_table = train_df[self._fit_columns].copy()

        metadata = Metadata.detect_from_dataframe(train_table, table_name="table")

        synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
        synthesizer.fit(train_table)
        self._synthesizer = synthesizer
        return self

    def generate_conditional(
        self,
        conditional_df: pd.DataFrame,
        conditional_column_names: list[str],
        target_column_names: list[str],
    ) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("Generator must be fit before sampling")

        sdv_sampling = importlib.import_module("sdv.sampling")
        Condition = sdv_sampling.Condition

        if len(conditional_df.index) == 0:
            return pd.DataFrame(columns=target_column_names)

        missing = [col for col in conditional_column_names if col not in conditional_df.columns]
        if missing:
            raise KeyError(f"conditional_df is missing conditional columns: {missing}")

        conditions = []
        conditioned_rows = conditional_df[conditional_column_names]
        for _, row in conditioned_rows.iterrows():
            conditions.append(Condition(num_rows=1, column_values=row.to_dict()))

        sampled = self._synthesizer.sample_from_conditions(conditions=conditions)

        missing_targets = [c for c in target_column_names if c not in sampled.columns]
        if missing_targets:
            raise KeyError(f"SDV output missing target columns: {missing_targets}")

        sampled = sampled[target_column_names].copy()
        sampled.index = conditional_df.index
        return sampled
