from __future__ import annotations

import pandas as pd

from tfo_sensor_selection.synthetic.base import ConditionalSyntheticGenerator


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _resolve_column_roles(
    all_column_names: list[str],
    conditional_column_names: list[str],
    target_column_names: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    all_columns = _dedupe_preserve_order(all_column_names)
    conditional_columns = _dedupe_preserve_order(conditional_column_names)
    target_columns = _dedupe_preserve_order(target_column_names)

    unknown = [col for col in conditional_columns + target_columns if col not in all_columns]
    if unknown:
        raise KeyError(f"Unknown columns requested for synthetic generation: {unknown}")

    overlap = sorted(set(conditional_columns).intersection(target_columns))
    if overlap:
        raise ValueError(f"Conditional and target columns must be disjoint, overlap: {overlap}")

    unaffected_columns = [
        col for col in all_columns if col not in conditional_columns and col not in target_columns
    ]
    return all_columns, conditional_columns, target_columns, unaffected_columns


def generate_missing_features(
    generator: ConditionalSyntheticGenerator,
    train_df: pd.DataFrame,
    source_df: pd.DataFrame,
    all_column_names: list[str],
    conditional_column_names: list[str],
    target_column_names: list[str],
    fit: bool = True,
) -> pd.DataFrame:
    """
    Generate fake features in source_df using the provided generator, which is fit on train_df.
    
    Args:
        generator: A ConditionalSyntheticGenerator to use for generation.
        train_df: DataFrame to fit the generator on (if fit=True).
        source_df: DataFrame containing the conditional columns and any existing target columns.
        all_column_names: List of all column names that should be in the output DataFrame.
        conditional_column_names: List of column names to use as input to the generator.
        target_column_names: List of column names that the generator should produce.
        fit: Whether to fit the generator on train_df before generating.
    
    Returns:
        A DataFrame with the generated features added.

    """
    
    all_columns, conditional_columns, target_columns, _ = _resolve_column_roles(
        all_column_names=all_column_names,
        conditional_column_names=conditional_column_names,
        target_column_names=target_column_names,
    )

    missing_source_columns = [col for col in all_columns if col not in source_df.columns]
    if missing_source_columns:
        raise KeyError(f"source_df must include all requested columns: {missing_source_columns}")

    if not target_columns:
        return source_df[all_columns].copy()

    if fit:
        generator.fit(
            train_df=train_df,
            fit_columns=conditional_columns + target_columns,
        )
    generated_targets = generator.generate_conditional(
        conditional_df=source_df[conditional_columns].copy(),
        conditional_column_names=conditional_columns,
        target_column_names=target_columns,
    )

    missing_generated_columns = [col for col in target_columns if col not in generated_targets.columns]
    if missing_generated_columns:
        raise KeyError(f"Generator output is missing target columns: {missing_generated_columns}")

    out = source_df[all_columns].copy()
    for col in target_columns:
        out[col] = generated_targets[col]

    return out
