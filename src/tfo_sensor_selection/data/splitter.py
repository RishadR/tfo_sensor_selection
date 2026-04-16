from typing import Tuple, Any

import pandas as pd


def holdout_split(
    df: pd.DataFrame,
    grouping_col: str,
    n_val_groups: int,
    n_test_groups: int,
    val_start_idx: int = 0,
    ignored_group: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train/val/test sets based on group membership in the specified grouping column.
    The groups are sorted and then assigned to val/test in a round-robin fashion starting from an index determined by
    the val_start_idx. This ensures a deterministic split that can be reproduced with the same val_start_idx, while 
    also allowing for different splits by changing the val_start_idx. The remaining groups that are not assigned to 
    val/test are used for training. 
    
    The function checks to ensure that the specified number of groups for val/test does not exceed the total number of
    groups available, and that the resulting splits are not empty.
    """
    
    if grouping_col not in df.columns:
        raise KeyError(f"Grouping column not found: {grouping_col}")

    if n_val_groups < 0 or n_test_groups < 0:
        raise ValueError("n_val_groups and n_test_groups must both be >= 0")
    
    if ignored_group is not None:
        df = pd.DataFrame(df[df[grouping_col] != ignored_group])
    
    groups = sorted(df[grouping_col].dropna().unique().tolist())
    n_groups = len(groups)
    if n_groups <= n_val_groups + n_test_groups:
        raise ValueError(
            f"Not enough groups for split: total={n_groups}, "
            f"val={n_val_groups}, test={n_test_groups}."
        )

    start_idx = val_start_idx % n_groups
    val_groups = {
        groups[(start_idx + offset) % n_groups]
        for offset in range(n_val_groups)
    }
    test_start_idx = (start_idx + n_val_groups) % n_groups
    test_groups = {
        groups[(test_start_idx + offset) % n_groups]
        for offset in range(n_test_groups)
    }

    test_mask = df[grouping_col].isin(test_groups)
    val_mask = df[grouping_col].isin(val_groups)
    train_mask = ~(test_mask | val_mask)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty:
        raise RuntimeError("Split produced an empty train partition; adjust group counts")
    if n_val_groups > 0 and val_df.empty:
        raise RuntimeError("Split produced an empty val partition; adjust group counts")
    if n_test_groups > 0 and test_df.empty:
        raise RuntimeError("Split produced an empty test partition; adjust group counts")

    return train_df, val_df, test_df
