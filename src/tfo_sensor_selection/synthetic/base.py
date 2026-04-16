from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ConditionalSyntheticGenerator(ABC):
    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        fit_columns: list[str],
    ) -> "ConditionalSyntheticGenerator":
        """
        Fit the given dataset for the fit_columns. (Ignores the rest of the columns)
        """
        raise NotImplementedError

    @abstractmethod
    def generate_conditional(
        self,
        conditional_df: pd.DataFrame,
        conditional_column_names: list[str],
        target_column_names: list[str],
    ) -> pd.DataFrame:
        """
        Generate target columns conditioned on the provided conditional columns.
        
        Args: 
            conditional_df: Contains the conditional values
            conditional_column_names: The column names in conditional_df to condition on - these values remain unchanged
            target_column_names: The column names to fake data for - these are the columns that will be synthesized
        Returns:
            the faked target columns as a DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
