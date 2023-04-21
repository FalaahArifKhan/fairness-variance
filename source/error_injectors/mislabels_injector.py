import numpy as np
import pandas as pd

from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class MislabelsInjector(AbstractErrorInjector):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    mislabels_percentage
        Mislabels percentage for the target column

    """
    def __init__(self, seed: int, mislabels_percentage: float):
        super().__init__(seed)
        self.mislabels_percentage = mislabels_percentage

    def _validate_input(self, target_column_series):
        if target_column_series.dtype != 'int64':
            raise ValueError("Target column values must be in the int64 format")

        allowed_values = (0, 1)
        for val in target_column_series.unique():
            if val not in allowed_values:
                raise ValueError(f"Value caused the issue is {val}. Target column values must be in {allowed_values}")

    def fit(self, df, target_column: str = None):
        self._validate_input(df[target_column])

    def transform(self, df: pd.DataFrame, target_column: str = None):
        df_copy = df.copy(deep=True)
        if self.mislabels_percentage == 0.0:
            return df_copy

        np.random.seed(self.seed)
        mislabels_sample_size = int(df_copy.shape[0] * self.mislabels_percentage)
        random_row_idxs = np.random.choice(df_copy.index, size=mislabels_sample_size, replace=False)
        df_copy.loc[random_row_idxs, target_column] = 1 - df_copy.loc[random_row_idxs, target_column]

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column)
        return transformed_df
