import numpy as np
import pandas as pd


class MislabelsGenerator:
    def __init__(self, seed):
        self.seed = seed

    def _validate_input(self, target_column_series):
        if target_column_series.dtype != 'int64':
            raise ValueError("Target column values must be in the int64 format")

        allowed_values = (0, 1)
        for val in target_column_series.unique():
            if val not in allowed_values:
                raise ValueError(f"Current value is {val}. Target column values must be in {allowed_values}")

    def fit(self, df, target_column):
        self._validate_input(df[target_column])

    def transform(self, df: pd.DataFrame, target_column, mislabels_percentage: float):
        if mislabels_percentage == 0.0:
            return df

        df_copy = df.copy(deep=True)
        np.random.seed(self.seed)
        mislabels_sample_size = int(df_copy.shape[0] * mislabels_percentage)
        random_row_idxs = np.random.choice(df_copy.index, size=mislabels_sample_size, replace=False)
        df_copy.loc[random_row_idxs, target_column] = 1 - df_copy.loc[random_row_idxs, target_column]

        return df_copy

    def fit_transform(self, df, target_column, mislabels_percentage):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column, mislabels_percentage)
        return transformed_df
