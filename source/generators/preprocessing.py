import numpy as np
import pandas as pd


class RandomNullsGenerator:
    def __init__(self, seed: int, columns_nulls_percentage_dct: dict):
        self.seed = seed
        self.columns_nulls_percentage_dct = columns_nulls_percentage_dct

    def _validate_input(self, df):
        for col in self.columns_nulls_percentage_dct.keys():
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_nulls_percentage_dct must be the dataframe column names")

    def fit(self, df, target_column: str = None):
        self._validate_input(df)

    def transform(self, df: pd.DataFrame, target_column: str = None):
        df_copy = df.copy(deep=True)
        for idx, (col_name, nulls_pct) in enumerate(self.columns_nulls_percentage_dct.items()):
            if nulls_pct == 0.0:
                continue

            np.random.seed(self.seed + idx)
            nulls_sample_size = int(df_copy.shape[0] * nulls_pct)
            random_row_idxs = np.random.choice(df_copy.index, size=nulls_sample_size, replace=False)
            df_copy.loc[random_row_idxs, col_name] = np.nan

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column)
        return transformed_df
