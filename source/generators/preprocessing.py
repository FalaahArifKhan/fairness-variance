import numpy as np
import pandas as pd

from source.generators.abstract_generator import AbstractGenerator


class RandomNullsGenerator(AbstractGenerator):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    columns_nulls_percentage_dct
        Dictionary where keys are column names and values are target percentages of nulls for a column

    """
    def __init__(self, seed: int, columns_nulls_percentage_dct: dict):
        super().__init__(seed)
        self.columns_nulls_percentage_dct = columns_nulls_percentage_dct

    def _validate_input(self, df):
        for col, col_nulls_pct in self.columns_nulls_percentage_dct.items():
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_nulls_percentage_dct must be the dataframe column names")
            if col_nulls_pct < 0 or col_nulls_pct > 1:
                raise ValueError(f"Value caused the issue is {col_nulls_pct}. "
                                 f"Column nulls percentage must be in [0.0-1.0] range.")

    def fit(self, df, target_column: str = None):
        self._validate_input(df)

    def transform(self, df: pd.DataFrame, target_column: str = None):
        df_copy = df.copy(deep=True)
        for idx, (col_name, nulls_pct) in enumerate(self.columns_nulls_percentage_dct.items()):
            if nulls_pct == 0.0:
                continue

            # Include existing nulls in the defined nulls percentage
            existing_nulls_count = df_copy[col_name].isna().sum()
            target_nulls_count = int(df_copy.shape[0] * nulls_pct)
            if existing_nulls_count > target_nulls_count:
                raise ValueError(f"Existing nulls count in {col_name} column is greater than target nulls count. "
                                 f"Increase a nulls percentage for the column to be greater than the percentage of existing nulls.")

            # Set nulls to other indices than indices of existing nulls
            nulls_sample_size = target_nulls_count - existing_nulls_count
            notna_idxs = df_copy[df_copy[col_name].notna()].index
            np.random.seed(self.seed + idx)
            random_row_idxs = np.random.choice(notna_idxs, size=nulls_sample_size, replace=False)
            df_copy.loc[random_row_idxs, col_name] = np.nan

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column)
        return transformed_df
