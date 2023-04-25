import math
import numpy as np
import pandas as pd

from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class RandomNullsInjectorV2(AbstractErrorInjector):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    columns_nulls_percentage_dct
        Dictionary where keys are column names and values are target percentages of nulls for a column

    """
    def __init__(self, seed: int, columns_to_transform: list,
                 row_idx_nulls_percentage: float, num_columns_to_effect: int):
        super().__init__(seed)
        self.columns_to_transform = columns_to_transform
        self.row_idx_nulls_percentage = row_idx_nulls_percentage
        self.num_columns_to_effect = num_columns_to_effect

    def _validate_input(self, df):
        for col in self.columns_to_transform:
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_to_transform must be the dataframe column names")

        if self.row_idx_nulls_percentage < 0 or self.row_idx_nulls_percentage > 1:
            raise ValueError("Column nulls percentage must be in [0.0-1.0] range.")

    def set_percentage_var(self, new_row_idx_nulls_percentage):
        self.row_idx_nulls_percentage = new_row_idx_nulls_percentage

    def fit(self, df, target_column: str = None):
        self._validate_input(df)

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        if self.row_idx_nulls_percentage == 0.0:
            return df_copy

        nulls_sample_size = int(df_copy.shape[0] * self.row_idx_nulls_percentage)
        np.random.seed(self.seed)
        random_row_idxs = np.random.choice(df_copy.index, size=nulls_sample_size, replace=False)
        # Choose a random number of columns to place nulls for each selected row index
        np.random.seed(self.seed)
        random_num_columns_for_nulls = np.random.choice(
            # [i + 1 for i in range(math.ceil(len(self.columns_to_transform) * self.row_idx_nulls_percentage))],
            [i + 1 for i in range(self.num_columns_to_effect)],
            size=nulls_sample_size, replace=True
        )

        random_sample_df = pd.DataFrame(columns=['random_row_idx', 'random_column_name'])
        iter = np.nditer(random_row_idxs, flags=['f_index'])
        for random_row_idx in iter:
            random_num_columns = random_num_columns_for_nulls[iter.index]
            np.random.seed(self.seed + iter.index)
            random_columns = np.random.choice(self.columns_to_transform, size=random_num_columns, replace=False)
            for random_column_name in np.nditer(random_columns):
                random_sample = {'random_row_idx': int(random_row_idx), 'random_column_name': random_column_name}
                random_sample_df = random_sample_df.append(random_sample, ignore_index = True)

        for idx, col_name in enumerate(self.columns_to_transform):
            col_random_row_idxs = random_sample_df[random_sample_df['random_column_name'] == col_name]['random_row_idx'].values
            if col_random_row_idxs.shape[0] == 0:
                continue

            df_copy.loc[col_random_row_idxs, col_name] = None

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df)
        return transformed_df
