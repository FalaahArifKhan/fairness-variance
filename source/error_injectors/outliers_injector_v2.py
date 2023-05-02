import math
import numpy as np
import pandas as pd

from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class OutliersInjectorV2(AbstractErrorInjector):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    columns_outliers_percentage_dct
        Dictionary where keys are column names and values are target percentages of outliers in not-null values of each column

    """
    def __init__(self, seed: int, columns_to_transform: list, row_idx_percentage: float,
                 max_num_columns_to_effect: int):
        super().__init__(seed)
        self.columns_to_transform = columns_to_transform
        self.row_idx_percentage = row_idx_percentage
        self.max_num_columns_to_effect = max_num_columns_to_effect
        self.columns_stats = dict()  # Dict for mean and std of selected columns

    def _validate_input(self, df):
        for col in self.columns_to_transform:
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_to_transform must be the dataframe column names")

        if self.row_idx_percentage < 0 or self.row_idx_percentage > 1:
            raise ValueError("Column nulls percentage must be in [0.0-1.0] range.")

    def set_columns_to_transform(self, new_columns_to_transform):
        self.columns_to_transform = new_columns_to_transform

    def set_percentage_var(self, new_row_idx_percentage):
        self.row_idx_percentage = new_row_idx_percentage

    def set_max_num_columns_to_effect(self, new_max_num_columns_to_effect):
        self.max_num_columns_to_effect = new_max_num_columns_to_effect

    def increment_seed(self):
        self.seed += 1

    def _detect_outliers_std(self, df, col_name):
        mean = self.columns_stats[col_name]['mean']
        std = self.columns_stats[col_name]['std']
        return df[(df[col_name] > (mean + 3 * std)) | (df[col_name] < (mean - 3 * std))]

    def _generate_outliers(self, val, col_name):
        if val > self.columns_stats[col_name]['mean']:
            new_val = val + 3 * self.columns_stats[col_name]['std']
            if isinstance(val, int):
                new_val = math.ceil(new_val)

            return new_val
        else:
            new_val = val - 3 * self.columns_stats[col_name]['std']
            if isinstance(val, int):
                new_val = math.floor(new_val)

            return new_val

    def _create_random_sample_df(self, df_copy):
        nulls_sample_size = int(df_copy.shape[0] * self.row_idx_percentage)
        np.random.seed(self.seed)
        random_row_idxs = np.random.choice(df_copy.index, size=nulls_sample_size, replace=False)
        # Choose a random number of columns to place nulls for each selected row index
        np.random.seed(self.seed)
        random_num_columns_for_nulls = np.random.choice(
            [i + 1 for i in range(self.max_num_columns_to_effect)],
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

        return random_sample_df

    def fit(self, df, target_column: str = None):
        self._validate_input(df)
        for col in self.columns_to_transform:
            self.columns_stats[col] = dict()
            self.columns_stats[col]['mean'] = np.mean(df[col])
            self.columns_stats[col]['std'] = np.std(df[col])
            self.columns_stats[col]['min'] = np.min(df[col])
            self.columns_stats[col]['max'] = np.max(df[col])

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        if self.row_idx_percentage == 0.0:
            return df_copy

        if len(self.columns_stats.keys()) == 0:
            self.fit(df_copy)

        random_sample_df = self._create_random_sample_df(df_copy)
        for idx, col_name in enumerate(self.columns_to_transform):
            col_random_row_idxs = random_sample_df[random_sample_df['random_column_name'] == col_name]['random_row_idx'].values
            if col_random_row_idxs.shape[0] == 0:
                continue

            df_copy.loc[col_random_row_idxs, col_name] = \
                df_copy.loc[col_random_row_idxs, col_name].apply(self._generate_outliers, args=(col_name,))

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df)
        return transformed_df
