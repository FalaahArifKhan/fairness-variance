import math
import numpy as np
import pandas as pd

from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class OutliersInjector(AbstractErrorInjector):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    columns_outliers_percentage_dct
        Dictionary where keys are column names and values are target percentages of outliers in not-null values of each column

    """
    def __init__(self, seed: int, columns_outliers_percentage_dct: dict):
        super().__init__(seed)
        self.columns_outliers_percentage_dct = columns_outliers_percentage_dct
        self.columns_stats = dict()  # Dict for mean and std of selected columns

    def _validate_input(self, df):
        for col, col_outliers_pct in self.columns_outliers_percentage_dct.items():
            if col not in list(df.columns):
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_outliers_percentage_dct must be the dataframe column names")
            if col_outliers_pct < 0 or col_outliers_pct > 1:
                raise ValueError(f"Value caused the issue is {col_outliers_pct}. "
                                 f"Column outliers percentage must be in [0.0-1.0] range.")

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

            # # Avoid outliers that are less than zero
            # if new_val < 0:
            #     new_val = val + 2 * (self.columns_stats[col_name]['mean'] - val) + 3 * self.columns_stats[col_name]['std']
            #     if isinstance(val, int):
            #         new_val = math.floor(new_val)
            # # TODO: is it correct?
            # # return max(val, self.columns_stats[col_name]['min'])
            # # return max(val, 0)
            return new_val

    def fit(self, df, target_column: str = None):
        self._validate_input(df)
        for col in self.columns_outliers_percentage_dct.keys():
            self.columns_stats[col] = dict()
            self.columns_stats[col]['mean'] = np.mean(df[col])
            self.columns_stats[col]['std'] = np.std(df[col])
            self.columns_stats[col]['min'] = np.min(df[col])
            self.columns_stats[col]['max'] = np.max(df[col])

    def transform(self, df: pd.DataFrame, target_column: str = None):
        df_copy = df.copy(deep=True)
        for idx, (col_name, outliers_pct) in enumerate(self.columns_outliers_percentage_dct.items()):
            if outliers_pct == 0.0:
                continue

            # TODO: is it correct?
            notna_idxs = df_copy[df_copy[col_name].notna()].index
            existing_outliers_idxs = self._detect_outliers_std(df_copy, col_name).index
            outliers_sample_idxs = [idx for idx in notna_idxs if idx not in existing_outliers_idxs]

            # New outliers size equals to (notna indices * outliers_pct - count of existing outliers)
            outliers_sample_size = int(notna_idxs.shape[0] * outliers_pct) - existing_outliers_idxs.shape[0]
            np.random.seed(self.seed + idx)
            random_row_idxs = np.random.choice(outliers_sample_idxs, size=outliers_sample_size, replace=False)
            df_copy.loc[random_row_idxs, col_name] = \
                df_copy.loc[random_row_idxs, col_name].apply(self._generate_outliers, args=(col_name,))

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column)
        return transformed_df
