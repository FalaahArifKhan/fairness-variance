import math
import numpy as np
import pandas as pd

from source.generators.abstract_generator import AbstractGenerator


class MislabelsGenerator(AbstractGenerator):
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
        if self.mislabels_percentage == 0.0:
            return df

        df_copy = df.copy(deep=True)
        np.random.seed(self.seed)
        mislabels_sample_size = int(df_copy.shape[0] * self.mislabels_percentage)
        random_row_idxs = np.random.choice(df_copy.index, size=mislabels_sample_size, replace=False)
        df_copy.loc[random_row_idxs, target_column] = 1 - df_copy.loc[random_row_idxs, target_column]

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column)
        return transformed_df


class OutliersGenerator(AbstractGenerator):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    columns_outliers_percentage_dct
        Dictionary where keys are column names and values are target percentages of outliers in not null values of each column

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
            val += 3 * self.columns_stats[col_name]['std']
            if isinstance(val, int):
                val = math.ceil(val)
            # return min(val, self.columns_stats[col_name]['max'])
            return val

        else:
            val -= 3 * self.columns_stats[col_name]['std']
            if isinstance(val, int):
                val = math.floor(val)
            # TODO: is it correct?
            # return max(val, self.columns_stats[col_name]['min'])
            # return max(val, 0)
            return val

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
