import math
import numpy as np
import pandas as pd

from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class MislabelsGenerator(AbstractErrorInjector):
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


class OutliersGenerator(AbstractErrorInjector):
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
            new_val = val + 3 * self.columns_stats[col_name]['std']  # + normal(min, max) // std
            if isinstance(val, int):
                new_val = math.ceil(new_val)
            return new_val
        else:
            new_val = val - 3 * self.columns_stats[col_name]['std']  # - normal(min, max) // std
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


class RandomNullsGenerator(AbstractErrorInjector):
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
