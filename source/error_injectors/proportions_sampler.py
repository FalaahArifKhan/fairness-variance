import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class ProportionsSampler(AbstractErrorInjector):
    """
    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    column_for_subsampling
        Column name to use to subsample an input dataset
    new_proportions_dct
        Dictionary where keys are all unique column values and values are target proportions of the unique column values to each other

    """
    def __init__(self, seed: int, column_for_subsampling, new_proportions_dct: dict):
        super().__init__(seed)
        self.column_for_subsampling = column_for_subsampling
        self.new_proportions_pct_dct = new_proportions_dct
        self.old_proportions_count_dct = None
        self.new_proportions_count_dct = None

    def _validate_input(self, df):
        total_sum = 0
        for val in self.new_proportions_pct_dct.values():
            if val > 1.0 or val < 0.0:
                raise ValueError(f"Value caused the issue is {val}. "
                                 f"Each value in new_proportions_dct must be in [0.0-1.0] range")
            total_sum += val

        if abs(total_sum - 1.0) > 0.000_001:
            raise ValueError(f"Total sum of new_proportions_dct values ({total_sum}) is not equal to 1.0")

        for col_value in df[self.column_for_subsampling].unique():
            if col_value not in self.new_proportions_pct_dct:
                raise ValueError(f"Value caused the issue is {col_value}. "
                                 f"Column value is not in new_proportions_dct, which must include all unique column values")

    def fit(self, df, target_column: str = None):
        self._validate_input(df)
        self.old_proportions_count_dct = df[self.column_for_subsampling].value_counts().to_dict()

    def transform(self, df: pd.DataFrame, target_column: str = None):
        # Find counts for new proportions
        min_key, min_val = min(self.old_proportions_count_dct.items(), key=lambda x: x[1])
        self.new_proportions_count_dct = {k: int(min_val * self.new_proportions_pct_dct[k] // self.new_proportions_pct_dct[min_key])
                                          for k, v in self.old_proportions_count_dct.items()}

        # Subsample using the new proportions
        df_subsample = pd.DataFrame()
        for col_val, subsample_rows_count in self.new_proportions_count_dct.items():
            group_idxs = df[df[self.column_for_subsampling] == col_val].index
            np.random.seed(self.seed)
            random_row_idxs = np.random.choice(group_idxs, size=subsample_rows_count, replace=False)
            df_subsample = pd.concat([df_subsample, df.loc[random_row_idxs, :]])

        df_subsample = shuffle(df_subsample, random_state=self.seed)
        return df_subsample.reset_index(drop=True)

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df, target_column)
        return transformed_df
