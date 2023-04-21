import numpy as np
from virny.datasets.data_loaders import CreditDataset

from tests.utils import detect_outliers_std
from source.error_injectors.outliers_injector import OutliersInjector
from source.error_injectors.mislabels_injector import MislabelsInjector
from source.error_injectors.random_nulls_injector import RandomNullsInjector
from source.error_injectors.random_nulls_injector_v2 import RandomNullsInjectorV2
from source.error_injectors.outliers_injector_v2 import OutliersInjectorV2


def test_mislabels_generator():
    seed = 42
    mislabels_percentage = 0.1
    data_loader = CreditDataset(subsample_size=50_000)
    data_loader.full_df = data_loader.full_df.reset_index(drop=True)

    generator = MislabelsInjector(seed, mislabels_percentage)
    new_df = generator.fit_transform(data_loader.full_df, data_loader.target)
    bool_series = data_loader.full_df[data_loader.target] == new_df[data_loader.target]
    mislabels_num = np.size(bool_series) - np.sum(bool_series)

    assert mislabels_num == int(data_loader.full_df[data_loader.target].shape[0] * mislabels_percentage)


def test_outliers_generator():
    seed = 42
    columns_outliers_percentage_dct = {
        'age': 0.1,
        'NumberRealEstateLoansOrLines': 0.2,
        'NumberOfOpenCreditLinesAndLoans': 0.3,
    }
    data_loader = CreditDataset(subsample_size=50_000, subsample_seed=seed)
    generator = OutliersInjector(seed, columns_outliers_percentage_dct)
    new_df = generator.fit_transform(data_loader.full_df, target_column=None)

    for col_name, outliers_pct in columns_outliers_percentage_dct.items():
        new_outliers = detect_outliers_std(data_loader.full_df, new_df, col_name)
        outliers_count = new_outliers.shape[0]

        notna_idxs = data_loader.full_df[col_name].notna().index
        assert outliers_count == int(notna_idxs.shape[0] * outliers_pct)


def test_outliers_injector_v2():
    seed = 42
    data_loader = CreditDataset(subsample_size=50_000)
    row_idx_percentage = 0.36
    columns_to_transform = ['NumberRealEstateLoansOrLines', 'NumberOfOpenCreditLinesAndLoans']
    injector = OutliersInjectorV2(seed,
                                  columns_to_transform=columns_to_transform,
                                  row_idx_percentage=row_idx_percentage)
    new_df = injector.fit_transform(data_loader.full_df, target_column=None)

    total_outliers_count = 0
    for col_name in columns_to_transform:
        new_outliers = detect_outliers_std(data_loader.full_df, new_df, col_name)
        outliers_count = new_outliers.shape[0]
        total_outliers_count += outliers_count

    assert total_outliers_count >= int(data_loader.full_df.shape[0] * row_idx_percentage)


def test_random_nulls_generator():
    seed = 42
    columns_nulls_percentage_dct = {
        'age': 0.1,
        'NumberRealEstateLoansOrLines': 0.2,
        'NumberOfOpenCreditLinesAndLoans': 0.3,
    }
    data_loader = CreditDataset(subsample_size=50_000)
    generator = RandomNullsInjector(seed, columns_nulls_percentage_dct)
    new_df = generator.fit_transform(data_loader.full_df, target_column=None)

    for col_name, nulls_pct in columns_nulls_percentage_dct.items():
        nulls_count = new_df[col_name].isna().sum()
        assert nulls_count == int(data_loader.full_df[col_name].shape[0] * nulls_pct)


def test_random_nulls_generator_with_null_columns():
    seed = 100
    columns_nulls_percentage_dct = {
        'age': 0.1,
        'MonthlyIncome': 0.2,
        'NumberOfDependents': 0.3,
    }
    data_loader = CreditDataset(subsample_size=50_000, subsample_seed=seed)
    generator = RandomNullsInjector(seed, columns_nulls_percentage_dct)
    new_df = generator.fit_transform(data_loader.full_df, target_column=None)

    for col_name, nulls_pct in columns_nulls_percentage_dct.items():
        nulls_count = new_df[col_name].isna().sum()
        assert nulls_count == int(data_loader.full_df[col_name].shape[0] * nulls_pct)


def test_random_nulls_injector_v2():
    seed = 42
    data_loader = CreditDataset(subsample_size=50_000)
    row_idx_nulls_percentage = 0.5
    columns_to_transform = ['NumberRealEstateLoansOrLines', 'NumberOfOpenCreditLinesAndLoans']
    injector = RandomNullsInjectorV2(seed,
                                     columns_to_transform=columns_to_transform,
                                     row_idx_nulls_percentage=row_idx_nulls_percentage)
    new_df = injector.fit_transform(data_loader.full_df, target_column=None)

    total_nulls_count = 0
    for col_name in columns_to_transform:
        nulls_count = new_df[col_name].isna().sum()
        total_nulls_count += nulls_count

    assert total_nulls_count == int(data_loader.full_df.shape[0] * row_idx_nulls_percentage)
