from virny.datasets.data_loaders import CreditDataset, CompasDataset

from source.error_injectors.random_nulls_injector import RandomNullsInjector
from source.error_injectors.random_nulls_injector_v2 import RandomNullsInjectorV2


def test_random_nulls_injector():
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


def test_random_nulls_injector_with_null_columns():
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
    data_loader = CompasDataset(subsample_size=5000)
    row_idx_nulls_percentage = 0.5
    columns_to_transform = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'sex']
    injector = RandomNullsInjectorV2(seed,
                                     columns_to_transform=columns_to_transform,
                                     row_idx_nulls_percentage=row_idx_nulls_percentage)
    new_df = injector.fit_transform(data_loader.full_df, target_column=None)

    total_nulls_count = 0
    for col_name in columns_to_transform:
        nulls_count = new_df[col_name].isna().sum()
        total_nulls_count += nulls_count

    assert total_nulls_count >= int(data_loader.full_df.shape[0] * row_idx_nulls_percentage)
    assert total_nulls_count == 4936


def test_random_nulls_injector_v2_other():
    seed = 42
    data_loader = CompasDataset(subsample_size=5000, subsample_seed=seed)
    row_idx_nulls_percentage = 0.3
    columns_to_transform = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'sex']
    injector = RandomNullsInjectorV2(seed,
                                     columns_to_transform=columns_to_transform,
                                     row_idx_nulls_percentage=row_idx_nulls_percentage)
    new_df = injector.fit_transform(data_loader.full_df, target_column=None)

    total_nulls_count = 0
    for col_name in columns_to_transform:
        nulls_count = new_df[col_name].isna().sum()
        total_nulls_count += nulls_count

    assert total_nulls_count >= int(data_loader.full_df.shape[0] * row_idx_nulls_percentage)
    assert total_nulls_count == 2247


def test_random_nulls_injector_v2_zero():
    seed = 42
    data_loader = CreditDataset(subsample_size=50_000)
    row_idx_nulls_percentage = 0.0
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
