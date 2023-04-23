from virny.datasets.data_loaders import CreditDataset, CompasDataset

from source.error_injectors.outliers_injector import OutliersInjector
from source.error_injectors.outliers_injector_v2 import OutliersInjectorV2
from source.utils.common_helpers import detect_outliers_std


def test_outliers_injector():
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
    data_loader = CompasDataset(subsample_size=5000, subsample_seed=seed)
    row_idx_percentage = 0.36
    # Only for numerical columns
    columns_to_transform = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
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
    assert total_outliers_count == 2988
