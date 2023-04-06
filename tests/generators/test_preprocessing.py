import numpy as np
from virny.datasets.data_loaders import CreditDataset

from source.generators.preprocessing import RandomNullsGenerator


def test_random_nulls_generator():
    seed = 42
    columns_nulls_percentage_dct = {
        'age': 0.1,
        'NumberRealEstateLoansOrLines': 0.2,
        'NumberOfOpenCreditLinesAndLoans': 0.3,
    }
    data_loader = CreditDataset(subsample_size=50_000)
    generator = RandomNullsGenerator(seed, columns_nulls_percentage_dct)
    new_df = generator.fit_transform(data_loader.full_df, target_column=None)

    for col_name, nulls_pct in columns_nulls_percentage_dct.items():
        bool_series = data_loader.full_df[col_name] == new_df[col_name]
        nulls_count = np.size(bool_series) - np.sum(bool_series)

        assert nulls_count == int(data_loader.full_df[col_name].shape[0] * nulls_pct)
