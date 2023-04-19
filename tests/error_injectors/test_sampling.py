from virny.datasets.data_loaders import CompasDataset

from source.error_injectors.proportions_sampler import ProportionsSampler


def test_proportions_generator():
    seed = 42
    column_for_subsampling = 'race'
    new_proportions_dct = {
        'African-American': 0.5,
        'Caucasian': 0.5,
    }
    data_loader = CompasDataset()
    generator = ProportionsSampler(seed, column_for_subsampling, new_proportions_dct)
    new_df = generator.fit_transform(data_loader.full_df, target_column=None)
    col_value_counts = new_df[column_for_subsampling].value_counts().to_dict()
    total_sum = 0
    for col_val, col_value_count in col_value_counts.items():
        if col_val is not None:
            total_sum += col_value_count

    for col_val, col_value_count in col_value_counts.items():
        subsample_val_proportion = col_value_count / total_sum
        assert abs(subsample_val_proportion - new_proportions_dct[col_val]) <= 0.02


def test_proportions_generator_with_null_columns():
    seed = 100
    column_for_subsampling = 'race'
    new_proportions_dct = {
        'African-American': 0.5,
        'Caucasian': 0.5,
    }
    data_loader = CompasDataset()
    generator = ProportionsSampler(seed, column_for_subsampling, new_proportions_dct)
    try:
        new_df = generator.fit_transform(data_loader.full_df, target_column=None)
        assert True
    except ValueError as _:
        assert False
