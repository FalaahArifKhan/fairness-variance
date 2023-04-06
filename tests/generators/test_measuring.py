import numpy as np
from virny.datasets.data_loaders import CreditDataset

from source.generators.measuring import MislabelsGenerator


def test_mislabels_generator():
    seed = 42
    mislabels_percentage = 0.1
    data_loader = CreditDataset(subsample_size=50_000)
    data_loader.full_df = data_loader.full_df.reset_index(drop=True)

    generator = MislabelsGenerator(seed, mislabels_percentage)
    new_df = generator.fit_transform(data_loader.full_df, data_loader.target)
    bool_series = data_loader.full_df[data_loader.target] == new_df[data_loader.target]
    mislabels_num = np.size(bool_series) - np.sum(bool_series)

    assert mislabels_num == int(data_loader.full_df[data_loader.target].shape[0] * mislabels_percentage)
