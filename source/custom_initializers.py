import copy


def create_experiment_data_loader(data_loader, generator=None):
    exp_iter_data_loader = copy.deepcopy(data_loader)  # Add deepcopy to avoid data leakage
    transformed_df = generator.fit_transform(exp_iter_data_loader.full_df, exp_iter_data_loader.target)

    exp_iter_data_loader.full_df = transformed_df
    exp_iter_data_loader.X_data = transformed_df[exp_iter_data_loader.features]
    exp_iter_data_loader.y_data = transformed_df[exp_iter_data_loader.target]
    exp_iter_data_loader.columns_with_nulls = \
        exp_iter_data_loader.X_data.columns[exp_iter_data_loader.X_data.isna().any().to_list()].to_list()

    return exp_iter_data_loader
