import copy
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from fairlearn.preprocessing import CorrelationRemover
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.model_selection import train_test_split

from virny.datasets.data_loaders import BaseDataLoader
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.preprocessing.basic_preprocessing import preprocess_dataset


def get_simple_preprocessor(data_loader):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('num', StandardScaler(), data_loader.numerical_columns),
    ])


def get_preprocessor_for_diabetes(data_loader):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('enc', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), ['diag_1', 'diag_2', 'diag_3']),  # there are no nulls in these columns, use it as a simple 1:1 mapper to output
        ('num', StandardScaler(), data_loader.numerical_columns),
    ])


def create_extra_test_sets(extra_data_loaders: list, column_transformer, test_set_fraction, seed):
    extra_test_sets = []
    for exp_data_loader in extra_data_loaders:
        cur_column_transformer = copy.deepcopy(column_transformer)
        exp_base_flow_dataset = preprocess_dataset(exp_data_loader, cur_column_transformer, test_set_fraction, seed)
        extra_test_sets.append((exp_base_flow_dataset.X_test, exp_base_flow_dataset.y_test))

    return extra_test_sets


def preprocess_dataset_with_col_transformer(data_loader: BaseDataLoader, column_transformer: ColumnTransformer,
                                            test_set_fraction: float, dataset_split_seed: int,
                                            train_set_subsample_size: int = None):
    if test_set_fraction < 0.0 or test_set_fraction > 1.0:
        raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

    # Split and preprocess the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                test_size=test_set_fraction,
                                                                random_state=dataset_split_seed)
    # Subsample X_train_val, y_train_val, and data_loader.full_df to train_set_subsample_size if defined
    full_df = data_loader.full_df
    if train_set_subsample_size is not None:
        # extra_shift = 42
        X_train_val = X_train_val.sample(train_set_subsample_size, random_state=dataset_split_seed)
        y_train_val = y_train_val.sample(train_set_subsample_size, random_state=dataset_split_seed)
        full_df = full_df.iloc[list(X_train_val.index) + list(X_test.index)]
        print("Top indexes of X_train_val: ", X_train_val.index[:20], flush=True)
        print("Top indexes of y_train_val: ", y_train_val.index[:20], flush=True)
        print('\n\n', flush=True)

    print('In-domain full_df.shape -- ', full_df.shape)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    X_train_features = column_transformer.fit_transform(X_train_val)
    X_test_features = column_transformer.transform(X_test)

    base_flow_dataset = BaseFlowDataset(init_features_df=full_df.drop(data_loader.target, axis=1, errors='ignore'),
                                        X_train_val=X_train_features,
                                        X_test=X_test_features,
                                        y_train_val=y_train_val,
                                        y_test=y_test,
                                        target=data_loader.target,
                                        numerical_columns=data_loader.numerical_columns,
                                        categorical_columns=data_loader.categorical_columns)
    return base_flow_dataset, column_transformer


def preprocess_mult_data_loaders_for_disp_imp(main_data_loader, extra_data_loaders: list, test_set_fraction: float,
                                              experiment_seed: int, train_set_subsample_size: int):
    data_loaders = [main_data_loader] + extra_data_loaders
    init_data_loaders = [copy.deepcopy(dl) for dl in data_loaders]

    # Add RACE column for DisparateImpactRemover and remove 'SEX', 'RAC1P' to create a blind estimator
    transformed_data_loaders = []
    for cur_data_loader in data_loaders:
        cur_data_loader.categorical_columns = [col for col in cur_data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        cur_data_loader.X_data['RACE'] = cur_data_loader.X_data['RAC1P'].apply(lambda x: 1 if x == '1' else 0)
        cur_data_loader.full_df = cur_data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        cur_data_loader.X_data = cur_data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)
        transformed_data_loaders.append(cur_data_loader)

    main_transformed_data_loader, extra_transformed_data_loaders = transformed_data_loaders[0], transformed_data_loaders[1:]

    # Preprocess the main dataset using the defined preprocessor
    column_transformer = get_simple_preprocessor(main_transformed_data_loader)
    main_base_flow_dataset, trained_column_transformer = (
        preprocess_dataset_with_col_transformer(data_loader=main_transformed_data_loader,
                                                column_transformer=column_transformer,
                                                test_set_fraction=test_set_fraction,
                                                dataset_split_seed=experiment_seed,
                                                train_set_subsample_size=train_set_subsample_size))
    # Subsample main_base_flow_dataset.init_features_df if train_set_subsample_size is defined
    if train_set_subsample_size is not None:
        subsampled_full_df = init_data_loaders[0].full_df.iloc[list(main_base_flow_dataset.X_train_val.index) +
                                                               list(main_base_flow_dataset.X_test.index)]
        main_base_flow_dataset.init_features_df = subsampled_full_df.drop(init_data_loaders[0].target, axis=1, errors='ignore')
    else:
        main_base_flow_dataset.init_features_df = init_data_loaders[0].full_df.drop(init_data_loaders[0].target, axis=1, errors='ignore')

    main_base_flow_dataset.X_train_val['RACE'] = main_transformed_data_loader.X_data.loc[main_base_flow_dataset.X_train_val.index, 'RACE']
    main_base_flow_dataset.X_test['RACE'] = main_transformed_data_loader.X_data.loc[main_base_flow_dataset.X_test.index, 'RACE']
    print('In-domain init_features_df.shape -- ', main_base_flow_dataset.init_features_df.shape)
    print('In-domain number of rows in X_train_val -- ', main_base_flow_dataset.X_train_val.shape[0])
    print('In-domain number of rows in X_test -- ', main_base_flow_dataset.X_test.shape[0])

    # Preprocess extra data loaders using the column transformer trained on the main data loader
    extra_base_flow_datasets = []
    for idx, cur_data_loader in enumerate(extra_transformed_data_loaders):
        cur_init_data_loader = init_data_loaders[idx + 1] # Skip the main init data loader
        cur_X_train_val, cur_X_test, cur_y_train_val, cur_y_test =\
            train_test_split(cur_data_loader.X_data, cur_data_loader.y_data,
                             test_size=test_set_fraction,
                             random_state=experiment_seed)
        # Preprocess the current extra dataset using the trained preprocessor
        cur_X_test_features = trained_column_transformer.transform(cur_X_test)
        cur_base_flow_dataset = BaseFlowDataset(init_features_df=cur_data_loader.full_df.drop(cur_data_loader.target, axis=1, errors='ignore'),
                                                X_train_val=pd.DataFrame(), # As an empty df
                                                X_test=cur_X_test_features,
                                                y_train_val=pd.DataFrame(), # As an empty df
                                                y_test=cur_y_test,
                                                target=cur_data_loader.target,
                                                numerical_columns=cur_data_loader.numerical_columns,
                                                categorical_columns=cur_data_loader.categorical_columns)

        cur_base_flow_dataset.init_features_df = cur_init_data_loader.full_df.drop(cur_init_data_loader.target, axis=1, errors='ignore')
        cur_base_flow_dataset.X_test['RACE'] = cur_data_loader.X_data.loc[cur_base_flow_dataset.X_test.index, 'RACE']
        print(f'Out-of-domain {idx + 1} init_features_df.shape -- ', cur_base_flow_dataset.init_features_df.shape)
        print(f'Out-of-domain {idx + 1} number of rows in X_train_val -- ', cur_base_flow_dataset.X_train_val.shape[0])
        print(f'Out-of-domain {idx + 1} number of rows in X_test -- ', cur_base_flow_dataset.X_test.shape[0])

        extra_base_flow_datasets.append(cur_base_flow_dataset)

    return main_base_flow_dataset, extra_base_flow_datasets


def create_models_in_range_dct(all_subgroup_metrics_per_model_dct: dict, all_group_metrics_per_model_dct: dict,
                               metrics_value_range_dct: dict, group: str):
    # Merge subgroup and group metrics for each model and align their columns
    all_metrics_for_all_models_df = pd.DataFrame()
    for model_name in all_subgroup_metrics_per_model_dct.keys():
        group_metrics_per_model_df = all_group_metrics_per_model_dct[model_name][
            (all_group_metrics_per_model_dct[model_name]['Group'] == group) &
            (all_group_metrics_per_model_dct[model_name]['Test_Set_Index'] == 0)
            ]
        subgroup_metrics_per_model_df = all_subgroup_metrics_per_model_dct[model_name][
            (all_subgroup_metrics_per_model_dct[model_name]['Subgroup'] == 'overall') &
            (all_subgroup_metrics_per_model_dct[model_name]['Test_Set_Index'] == 0)
            ]
        subgroup_metrics_per_model_df['Group'] = subgroup_metrics_per_model_df['Subgroup']
        aligned_subgroup_metrics_per_model_df = subgroup_metrics_per_model_df[group_metrics_per_model_df.columns]

        combined_metrics_per_model_df = pd.concat([group_metrics_per_model_df, aligned_subgroup_metrics_per_model_df]).reset_index(drop=True)
        all_metrics_for_all_models_df = pd.concat([all_metrics_for_all_models_df, combined_metrics_per_model_df])

    all_metrics_for_all_models_df = all_metrics_for_all_models_df.reset_index(drop=True)
    all_metrics_for_all_models_df = all_metrics_for_all_models_df.drop(['Group', 'Test_Set_Index'], axis=1)

    # Create new columns based on values in Metric and Metric_Value columns
    pivoted_model_metrics_df = all_metrics_for_all_models_df.pivot(columns='Metric', values='Metric_Value',
                                                                   index=[col for col in all_metrics_for_all_models_df.columns
                                                                          if col not in ('Metric', 'Metric_Value')]).reset_index()

    # Create a pandas condition for filtering based on the input value ranges
    models_in_range_df = pd.DataFrame()
    for idx, (metric_group, value_range) in enumerate(metrics_value_range_dct.items()):
        pd_condition = None
        if '&' not in metric_group:
            min_range_val, max_range_val = value_range
            if max_range_val < min_range_val:
                raise ValueError('The second element in the input range must be greater than the first element, '
                                 'so to be in the following format -- (min_range_val, max_range_val)')
            metric = metric_group
            pd_condition = (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)
        else:
            metrics = metric_group.split('&')
            for idx, metric in enumerate(metrics):
                min_range_val, max_range_val = metrics_value_range_dct[metric]
                if max_range_val < min_range_val:
                    raise ValueError('The second element in the input range must be greater than the first element, '
                                     'so to be in the following format -- (min_range_val, max_range_val)')
                if idx == 0:
                    pd_condition = (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)
                else:
                    pd_condition &= (pivoted_model_metrics_df[metric] >= min_range_val) & (pivoted_model_metrics_df[metric] <= max_range_val)

        num_satisfied_models_df = pivoted_model_metrics_df[pd_condition]['Model_Name'].value_counts().reset_index()
        num_satisfied_models_df.rename(columns = {'Model_Name': 'Number_of_Models'}, inplace = True)
        num_satisfied_models_df.rename(columns = {'index': 'Model_Name'}, inplace = True)
        num_satisfied_models_df['Metric_Group'] = metric_group
        if idx == 0:
            models_in_range_df = num_satisfied_models_df
        else:
            # Concatenate based on rows
            models_in_range_df = pd.concat([models_in_range_df, num_satisfied_models_df], ignore_index=True, sort=False)

    return models_in_range_df


def remove_disparate_impact(init_base_flow_dataset, alpha):
    """
    Based on this documentation:
     https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    # sensitive_attribute = 'RACE'
    sensitive_attribute = 'race_binary'
    train_df = base_flow_dataset.X_train_val
    train_df[base_flow_dataset.target] = base_flow_dataset.y_train_val
    test_df = base_flow_dataset.X_test
    test_df[base_flow_dataset.target] = base_flow_dataset.y_test

    train_binary_dataset = BinaryLabelDataset(df=train_df,
                                              label_names=[base_flow_dataset.target],
                                              protected_attribute_names=[sensitive_attribute],
                                              favorable_label=1,
                                              unfavorable_label=0)
    test_binary_dataset = BinaryLabelDataset(df=test_df,
                                             label_names=[base_flow_dataset.target],
                                             protected_attribute_names=[sensitive_attribute],
                                             favorable_label=1,
                                             unfavorable_label=0)

    di = DisparateImpactRemover(repair_level=alpha, sensitive_attribute=sensitive_attribute)
    train_repaired_df, _ = di.fit_transform(train_binary_dataset).convert_to_dataframe()
    test_repaired_df , _ = di.fit_transform(test_binary_dataset).convert_to_dataframe()
    train_repaired_df.index = train_repaired_df.index.astype(dtype='int64')
    test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')

    base_flow_dataset.X_train_val = train_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)
    base_flow_dataset.y_train_val = train_repaired_df[base_flow_dataset.target]
    base_flow_dataset.X_test = test_repaired_df.drop([base_flow_dataset.target, sensitive_attribute], axis=1)
    base_flow_dataset.y_test = test_repaired_df[base_flow_dataset.target]

    return base_flow_dataset


def remove_disparate_impact_with_mult_sets(init_base_flow_dataset, alpha, init_extra_base_flow_datasets):
    """
    Based on this documentation:
     https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    extra_base_flow_datasets = copy.deepcopy(init_extra_base_flow_datasets)
    sensitive_attribute = 'RACE'
    other_feature_columns = [col for col in base_flow_dataset.X_train_val.columns if col != sensitive_attribute]

    # Rearrange columns for consistency
    base_flow_dataset.X_train_val = base_flow_dataset.X_train_val[other_feature_columns + [sensitive_attribute]]

    # Align columns in the test sets with the main train set
    base_flow_datasets = [base_flow_dataset] + extra_base_flow_datasets
    aligned_base_flow_datasets = []
    for i in range(len(base_flow_datasets)):
        cur_base_flow_ds = base_flow_datasets[i]
        for col in other_feature_columns:
            if col not in cur_base_flow_ds.X_test.columns:
                cur_base_flow_ds.X_test[col] = 0.0

        cur_base_flow_ds.X_test = cur_base_flow_ds.X_test[other_feature_columns + [sensitive_attribute]]
        aligned_base_flow_datasets.append(cur_base_flow_ds)

    main_base_flow_dataset = base_flow_datasets[0]
    extra_aligned_base_flow_datasets = base_flow_datasets[1:]

    # Apply Disparate Impact Remover on the in-domain dataset
    train_df = main_base_flow_dataset.X_train_val
    train_df[main_base_flow_dataset.target] = main_base_flow_dataset.y_train_val
    test_df = main_base_flow_dataset.X_test
    test_df[main_base_flow_dataset.target] = main_base_flow_dataset.y_test

    train_binary_dataset = BinaryLabelDataset(df=train_df,
                                              label_names=[main_base_flow_dataset.target],
                                              protected_attribute_names=[sensitive_attribute],
                                              favorable_label=1,
                                              unfavorable_label=0)
    test_binary_dataset = BinaryLabelDataset(df=test_df,
                                             label_names=[main_base_flow_dataset.target],
                                             protected_attribute_names=[sensitive_attribute],
                                             favorable_label=1,
                                             unfavorable_label=0)

    di = DisparateImpactRemover(repair_level=alpha, sensitive_attribute=sensitive_attribute)
    train_repaired_df, _ = di.fit_transform(train_binary_dataset).convert_to_dataframe()
    test_repaired_df , _ = di.fit_transform(test_binary_dataset).convert_to_dataframe()
    train_repaired_df.index = train_repaired_df.index.astype(dtype='int64')
    test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')

    main_base_flow_dataset.X_train_val = train_repaired_df.drop([main_base_flow_dataset.target, sensitive_attribute], axis=1)
    main_base_flow_dataset.y_train_val = train_repaired_df[main_base_flow_dataset.target]
    main_base_flow_dataset.X_test = test_repaired_df.drop([main_base_flow_dataset.target, sensitive_attribute], axis=1)
    main_base_flow_dataset.y_test = test_repaired_df[main_base_flow_dataset.target]

    # Apply Disparate Impact Remover on the out-of-domain datasets
    extra_test_sets = []
    for aligned_ds in extra_aligned_base_flow_datasets:
        test_df = aligned_ds.X_test
        test_df[aligned_ds.target] = aligned_ds.y_test

        test_binary_dataset = BinaryLabelDataset(df=test_df,
                                                 label_names=[aligned_ds.target],
                                                 protected_attribute_names=[sensitive_attribute],
                                                 favorable_label=1,
                                                 unfavorable_label=0)
        test_repaired_df , _ = di.fit_transform(test_binary_dataset).convert_to_dataframe()
        test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')

        extra_X_test = test_repaired_df.drop([aligned_ds.target, sensitive_attribute], axis=1)
        extra_y_test = test_repaired_df[aligned_ds.target]
        extra_test_sets.append((extra_X_test, extra_y_test, aligned_ds.init_features_df))

    return main_base_flow_dataset, extra_test_sets


def remove_correlation(init_base_flow_dataset, alpha):
    """
    Based on this tutorial: https://fairlearn.org/v0.8/auto_examples/plot_correlationremover_before_after.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    # sensitive_features_for_cr = ['cat__SEX_1', 'cat__SEX_2']
    sensitive_features_for_cr = ['cat__RAC1P_1', 'cat__RAC1P_2', 'cat__RAC1P_3', 'cat__RAC1P_5',
                                 'cat__RAC1P_6', 'cat__RAC1P_7', 'cat__RAC1P_8', 'cat__RAC1P_9']
    other_feature_columns = [col for col in base_flow_dataset.X_train_val.columns if col not in sensitive_features_for_cr]

    # Rearrange columns for consistency
    base_flow_dataset.X_train_val = base_flow_dataset.X_train_val[other_feature_columns + sensitive_features_for_cr]

    # Align columns in the in-domain test set with the train set
    for col in other_feature_columns:
        if col not in base_flow_dataset.X_test.columns:
            base_flow_dataset.X_test[col] = 0.0
    base_flow_dataset.X_test = base_flow_dataset.X_test[other_feature_columns + sensitive_features_for_cr]

    cr = CorrelationRemover(sensitive_feature_ids=sensitive_features_for_cr, alpha=alpha)
    # Fit and transform for the X_train_val set
    X_train_val_preprocessed = cr.fit_transform(base_flow_dataset.X_train_val)
    X_train_val_preprocessed = pd.DataFrame(X_train_val_preprocessed, columns=other_feature_columns, index=base_flow_dataset.X_train_val.index)
    for col in sensitive_features_for_cr:
        X_train_val_preprocessed[col] = base_flow_dataset.X_train_val[col]

    # Fit and transform for the X_test set
    X_test_preprocessed = cr.transform(base_flow_dataset.X_test)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=other_feature_columns, index=base_flow_dataset.X_test.index)
    for col in sensitive_features_for_cr:
        X_test_preprocessed[col] = base_flow_dataset.X_test[col]

    base_flow_dataset.X_train_val = X_train_val_preprocessed
    base_flow_dataset.X_test = X_test_preprocessed

    return base_flow_dataset


def remove_correlation_for_mult_test_sets(init_base_flow_dataset, alpha, extra_test_sets):
    """
    Based on this tutorial: https://fairlearn.org/v0.8/auto_examples/plot_correlationremover_before_after.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    sensitive_features_for_cr = ['cat__SEX_1', 'cat__SEX_2']
    other_feature_columns = [col for col in base_flow_dataset.X_train_val.columns if col not in sensitive_features_for_cr]

    # Rearrange columns for consistency
    base_flow_dataset.X_train_val = base_flow_dataset.X_train_val[other_feature_columns + sensitive_features_for_cr]

    # Align columns in the in-domain test set with the train set
    for col in other_feature_columns:
        if col not in base_flow_dataset.X_test.columns:
            base_flow_dataset.X_test[col] = 0.0
    base_flow_dataset.X_test = base_flow_dataset.X_test[other_feature_columns + sensitive_features_for_cr]

    # Align columns in the out-of-domain test sets with the train set
    for i in range(len(extra_test_sets)):
        cur_X_test, cur_y_test = extra_test_sets[i]
        for col in other_feature_columns:
            if col not in cur_X_test.columns:
                cur_X_test[col] = 0.0

        cur_X_test = cur_X_test[other_feature_columns + sensitive_features_for_cr]
        extra_test_sets[i] = (cur_X_test, cur_y_test)

    cr = CorrelationRemover(sensitive_feature_ids=sensitive_features_for_cr, alpha=alpha)
    # Fit and transform for the X_train_val set
    X_train_val_preprocessed = cr.fit_transform(base_flow_dataset.X_train_val)
    X_train_val_preprocessed = pd.DataFrame(X_train_val_preprocessed, columns=other_feature_columns, index=base_flow_dataset.X_train_val.index)
    X_train_val_preprocessed["cat__SEX_1"] = base_flow_dataset.X_train_val["cat__SEX_1"]
    X_train_val_preprocessed["cat__SEX_2"] = base_flow_dataset.X_train_val["cat__SEX_2"]

    # Fit and transform for the X_test set
    X_test_preprocessed = cr.transform(base_flow_dataset.X_test)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=other_feature_columns, index=base_flow_dataset.X_test.index)
    X_test_preprocessed["cat__SEX_1"] = base_flow_dataset.X_test["cat__SEX_1"]
    X_test_preprocessed["cat__SEX_2"] = base_flow_dataset.X_test["cat__SEX_2"]
    base_flow_dataset.X_train_val = X_train_val_preprocessed
    base_flow_dataset.X_test = X_test_preprocessed

    # Remove correlation in extra test sets
    preprocessed_extra_test_sets = []
    for X_test, y_test in extra_test_sets:
        cur_X_test_preprocessed = cr.transform(X_test)
        cur_X_test_preprocessed = pd.DataFrame(cur_X_test_preprocessed, columns=other_feature_columns, index=X_test.index)
        cur_X_test_preprocessed["cat__SEX_1"] = X_test["cat__SEX_1"]
        cur_X_test_preprocessed["cat__SEX_2"] = X_test["cat__SEX_2"]

        preprocessed_extra_test_sets.append((cur_X_test_preprocessed, y_test))

    return base_flow_dataset, preprocessed_extra_test_sets
