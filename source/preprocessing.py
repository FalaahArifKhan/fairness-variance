import copy
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from fairlearn.preprocessing import CorrelationRemover
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
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


def preprocess_mult_data_loaders_for_disp_imp(data_loaders: list, test_set_fraction: float, experiment_seed: int):
    base_flow_datasets = []
    for cur_data_loader in data_loaders:
        # Add RACE column for DisparateImpactRemover and remove 'SEX', 'RAC1P' to create a blind estimator
        init_data_loader = copy.deepcopy(cur_data_loader)
        cur_data_loader.categorical_columns = [col for col in cur_data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        cur_data_loader.X_data['RACE'] = cur_data_loader.X_data['RAC1P'].apply(lambda x: 1 if x == '1' else 0)
        cur_data_loader.full_df = cur_data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        cur_data_loader.X_data = cur_data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

        # Preprocess the dataset using the defined preprocessor
        # column_transformer = get_simple_preprocessor(cur_data_loader)
        column_transformer = get_preprocessor_for_diabetes(cur_data_loader)
        cur_base_flow_dataset = preprocess_dataset(cur_data_loader, column_transformer, test_set_fraction, experiment_seed)
        cur_base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
        cur_base_flow_dataset.X_train_val['RACE'] = cur_data_loader.X_data.loc[cur_base_flow_dataset.X_train_val.index, 'RACE']
        cur_base_flow_dataset.X_test['RACE'] = cur_data_loader.X_data.loc[cur_base_flow_dataset.X_test.index, 'RACE']

        base_flow_datasets.append(cur_base_flow_dataset)

    init_base_flow_dataset, extra_base_flow_datasets = base_flow_datasets[0], base_flow_datasets[1:]
    return init_base_flow_dataset, extra_base_flow_datasets


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
        extra_test_sets.append((extra_X_test, extra_y_test))

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
