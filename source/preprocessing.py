import copy
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fairlearn.preprocessing import CorrelationRemover
from virny.preprocessing.basic_preprocessing import preprocess_dataset


def get_simple_preprocessor(data_loader):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('num', StandardScaler(), data_loader.numerical_columns),
    ])


def create_extra_test_sets(extra_data_loaders: list, column_transformer, test_set_fraction, seed):
    extra_test_sets = []
    for exp_data_loader in extra_data_loaders:
        cur_column_transformer = copy.deepcopy(column_transformer)
        exp_base_flow_dataset = preprocess_dataset(exp_data_loader, cur_column_transformer, test_set_fraction, seed)
        extra_test_sets.append((exp_base_flow_dataset.X_test, exp_base_flow_dataset.y_test))

    return extra_test_sets


def remove_correlation(init_base_flow_dataset, alpha):
    """
    Based on this tutorial: https://fairlearn.org/v0.8/auto_examples/plot_correlationremover_before_after.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)

    sensitive_features_for_cr = ['cat__SEX_1', 'cat__SEX_2']
    other_features_train_val = [col for col in base_flow_dataset.X_train_val.columns if col not in sensitive_features_for_cr]
    other_features_test = [col for col in base_flow_dataset.X_test.columns if col not in sensitive_features_for_cr]

    # Rearrange columns for consistency
    base_flow_dataset.X_train_val = base_flow_dataset.X_train_val[other_features_train_val + sensitive_features_for_cr]
    base_flow_dataset.X_test = base_flow_dataset.X_test[other_features_test + sensitive_features_for_cr]

    cr = CorrelationRemover(sensitive_feature_ids=sensitive_features_for_cr, alpha=alpha)
    # Fit and transform for the X_train_val set
    X_train_val_preprocessed = cr.fit_transform(base_flow_dataset.X_train_val)
    X_train_val_preprocessed = pd.DataFrame(X_train_val_preprocessed, columns=other_features_train_val, index=base_flow_dataset.X_train_val.index)
    X_train_val_preprocessed["cat__SEX_1"] = base_flow_dataset.X_train_val["cat__SEX_1"]
    X_train_val_preprocessed["cat__SEX_2"] = base_flow_dataset.X_train_val["cat__SEX_2"]

    # Fit and transform for the X_test set
    X_test_preprocessed = cr.transform(base_flow_dataset.X_test)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=other_features_test, index=base_flow_dataset.X_test.index)
    X_test_preprocessed["cat__SEX_1"] = base_flow_dataset.X_test["cat__SEX_1"]
    X_test_preprocessed["cat__SEX_2"] = base_flow_dataset.X_test["cat__SEX_2"]

    base_flow_dataset.X_train_val = X_train_val_preprocessed
    base_flow_dataset.X_test = X_test_preprocessed

    return base_flow_dataset


def remove_correlation_for_mult_test_sets(init_base_flow_dataset, alpha, extra_test_sets):
    """
    Based on this tutorial: https://fairlearn.org/v0.8/auto_examples/plot_correlationremover_before_after.html

    """
    base_flow_dataset = copy.deepcopy(init_base_flow_dataset)
    sensitive_features_for_cr = ['cat__SEX_1', 'cat__SEX_2']

    # Align feature columns in all test sets
    feature_columns_set = set(base_flow_dataset.X_train_val.columns) & set(base_flow_dataset.X_test.columns)
    for test_set in extra_test_sets:
        feature_columns_set &= set(test_set[0].columns)

    feature_columns_set.remove('cat__SEX_1')
    feature_columns_set.remove('cat__SEX_2')
    other_feature_columns = list(feature_columns_set)

    # Rearrange columns for consistency
    base_flow_dataset.X_train_val = base_flow_dataset.X_train_val[other_feature_columns + sensitive_features_for_cr]
    base_flow_dataset.X_test = base_flow_dataset.X_test[other_feature_columns + sensitive_features_for_cr]

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
        X_test = X_test[other_feature_columns + sensitive_features_for_cr]

        cur_X_test_preprocessed = cr.transform(X_test)
        cur_X_test_preprocessed = pd.DataFrame(cur_X_test_preprocessed, columns=other_feature_columns, index=X_test.index)
        cur_X_test_preprocessed["cat__SEX_1"] = X_test["cat__SEX_1"]
        cur_X_test_preprocessed["cat__SEX_2"] = X_test["cat__SEX_2"]

        preprocessed_extra_test_sets.append((cur_X_test_preprocessed, y_test))

    return base_flow_dataset, preprocessed_extra_test_sets
