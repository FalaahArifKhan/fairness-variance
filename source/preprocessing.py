import copy
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fairlearn.preprocessing import CorrelationRemover


def get_simple_preprocessor(data_loader):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('num', StandardScaler(), data_loader.numerical_columns),
    ])


def remove_correlation(init_base_flow_dataset, alpha):
    """
    Based on this tutorial: https://fairlearn.org/v0.8/auto_examples/plot_correlationremover_before_after.html

    :param init_base_flow_dataset:
    :param alpha:
    :return:
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
