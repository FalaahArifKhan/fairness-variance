import os
import logging
from datetime import datetime, timezone
from sklearn.metrics import confusion_matrix

from source.custom_classes.custom_logger import CustomHandler


def get_logger():
    logger = logging.getLogger('root')
    logger.setLevel('INFO')
    logging.disable(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(CustomHandler())

    return logger


def save_metrics_to_file(metrics_df, result_filename, save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_time_str = now.strftime("%Y%m%d__%H%M%S")
    filename = f"{result_filename}_{date_time_str}.csv"
    # metrics_df = metrics_df.reset_index()
    # metrics_df = metrics_df.rename(columns={"index": "Metric"})
    metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)


def partition_by_group_intersectional(df, column_names, priv_values):
    priv = df[(df[column_names[0]] == priv_values[0]) & (df[column_names[1]] == priv_values[1])]
    dis = df[(df[column_names[0]] != priv_values[0]) & (df[column_names[1]] != priv_values[1])]
    return priv, dis


def partition_by_group_binary(df, column_name, priv_value):
    priv = df[df[column_name] == priv_value]
    dis = df[df[column_name] != priv_value]
    if len(priv)+len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis


def check_sensitive_attrs_in_columns(df_columns, sensitive_attributes):
    for sensitive_attr in sensitive_attributes:
        if sensitive_attr not in df_columns:
            return False

    return True


def create_test_groups(X_test, full_df, sensitive_attributes, priv_values):
    # Check if input sensitive attributes are in X_test.columns.
    # If no, add them only to create test groups
    if check_sensitive_attrs_in_columns(X_test.columns, sensitive_attributes):
        X_test_with_sensitive_attrs = X_test
    else:
        cols_with_sensitive_attrs = set(list(X_test.columns) + sensitive_attributes)
        X_test_with_sensitive_attrs = full_df[cols_with_sensitive_attrs].loc[X_test.index]

    groups = {}
    groups[sensitive_attributes[0] + '_' + sensitive_attributes[1] + '_priv'], groups[sensitive_attributes[0] + '_' + sensitive_attributes[1] + '_dis'] = \
        partition_by_group_intersectional(X_test_with_sensitive_attrs, sensitive_attributes, priv_values)
    groups[sensitive_attributes[0] + '_priv'], groups[sensitive_attributes[0] + '_dis'] = \
        partition_by_group_binary(X_test_with_sensitive_attrs, sensitive_attributes[0], priv_values[0])
    groups[sensitive_attributes[1] + '_priv'], groups[sensitive_attributes[1] + '_dis'] = \
        partition_by_group_binary(X_test_with_sensitive_attrs, sensitive_attributes[1], priv_values[1])

    return groups


def confusion_matrix_metrics(y_true, y_preds):
    metrics={}
    TN, FP, FN, TP = confusion_matrix(y_true, y_preds).ravel()
    metrics['TPR'] = TP/(TP+FN)
    metrics['TNR'] = TN/(TN+FP)
    metrics['PPV'] = TP/(TP+FP)
    metrics['FNR'] = FN/(FN+TP)
    metrics['FPR'] = FP/(FP+TN)
    metrics['Accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    metrics['F1'] = (2*TP)/(2*TP+FP+FN)
    metrics['Selection-Rate'] = (TP+FP)/(TP+FP+TN+FN)
    metrics['Positive-Rate'] = (TP+FP)/(TP+FN)

    return metrics
