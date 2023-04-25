import os
import logging
from datetime import datetime, timezone
from sklearn.metrics import confusion_matrix

from configs.constants import INTERSECTION_SIGN
from source.custom_classes.custom_logger import CustomHandler


def get_logger():
    logger = logging.getLogger('root')
    logger.setLevel('INFO')
    logging.disable(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(CustomHandler())

    return logger


def validate_config(config_obj):
    if not isinstance(config_obj.dataset_name, str):
        raise ValueError('dataset_name must be string')
    elif not isinstance(config_obj.test_set_fraction, float):
        raise ValueError('test_set_fraction must be float in [0.0, 1.0] range')
    elif not isinstance(config_obj.bootstrap_fraction, float):
        raise ValueError('bootstrap_fraction must be float in [0.0, 1.0] range')
    elif not isinstance(config_obj.n_estimators, int) or config_obj.n_estimators <= 0:
        raise ValueError('n_estimators must be integer greater than 0')
    elif not isinstance(config_obj.runs_seed_lst, list):
        raise ValueError('runs_seed_lst must be python list')
    elif not isinstance(config_obj.sensitive_attributes_dct, dict):
        raise ValueError('sensitive_attributes_dct must be python dictionary')
    elif isinstance(config_obj.sensitive_attributes_dct, dict):
        for sensitive_attr in config_obj.sensitive_attributes_dct.keys():
            if sensitive_attr.count(INTERSECTION_SIGN) > 1:
                raise ValueError('sensitive_attributes_dct must contain only plain sensitive attributes or '
                                 'intersections of two sensitive attributes (not more attributes intersections)')


def save_metrics_to_file(metrics_df, result_filename, save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_time_str = now.strftime("%Y%m%d__%H%M%S")
    filename = f"{result_filename}_{date_time_str}.csv"
    metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)


def partition_by_group_intersectional(df, attr1, attr2, priv_value1, priv_value2):
    priv = df[(df[attr1] == priv_value1) & (df[attr2] == priv_value2)]
    dis = df[(df[attr1] != priv_value1) & (df[attr2] != priv_value2)]
    return priv, dis


def partition_by_group_binary(df, column_name, priv_value):
    priv = df[df[column_name] == priv_value]
    dis = df[df[column_name] != priv_value]
    if len(priv)+len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis


def check_sensitive_attrs_in_columns(df_columns, sensitive_attributes_dct):
    for sensitive_attr in sensitive_attributes_dct.keys():
        if sensitive_attr not in df_columns:
            return False

    return True


def create_test_groups(X_test, full_df, sensitive_attributes_dct):
    # Check if input sensitive attributes are in X_test.columns.
    # If no, add them only to create test groups
    if check_sensitive_attrs_in_columns(X_test.columns, sensitive_attributes_dct):
        X_test_with_sensitive_attrs = X_test
    else:
        plain_sensitive_attributes = [attr for attr in sensitive_attributes_dct.keys() if INTERSECTION_SIGN not in attr]
        cols_with_sensitive_attrs = set(list(X_test.columns) + plain_sensitive_attributes)
        X_test_with_sensitive_attrs = full_df[cols_with_sensitive_attrs].loc[X_test.index]

    groups = dict()
    for attr in sensitive_attributes_dct.keys():
        if INTERSECTION_SIGN in attr:
            if attr.count(INTERSECTION_SIGN) == 1:
                attr1, attr2 = attr.split(INTERSECTION_SIGN)
                groups[attr1 + INTERSECTION_SIGN + attr2 + '_priv'], groups[attr1 + INTERSECTION_SIGN + attr2 + '_dis'] = \
                    partition_by_group_intersectional(X_test_with_sensitive_attrs, attr1, attr2,
                                                      sensitive_attributes_dct[attr1], sensitive_attributes_dct[attr2])
        else:
            groups[attr + '_priv'], groups[attr + '_dis'] = \
                partition_by_group_binary(X_test_with_sensitive_attrs, attr, sensitive_attributes_dct[attr])

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