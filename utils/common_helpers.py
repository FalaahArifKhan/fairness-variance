import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix


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


def set_protected_groups(X_test, column_names, priv_values):
    groups={}
    groups[column_names[0]+'_'+column_names[1]+'_priv'], groups[column_names[0]+'_'+column_names[1]+'_dis'] = partition_by_group_intersectional(X_test, column_names, priv_values)
    groups[column_names[0]+'_priv'], groups[column_names[0]+'_dis'] =  partition_by_group_binary(X_test, column_names[0], priv_values[0])
    groups[column_names[1]+'_priv'], groups[column_names[1]+'_dis'] =  partition_by_group_binary(X_test, column_names[1], priv_values[1])
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
