import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_transformer(X_train, categorical_columns, numerical_columns):
    return ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(categories=[list(set(X_train[col])) for col in categorical_columns], sparse=False),
             categorical_columns),
            ('numerical_features', StandardScaler(), numerical_columns)
        ])


def get_dummies(data, categorical_columns, numerical_columns):
    """
    Return a dataset made by one-hot encoding for categorical columns and concatenate with numerical columns
    """
    feature_df = pd.get_dummies(data[categorical_columns], columns=categorical_columns)
    for col in numerical_columns:
        if col in data.columns:
            feature_df[col] = data[col]
    return feature_df


def create_tuned_base_model(init_model, model_name, models_tuned_params_df):
    model_params = eval(models_tuned_params_df.loc[models_tuned_params_df['Model_Name'] == model_name,
                                                   'Model_Best_Params'].iloc[0])
    return init_model.set_params(**model_params)


def make_features_dfs(X_train, X_test, dataset):
    X_train_features = get_dummies(X_train, dataset.categorical_columns, dataset.numerical_columns)
    X_test_features = get_dummies(X_test, dataset.categorical_columns, dataset.numerical_columns)

    # Align columns
    features_columns = list(set(X_train_features.columns) & set(X_test_features.columns))
    X_train_features = X_train_features[features_columns]
    X_test_features = X_test_features[features_columns]

    scaler = StandardScaler()
    X_train_features[features_columns] = scaler.fit_transform(X_train_features[features_columns])
    X_test_features[features_columns] = scaler.transform(X_test_features[features_columns])

    return X_train_features, X_test_features


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
    groups[column_names[0]+'_priv'], groups[column_names[0]+'_dis'] = partition_by_group_binary(X_test, column_names[0], priv_values[0])
    groups[column_names[1]+'_priv'], groups[column_names[1]+'_dis'] = partition_by_group_binary(X_test, column_names[1], priv_values[1])
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
