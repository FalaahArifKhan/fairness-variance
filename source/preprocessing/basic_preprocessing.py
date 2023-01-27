import pandas as pd
from sklearn.preprocessing import StandardScaler


__all__ = []


def get_dummies(data, categorical_columns, numerical_columns):
    """
    Return a dataset made by one-hot encoding for categorical columns and concatenate with numerical columns
    """
    feature_df = pd.get_dummies(data[categorical_columns], columns=categorical_columns)
    for col in numerical_columns:
        if col in data.columns:
            feature_df[col] = data[col]
    return feature_df


def make_features_dfs(X_train, X_test, dataset):
    X_train_features = get_dummies(X_train, dataset.categorical_columns, dataset.numerical_columns)
    X_test_features = get_dummies(X_test, dataset.categorical_columns, dataset.numerical_columns)

    # Align columns
    features_columns = list(set(X_train_features.columns) & set(X_test_features.columns))
    X_train_features = X_train_features[features_columns]
    X_test_features = X_test_features[features_columns]

    scaler = StandardScaler()
    X_train_features[dataset.numerical_columns] = scaler.fit_transform(X_train_features[dataset.numerical_columns])
    X_test_features[dataset.numerical_columns] = scaler.transform(X_test_features[dataset.numerical_columns])

    return X_train_features, X_test_features
