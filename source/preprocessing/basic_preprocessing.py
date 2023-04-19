from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from source.preprocessing.null_imputer import NullImputer


def get_simple_preprocessor(data_loader):
    return ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('numerical_features', StandardScaler(), data_loader.numerical_columns),
    ])


def get_null_imputer_preprocessor(data_loader, categorical_strategy="mode", numerical_strategy="median"):
    categorial_null_columns = list(set(data_loader.columns_with_nulls).intersection(data_loader.categorical_columns))
    numerical_null_columns = list(set(data_loader.columns_with_nulls).intersection(data_loader.numerical_columns))

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", NullImputer(categorial_null_columns, how=categorical_strategy)),
            ("encoder", OneHotEncoder(sparse=False)),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", NullImputer(numerical_null_columns, how=numerical_strategy)),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, data_loader.categorical_columns),
            ("num", numeric_transformer, data_loader.numerical_columns),
        ]
    )

    return preprocessor
