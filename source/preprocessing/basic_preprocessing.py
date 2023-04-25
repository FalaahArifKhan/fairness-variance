from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from virny.custom_classes.base_dataset import BaseFlowDataset

from source.utils.common_helpers import detect_outliers_std
from source.preprocessing.null_imputer import NullImputer
from source.datasets.base import BaseDataLoader


def get_simple_preprocessor(data_loader):
    return ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('numerical_features', StandardScaler(), data_loader.numerical_columns),
    ])


def get_null_imputer_preprocessor(data_loader, categorical_strategy="mode", numerical_strategy="median",
                                  categorical_trimmed=0.0, numerical_trimmed=0.0):
    categorial_null_columns = list(set(data_loader.columns_with_nulls).intersection(data_loader.categorical_columns))
    numerical_null_columns = list(set(data_loader.columns_with_nulls).intersection(data_loader.numerical_columns))

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", NullImputer(categorial_null_columns, how=categorical_strategy, trimmed=categorical_trimmed)),
            ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore')),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", NullImputer(numerical_null_columns, how=numerical_strategy, trimmed=numerical_trimmed)),
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


def preprocess_experiment_dataset(data_loader: BaseDataLoader, column_transformer: ColumnTransformer,
                                  test_set_fraction: float, dataset_split_seed: int):
    if test_set_fraction < 0.0 or test_set_fraction > 1.0:
        raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

    # Split and preprocess the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                test_size=test_set_fraction,
                                                                random_state=dataset_split_seed)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    X_train_features = column_transformer.fit_transform(X_train_val)
    X_test_features = column_transformer.transform(X_test)

    base_flow_dataset = BaseFlowDataset(init_features_df=data_loader.full_df.drop(data_loader.target, axis=1, errors='ignore'),
                                        X_train_val=X_train_features,
                                        X_test=X_test_features,
                                        y_train_val=y_train_val,
                                        y_test=y_test,
                                        target=data_loader.target,
                                        numerical_columns=data_loader.numerical_columns,
                                        categorical_columns=data_loader.categorical_columns)

    return base_flow_dataset, (X_train_val, X_test, y_train_val, y_test), column_transformer


def create_stress_testing_sets(original_X_test, original_y_test, error_injector, injector_config_lst, fitted_column_transformer):
    # Create test sets for model stress testing
    extra_test_sets_lst = []
    for percentage_var in injector_config_lst:
        X_test = original_X_test.copy(deep=True)
        error_injector.set_percentage_var(percentage_var)
        error_injector.increment_seed()
        print('error_injector.seed -- ', error_injector.seed)
        transformed_X_test = error_injector.transform(X_test)  # Use only transform without fit
        print('transformed_X_test:\n', transformed_X_test.isna().sum())
        new_X_test_features = fitted_column_transformer.transform(transformed_X_test)  # Preprocess the feature set

        extra_test_sets_lst.append((new_X_test_features, original_y_test))

    return extra_test_sets_lst
