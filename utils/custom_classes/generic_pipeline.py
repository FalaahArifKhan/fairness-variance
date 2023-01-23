from sklearn.model_selection import train_test_split

from utils.preprocessing import make_features_dfs
from utils.common_helpers import create_test_groups


class GenericPipeline:
    def __init__(self, dataset, sensitive_attributes, priv_values, base_model=None, encoder=None, metric_names=None):
        self.full_df = dataset.dataset
        self.features = dataset.features
        self.target = dataset.target
        self.categorical_columns = dataset.categorical_columns
        self.numerical_columns = dataset.numerical_columns
        self.X_data = dataset.X_data
        self.y_data = dataset.y_data
        self.sensitive_attributes = sensitive_attributes
        self.priv_values = priv_values
        self.columns_with_nulls = dataset.columns_with_nulls
        self.columns_without_nulls = list(set(self.features) - set(self.columns_with_nulls)) # For NullPredictors
        self.encoder = encoder
        self.base_model = base_model
        self.metric_names = metric_names

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.X_train_val = None
        self.y_train_val = None
        self.test_groups = None
        self.pipeline = None
        self.results = {}

    def create_preprocessed_train_test_split(self, dataset, test_set_fraction, seed):
        X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data,
                                                            test_size=test_set_fraction,
                                                            random_state=seed)
        print("Baseline X_train shape: ", X_train.shape)
        print("Baseline X_test shape: ", X_test.shape)

        X_train_features, X_test_features = make_features_dfs(X_train, X_test, dataset)
        self.X_train_val = X_train_features
        self.X_test = X_test_features
        self.y_train_val = y_train
        self.y_test = y_test
        self.test_groups = create_test_groups(X_test, self.full_df, self.sensitive_attributes, self.priv_values)

        return self.X_train_val, self.y_train_val, self.X_test, self.y_test

    def create_train_test_val_split(self, seed, sample_size=None):
        X_, X_test, y_, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=seed)
        if (sample_size is None) or (sample_size > X_train.shape[0]):
            sample_size = X_train.shape[0]
        self.X_train = X_train.sample(n=sample_size, random_state=seed)
        self.y_train = y_train[self.X_train.index]
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.test_groups = create_test_groups(self.X_test, self.full_df, self.sensitive_attributes, self.priv_values)

        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def create_train_test_val_split_balanced(self, seed, sample_size=None, group_by='SEX'):
        X_, X_test, y_, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=seed)
        if (sample_size is None) or (sample_size > min(X_train[group_by].value_counts())):
            sample_size = min(X_train[group_by].value_counts())
        self.X_train = X_train.groupby(group_by).sample(n=sample_size, random_state=seed)
        self.y_train = y_train[self.X_train.index]
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.test_groups = create_test_groups(self.X_test, self.full_df, self.sensitive_attributes, self.priv_values)

        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def set_train_test_val_data_by_index(self, train_idx, test_idx, val_idx):
        self.X_train = self.X_data.loc[train_idx]
        self.y_train = self.y_data.loc[train_idx]
        self.X_test = self.X_data.loc[test_idx]
        self.y_test = self.y_data.loc[test_idx]
        self.X_val = self.X_data.loc[val_idx]
        self.y_val = self.y_data.loc[val_idx]
        self.groups = create_test_groups(self.X_test, self.full_df, self.sensitive_attributes, self.priv_values)

        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val
    
    def construct_pipeline(self):
        return
    
    def set_pipeline(self, custom_pipeline):
        return
    
    def fit_model_batch(self, base_model):
        return
    
    def fit_model_incremental(self, base_model):
        return
