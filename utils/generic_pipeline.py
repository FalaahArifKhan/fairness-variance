import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils.common_helpers import *

class GenericPipeline():
    def __init__(self, dataset, protected_groups, priv_values, base_model=None, encoder=None, metric_names=None):
        self.features = dataset.features
        self.target = dataset.target
        self.categorical_columns = dataset.categorical_columns
        self.numerical_columns = dataset.numerical_columns
        self.X_data = dataset.X_data
        self.y_data = dataset.y_data
        self.protected_groups = protected_groups
        self.priv_values = priv_values
        self.columns_with_nulls = dataset.columns_with_nulls
        self.columns_without_nulls = list(set(self.features) - set(self.columns_with_nulls)) #For NullPredictors
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.test_groups = None
        self.encoder = encoder
        self.base_model = base_model
        self.pipeline = None
        self.metric_names = metric_names
        self.results = {}

    def create_train_test_val_split(self, SEED, sample_size=None):
        X_, X_test, y_, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=SEED)
        if (sample_size == None) or (sample_size > X_train.shape[0]):
            sample_size = X_train.shape[0]
        self.X_train = X_train.sample(n=sample_size, random_state=SEED)
        self.y_train = y_train[self.X_train.index]
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.test_groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def create_train_test_val_split_balanced(self, SEED, sample_size=None, group_by='SEX'):
        X_, X_test, y_, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=SEED)
        if (sample_size == None) or (sample_size > min(X_train[group_by].value_counts())):
            sample_size = min(X_train[group_by].value_counts())
        self.X_train = X_train.groupby(group_by).sample(n=sample_size, random_state=SEED)
        self.y_train = y_train[self.X_train.index]
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.test_groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def set_train_test_val_data_by_index(self, train_idx, test_idx, val_idx):
        self.X_train = self.X_data.loc[train_idx]
        self.y_train = self.y_data.loc[train_idx]
        self.X_test = self.X_data.loc[test_idx]
        self.y_test = self.y_data.loc[test_idx]
        self.X_val = self.X_data.loc[val_idx]
        self.y_val = self.y_data.loc[val_idx]
        self.groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val
    
    def construct_pipeline(self):
        return
    
    def set_pipeline(self, custom_pipeline):
        return
    
    def fit_model_batch(self, base_model):
        return
    
    def fit_model_incremental(self, base_model):
        return

    