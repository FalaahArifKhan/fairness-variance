from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


SEED = 42
TEST_SET_FRACTION = 0.2
BOOTSTRAP_FRACTION = 0.8
DATASET_CONFIG = {
    'state': "GA",
    'year': 2018,
}


MODELS_CONFIG = [
    {
        'model_name': 'LogisticRegression',
        'model': LogisticRegression(random_state=SEED),
        'params': {
            'penalty': ['none', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': range(50, 251, 50),
        }
    },
    {
        'model_name': 'DecisionTreeClassifier',
        'model': DecisionTreeClassifier(random_state=SEED),
        'params': {
            "max_depth": [5, 10, 20, 30],
            "max_features": [0.6, 'sqrt'],
            "criterion": ["gini", "entropy"]
        }
    },
    {
        'model_name': 'RandomForestClassifier',
        'model': RandomForestClassifier(random_state=SEED),
        'params': {
            "max_depth": [3, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
            "n_estimators": [10, 20, 50, 100],
            "max_features": [0.6, 'auto', 'sqrt']
        }
    },
    {
        'model_name': 'XGBClassifier',
        'model': XGBClassifier(random_state=SEED, verbosity = 0),
        'params': {
            'learning_rate': [0.1],
            'n_estimators': [100, 200, 300],
            'max_depth': range(5, 16, 5),
            'objective':  ['binary:logistic'],
        }
    },
    {
        'model_name': 'KNeighborsClassifier',
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors' : [5, 7, 9, 11, 13, 15],
            'weights' : ['uniform', 'distance'],
            'metric' : ['minkowski', 'euclidean', 'manhattan']
        }
    },
    {
        'model_name': 'MLPClassifier_1L_100',
        'model': MLPClassifier(hidden_layer_sizes=(100,)),
        'params': {}
    },
    {
        'model_name': 'MLPClassifier_3L_100_50_100',
        'model': MLPClassifier(hidden_layer_sizes=(100, 50, 100)),
        'params': {}
    },
]
