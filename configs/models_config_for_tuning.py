import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


def get_folktables_employment_models_params_for_tuning(models_tuning_seed):
    return {
        # 'LGBMClassifier': {
        #     'model': LGBMClassifier(random_state=models_tuning_seed),
        #     'params': {
        #         'max_depth' : [i for i in range(3,12)],
        #         'num_leaves' : [int(x) for x in np.linspace(start = 20, stop = 3000, num = 10)],
        #         'min_data_in_leaf' : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
        #     }
        # },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'penalty': ['l1', 'l2'],
                'C' : [0.001, 0.01, 0.1, 1],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            }
        },
        # 'RandomForestClassifier': {
        #     'model': RandomForestClassifier(random_state=models_tuning_seed),
        #     'params': {
        #         # 'n_estimators': [100, 200, 500, 700, 1000],
        #         # 'n_estimators': [50, 100, 200, 300, 400],
        #         'n_estimators': [100, 200, 500],
        #         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4],
        #         'bootstrap': [True, False]
        #     }
        # },
#         'MLPClassifier': {
#             'model': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=models_tuning_seed, max_iter=1000),
#             'params': {
#                 'activation': ['logistic', 'tanh', 'relu'],
#                 'solver': ['lbfgs', 'sgd', 'adam'],
#                 'learning_rate': ['constant', 'invscaling', 'adaptive']
#             }
#         }
    }


def get_model_params_for_mult_repair_levels(models_tuning_seed):
    return {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=models_tuning_seed),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },
    }


def get_folktables_employment_models_params_for_tuning2(models_tuning_seed):
    return {
        'MLPClassifier': {
            'model': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        }
    }


def get_compas_models_params_for_tuning(models_tuning_seed):
    return {
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(random_state=models_tuning_seed),
            'params': {
                "max_depth": [20, 30],
                "min_samples_split" : [0.1],
                "max_features": ['sqrt'],
                "criterion": ["gini", "entropy"]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=models_tuning_seed),
            'params': {
                'penalty': ['l2'],
                'C' : [0.0001, 0.1, 1, 100],
                'solver': ['newton-cg', 'lbfgs'],
                'max_iter': [250],
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=models_tuning_seed),
            'params': {
                "max_depth": [6, 10],
                "min_samples_leaf": [1],
                "n_estimators": [50, 100],
                "max_features": [0.6]
            }
        },
        'XGBClassifier': {
            'model': XGBClassifier(random_state=models_tuning_seed, verbosity=0),
            'params': {
                'learning_rate': [0.1],
                'n_estimators': [200],
                'max_depth': [5, 7],
                'lambda':  [10, 100]
            }
        }
    }
