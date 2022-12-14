from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from folktables import ACSEmployment


SEED=10
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
    # {
    #     'model_name': 'XGBClassifier',
    #     'model': XGBClassifier(random_state=SEED, verbosity = 0),
    #     'params': {
    #         'learning_rate': [0.1],
    #         'n_estimators': [200, 300],
    #         'max_depth': range(5, 16, 5),
    #         'objective':  ['binary:logistic'],
    #     }
    # }
]
