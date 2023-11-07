import copy

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.datasets import CreditCardDefaultDataset


def get_simple_preprocessor(data_loader):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
        ('num', StandardScaler(), data_loader.numerical_columns),
    ])


data_loader = CreditCardDefaultDataset()
data_loader.X_data.head()

init_data_loader = copy.deepcopy(data_loader)
test_set_fraction = 0.2
experiment_seed = 42

data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('sex')]
data_loader.X_data['sex_binary'] = data_loader.X_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
data_loader.full_df = data_loader.full_df.drop(['sex'], axis=1)
data_loader.X_data = data_loader.X_data.drop(['sex'], axis=1)

# Preprocess the dataset using the defined preprocessor
column_transformer = get_simple_preprocessor(data_loader)
base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
base_flow_dataset.X_train_val['sex_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'sex_binary']
base_flow_dataset.X_test['sex_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'sex_binary']

mlp_params = {
            'model': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=42, max_iter=1000),
            'params': {
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        }

xgb_params = {
    'model': XGBClassifier(random_state=42, verbosity=0),
    'params': {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01,0.05,0.1],
    'max_depth': [3, 5, 10],
}
}

lr_params = {
    'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'penalty': ['l1', 'l2'],
                'C' : [0.001, 0.01, 0.1, 1],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            }
}

grid_search_mlp = GridSearchCV(
                            estimator=mlp_params['model'],
                            param_grid=mlp_params['params'],
                            scoring={
                                "F1_Score": make_scorer(f1_score, average='macro'),
                                "Accuracy_Score": make_scorer(accuracy_score),
                            },
                            refit="F1_Score",
                            n_jobs=-1,
                            verbose=4
                            )

grid_search = GridSearchCV(
                            estimator=xgb_params['model'],
                            param_grid=xgb_params['params'],
                            scoring={
                                "F1_Score": make_scorer(f1_score, average='macro'),
                                "Accuracy_Score": make_scorer(accuracy_score),
                            },
                            refit="F1_Score",
                            n_jobs=-1,
                            verbose=4
                            )

grid_search_lr = GridSearchCV(
                            estimator=lr_params['model'],
                            param_grid=lr_params['params'],
                            scoring={
                                "F1_Score": make_scorer(f1_score, average='macro'),
                                "Accuracy_Score": make_scorer(accuracy_score),
                            },
                            refit="F1_Score",
                            n_jobs=-1,
                            verbose=2
                            )

grid_search_lr.fit(base_flow_dataset.X_train_val, base_flow_dataset.y_train_val.ravel())

model = grid_search_lr.best_estimator_
print("Accuracy on test set: ", accuracy_score(base_flow_dataset.y_test, model.predict(base_flow_dataset.X_test)))
print("F1 on test set: ", f1_score(base_flow_dataset.y_test, model.predict(base_flow_dataset.X_test), average='macro'))
