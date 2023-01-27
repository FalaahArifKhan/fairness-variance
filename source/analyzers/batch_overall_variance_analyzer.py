from source.analyzers.abstract_overall_variance_analyzer import AbstractOverallVarianceAnalyzer


class BatchOverallVarianceAnalyzer(AbstractOverallVarianceAnalyzer):
    """
    BatchOverallVarianceAnalyzer description.

    Parameters
    ----------
    base_model
        Base model for stability measuring
    base_model_name
        Model name like 'HoeffdingTreeClassifier' or 'LogisticRegression'
    bootstrap_fraction
        [0-1], fraction from train_pd_dataset for fitting an ensemble of base models
    X_train
        Processed features train set
    y_train
        Targets train set
    X_test
        Processed features test set
    y_test
        Targets test set
    target_column
        Name of the target column
    dataset_name
        Name of dataset, used for correct results naming
    n_estimators
        Number of estimators in ensemble to measure base_model stability

    """
    def __init__(self, base_model, base_model_name: str, bootstrap_fraction: float,
                 X_train, y_train, X_test, y_test, target_column: str,
                 dataset_name: str, n_estimators: int):
        super().__init__(base_model, base_model_name, bootstrap_fraction,
                         X_train, y_train, X_test, y_test, dataset_name, n_estimators)
        self.target_column = target_column

    def _fit_model(self, classifier, X_train, y_train):
        return classifier.fit(X_train, y_train)

    def _batch_predict(self, classifier, X_test):
        return classifier.predict(X_test)

    def _batch_predict_proba(self, classifier, X_test):
        return classifier.predict_proba(X_test)[:, 0]
