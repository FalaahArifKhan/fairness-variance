from configs.config import BOOTSTRAP_FRACTION
from utils.analyzers.abstract_stability_analyzer import AbstractStabilityAnalyzer


class BatchStabilityAnalyzer(AbstractStabilityAnalyzer):
    def __init__(self, base_model, base_model_name: str,
                 X_train, y_train, X_test, y_test, target_column: str,
                 dataset_name: str, n_estimators: int):
        """
        :param target_column: name of the y-column
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        super().__init__(base_model, base_model_name, BOOTSTRAP_FRACTION,
                         X_train, y_train, X_test, y_test, dataset_name, n_estimators)
        self.target_column = target_column

    def _fit_model(self, classifier, X_train, y_train):
        return classifier.fit(X_train, y_train)

    def _batch_predict(self, classifier, X_test):
        return classifier.predict(X_test)

    def _batch_predict_proba(self, classifier, X_test):
        return classifier.predict_proba(X_test)[:, 0]
