from source.analyzers.abstract_subgroups_analyzer import AbstractSubgroupsAnalyzer
from source.utils.common_helpers import confusion_matrix_metrics


class SubgroupsStatisticalBiasAnalyzer(AbstractSubgroupsAnalyzer):
    def __init__(self, X_test, y_test, sensitive_attributes_dct, test_groups=None):
        super().__init__(X_test, y_test, sensitive_attributes_dct, test_groups)

    def _compute_metrics(self, y_test, y_preds):
        return confusion_matrix_metrics(y_test, y_preds)