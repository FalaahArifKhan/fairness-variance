from source.analyzers.abstract_subgroups_analyzer import AbstractSubgroupsAnalyzer
from source.utils.common_helpers import confusion_matrix_metrics


class SubgroupsStatisticalBiasAnalyzer(AbstractSubgroupsAnalyzer):
    """
    SubgroupsStatisticalBiasAnalyzer description.

    Parameters
    ----------
    X_test
        Processed features test set
    y_test
        Targets test set
    sensitive_attributes_dct
        A dictionary where keys are sensitive attributes names (including attributes intersections),
         and values are privilege values for these subgroups
    test_groups
        A dictionary where keys are sensitive attributes, and values input dataset rows
         that are correspondent to these sensitive attributes

    """
    def __init__(self, X_test, y_test, sensitive_attributes_dct, test_groups=None):
        super().__init__(X_test, y_test, sensitive_attributes_dct, test_groups)

    def _compute_metrics(self, y_test, y_preds):
        return confusion_matrix_metrics(y_test, y_preds)
