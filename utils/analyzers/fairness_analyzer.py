import numpy as np
import pandas as pd

from utils.stability_utils import count_prediction_stats
from utils.analyzers.abstract_subgroups_analyzer import AbstractSubgroupsAnalyzer


class FairnessAnalyzer(AbstractSubgroupsAnalyzer):
    def __init__(self, X_test, y_test, protected_groups, priv_values, test_groups=None):
        super().__init__(X_test, y_test, protected_groups, priv_values, test_groups)
        self.overall_stability_metrics = None

    def set_overall_stability_metrics(self, overall_stability_metrics):
        self.overall_stability_metrics = overall_stability_metrics

    def _compute_metrics(self, y_test, group_models_predictions):
        _, _, prediction_stats = count_prediction_stats(y_test, group_models_predictions)
        precision = 4
        return {
            'General_Ensemble_Accuracy': np.round(prediction_stats.accuracy, precision),
            'Mean': np.round(np.mean(prediction_stats.means_lst), precision),
            'Std': np.round(np.mean(prediction_stats.stds_lst), precision),
            'IQR': np.round(np.mean(prediction_stats.iqr_lst), precision),
            'Entropy': np.round(np.mean(prediction_stats.entropy_lst), precision),
            'Jitter': np.round(prediction_stats.jitter, precision),
            'Per_Sample_Accuracy': np.round(np.mean(prediction_stats.per_sample_accuracy_lst), precision),
            'Label_Stability': np.round(np.mean(prediction_stats.label_stability_lst), precision),
        }

    def compute_subgroups_metrics(self, models_predictions, save_results, result_filename, save_dir_path):
        models_predictions = {
            model_idx: pd.Series(models_predictions[model_idx], index=self.y_test.index)
            for model_idx in models_predictions.keys()
        }

        results = {}
        results['overall'] = self.overall_stability_metrics
        for group_name in self.test_groups.keys():
            X_test_group = self.test_groups[group_name]
            group_models_predictions = {
                model_idx: models_predictions[model_idx][X_test_group.index].reset_index(drop=True)
                for model_idx in models_predictions.keys()
            }
            results[group_name] = self._compute_metrics(self.y_test[X_test_group.index].reset_index(drop=True),
                                                        group_models_predictions)

        self.fairness_metrics_dict = results
        if save_results:
            self.save_metrics_to_file(result_filename, save_dir_path)

        return self.fairness_metrics_dict
