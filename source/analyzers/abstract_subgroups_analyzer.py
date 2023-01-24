import os
import pandas as pd

from datetime import datetime, timezone
from abc import ABCMeta, abstractmethod


class AbstractSubgroupsAnalyzer(metaclass=ABCMeta):
    def __init__(self, X_test, y_test, sensitive_attributes_dct, test_groups):
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.X_test = X_test
        self.y_test = y_test
        self.test_groups = test_groups
        self.fairness_metrics_dict = {}

    @abstractmethod
    def _compute_metrics(self, y_test, y_preds):
        pass

    def compute_subgroups_metrics(self, y_preds, save_results, result_filename, save_dir_path):
        y_pred_all = pd.Series(y_preds, index=self.y_test.index)

        results = dict()
        results['overall'] = self._compute_metrics(self.y_test, y_pred_all)
        for group_name in self.test_groups.keys():
            X_test_group = self.test_groups[group_name]
            results[group_name] = self._compute_metrics(self.y_test[X_test_group.index], y_pred_all[X_test_group.index])
        self.fairness_metrics_dict = results

        if save_results:
            self.save_metrics_to_file(result_filename, save_dir_path)

        return self.fairness_metrics_dict

    def save_metrics_to_file(self, result_filename, save_dir_path):
        metrics_df = pd.DataFrame(self.fairness_metrics_dict)
        os.makedirs(save_dir_path, exist_ok=True)

        now = datetime.now(timezone.utc)
        date_time_str = now.strftime("%Y%m%d__%H%M%S")
        filename = f"{result_filename}_{date_time_str}.csv"
        metrics_df = metrics_df.reset_index()
        metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)
