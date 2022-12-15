import os
import pandas as pd
from datetime import datetime, timezone

from utils.common_helpers import set_protected_groups, confusion_matrix_metrics


class GenericAnalyzer():
    def __init__(self, X_test, y_test, protected_groups, priv_values, test_groups=None, metric_names=None):
        self.protected_groups = protected_groups
        self.priv_values = priv_values
        self.X_test = X_test
        self.y_test = y_test
        self.test_groups = test_groups if test_groups \
            else set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        self.metric_names = metric_names
        self.results = {}

    def compute_metrics(self, y_preds, model_name, save_results=False, metric_names=None):
        y_pred_all = pd.Series(y_preds, index=self.y_test.index)
        '''
        if self.metric_names == None:
            if metric_names == None:
                raise Exception("metric_names is empty. Pass list of metric names to compute!")
            else:
                self.metric_names = metric_names
        '''
        results = {}
        results['overall'] = confusion_matrix_metrics(self.y_test, y_pred_all)

        for group_name in self.test_groups.keys():
            X_test_group = self.test_groups[group_name]
            results[group_name] = confusion_matrix_metrics(self.y_test[X_test_group.index],
                                                           y_pred_all[X_test_group.index])

        self.results = results
        return self.results

    def save_metrics_to_file(self, dataset_name, base_model_name,
                             save_dir_path=os.path.join('..', '..', 'results', 'hypothesis_space'),
                             exp_num=1):
        metrics_df = pd.DataFrame(self.results)
        os.makedirs(save_dir_path, exist_ok=True)

        now = datetime.now(timezone.utc)
        date_time_str = now.strftime("%Y%m%d__%H%M%S")
        filename = f"Hypothesis_Space_Metrics_{dataset_name}_Experiment_{exp_num}_{base_model_name}_{date_time_str}.csv"
        metrics_df = metrics_df.reset_index()
        metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)
