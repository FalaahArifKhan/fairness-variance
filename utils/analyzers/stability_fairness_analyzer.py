import os
import pandas as pd

from configs.constants import ModelSetting
from utils.analyzers.fairness_analyzer import FairnessAnalyzer
from utils.analyzers.batch_stability_analyzer import BatchStabilityAnalyzer


class StabilityFairnessAnalyzer:
    def __init__(self, model_setting, n_estimators: int, base_model, base_model_name: str,
                 X_train, y_train, X_test, y_test,
                 protected_groups, priv_values, test_groups: dict,
                 target_column: str, dataset_name: str):
        """

        :param model_setting: constant from ModelSetting
        :param n_estimators: the number of estimators for bootstrap
        :param base_model: the base model to analyze
        :param base_model_name: the model name to save metrics in correspondent file if needed
        :param X_train, y_train, X_test, y_test: default (train + val, test) splits
        :param protected_groups: protected groups (e.g., race or sex) to compute metrics
            for privilege and dis-privilege subgroups
        :param priv_values: list of privilege group values like (SEX_priv, RAC1P_priv) --> (1, 1)
        :param test_groups: dict, rows from X_test for each of subgroups; used for metrics computation for subgroups
        :param target_column: target column for classification
        :param dataset_name: the name of dataset, used for correct results naming
        """
        if model_setting == ModelSetting.BATCH:
            stability_analyzer = BatchStabilityAnalyzer(base_model, base_model_name,
                                                        X_train, y_train, X_test, y_test,
                                                        dataset_name, target_column, n_estimators)
        else:
            raise ValueError('model_setting is incorrect or not supported')

        self.dataset_name = stability_analyzer.dataset_name
        self.n_estimators = stability_analyzer.n_estimators
        self.base_model_name = stability_analyzer.base_model_name

        self.__stability_analyzer = stability_analyzer
        self.__fairness_analyzer = FairnessAnalyzer(X_test, y_test, protected_groups, priv_values, test_groups)
        self.stability_metrics_dct = dict()
        self.fairness_metrics_dct = dict()

    def compute_metrics(self, save_results, result_filename, save_dir_path, make_plots=True,):
        """
        Measure variance metrics for subgroups for the base model.
        Display stability plots for analysis if needed.
         Save results to a .csv file if needed.

        :param save_results: bool if we need to save metrics in a file
        :param make_plots: bool, if display plots for analysis
        """
        y_preds, y_test_true = self.__stability_analyzer.compute_metrics(make_plots, save_results=False)
        self.stability_metrics_dct = self.__stability_analyzer.get_metrics_dict()

        # Count and display fairness metrics
        self.__fairness_analyzer.set_overall_stability_metrics(self.stability_metrics_dct)
        self.fairness_metrics_dct = self.__fairness_analyzer.compute_subgroups_metrics(
            self.__stability_analyzer.models_predictions, save_results, result_filename, save_dir_path
        )

        return y_preds, pd.DataFrame(self.fairness_metrics_dct)
