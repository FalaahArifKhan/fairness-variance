from copy import deepcopy

from virny.custom_classes.metrics_composer import MetricsComposer


class ExperimentsComposer:
    def __init__(self, models_metrics_dct: dict, sensitive_attrs: list):
        self.models_metrics_dct = deepcopy(models_metrics_dct)
        self.sensitive_attrs = sensitive_attrs
        self.sensitive_attributes_dct = {attr: None for attr in sensitive_attrs}

    def create_exp_subgroup_metrics_dct(self):
        structured_exp_results_dct = dict()
        for model_name in self.models_metrics_dct.keys():
            structured_exp_results_dct[model_name] = dict()
            distinct_preprocessing_techniques_lst = self.models_metrics_dct[model_name]['Preprocessing_Technique'].unique()

            for preprocessing_technique in distinct_preprocessing_techniques_lst:
                structured_exp_results_dct[model_name][preprocessing_technique] = dict()
                distinct_exp_iters_lst = \
                    self.models_metrics_dct[model_name][
                            (self.models_metrics_dct[model_name]['Preprocessing_Technique'] == preprocessing_technique)
                        ]['Experiment_Iteration'].unique()
                injector_config_lst = eval(self.models_metrics_dct[model_name][
                                               (self.models_metrics_dct[model_name]['Preprocessing_Technique'] == preprocessing_technique)
                                           ]['Injector_Config_Lst'].unique()[0])

                for exp_iter in distinct_exp_iters_lst:
                    structured_exp_results_dct[model_name][preprocessing_technique][exp_iter] = dict()
                    distinct_test_set_idx = \
                        self.models_metrics_dct[model_name][
                                (self.models_metrics_dct[model_name]['Preprocessing_Technique'] == preprocessing_technique) &
                                (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter)
                            ]['Test_Set_Index'].unique()

                    for test_set_idx in distinct_test_set_idx:
                        model_metrics_per_run_per_percentage = \
                            self.models_metrics_dct[model_name][
                                    (self.models_metrics_dct[model_name]['Preprocessing_Technique'] == preprocessing_technique) &
                                    (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter) &
                                    (self.models_metrics_dct[model_name]['Test_Set_Index'] == test_set_idx)
                                ]
                        if test_set_idx == 0:
                            dct_pct_key = 0.0 if not isinstance(injector_config_lst[0], str) else 'baseline'
                        else:
                            dct_pct_key = injector_config_lst[test_set_idx - 1]  # minus 1 since we do not save 0% errors in injector_config_lst

                        structured_exp_results_dct[model_name][preprocessing_technique][exp_iter][dct_pct_key] =\
                            model_metrics_per_run_per_percentage

        return structured_exp_results_dct

    def compose_group_metrics(self, structured_exp_results_dct):
        structured_exp_avg_group_metrics_dct = dict()
        for model_name in structured_exp_results_dct.keys():
            for preprocessing_technique in structured_exp_results_dct[model_name].keys():
                for exp_iter in structured_exp_results_dct[model_name][preprocessing_technique].keys():
                    for percentage in structured_exp_results_dct[model_name][preprocessing_technique][exp_iter].keys():
                        model_subgroup_metrics_df = structured_exp_results_dct[model_name][preprocessing_technique][exp_iter][percentage]
                        model_subgroup_metrics_df = model_subgroup_metrics_df.drop(['Bootstrap_Model_Seed', 'Record_Create_Date_Time'], axis=1)
                        metrics_composer = MetricsComposer(
                            {model_name: model_subgroup_metrics_df},
                            self.sensitive_attributes_dct
                        )
                        structured_exp_avg_group_metrics_dct.setdefault(model_name, {})\
                            .setdefault(preprocessing_technique, {})\
                            .setdefault(exp_iter, {})[percentage] = metrics_composer.compose_metrics()

        return structured_exp_avg_group_metrics_dct
