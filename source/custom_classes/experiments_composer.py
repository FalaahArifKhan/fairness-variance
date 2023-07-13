from copy import deepcopy

from virny.custom_classes.metrics_composer import MetricsComposer


class ExperimentsComposer:
    def __init__(self, models_metrics_dct: dict, sensitive_attrs: list):
        self.models_metrics_dct = deepcopy(models_metrics_dct)
        self.sensitive_attrs = sensitive_attrs
        self.sensitive_attributes_dct = {attr: None for attr in sensitive_attrs}

    def create_exp_subgroup_metrics_dct(self):
        """
        Create a subgroup metrics dictionary based on self.models_metrics_dct.
        The hierarchy of the created dictionary is the following:
            model_name: {
                preprocessing_technique: {
                    exp_iter: {
                        dct_pct_key: pandas df with subgroup metrics for this model, preprocessing technique, and experiment iteration
                    }
                }
            }

        Returns
        -------
        Return a new structured dictionary with subgroup metrics per each model, preprocessing technique,
         and experiment iteration.

        """
        structured_exp_results_dct = dict()
        for model_name in self.models_metrics_dct.keys():
            structured_exp_results_dct[model_name] = dict()
            distinct_exp_iters_lst = self.models_metrics_dct[model_name]['Experiment_Iteration'].unique()

            for exp_iter in distinct_exp_iters_lst:
                structured_exp_results_dct[model_name][exp_iter] = dict()
                distinct_intervention_params = \
                    self.models_metrics_dct[model_name][
                            (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter)
                        ]['Intervention_Param'].unique()

                for intervention_param in distinct_intervention_params:
                    model_metrics_per_run_per_percentage = \
                        self.models_metrics_dct[model_name][
                                (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter) &
                                (self.models_metrics_dct[model_name]['Intervention_Param'] == intervention_param)
                            ]
                    structured_exp_results_dct[model_name][exp_iter][intervention_param] =\
                        model_metrics_per_run_per_percentage

        return structured_exp_results_dct

    def create_exp_subgroup_metrics_dct_for_mult_test_sets(self):
        structured_exp_results_dct = dict()
        for model_name in self.models_metrics_dct.keys():
            structured_exp_results_dct[model_name] = dict()
            distinct_exp_iters_lst = self.models_metrics_dct[model_name]['Experiment_Iteration'].unique()

            for exp_iter in distinct_exp_iters_lst:
                structured_exp_results_dct[model_name][exp_iter] = dict()
                distinct_intervention_params = \
                    self.models_metrics_dct[model_name][
                        (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter)
                    ]['Intervention_Param'].unique()

                for intervention_param in distinct_intervention_params:
                    structured_exp_results_dct[model_name][exp_iter][intervention_param] = dict()
                    distinct_test_set_indexes = \
                        self.models_metrics_dct[model_name][
                            (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter) &
                            (self.models_metrics_dct[model_name]['Intervention_Param'] == intervention_param)
                        ]['Test_Set_Index'].unique()

                    for test_set_index in distinct_test_set_indexes:
                        model_metrics_per_run_per_percentage = \
                            self.models_metrics_dct[model_name][
                                (self.models_metrics_dct[model_name]['Experiment_Iteration'] == exp_iter) &
                                (self.models_metrics_dct[model_name]['Intervention_Param'] == intervention_param) &
                                (self.models_metrics_dct[model_name]['Test_Set_Index'] == test_set_index)
                            ]
                        structured_exp_results_dct[model_name][exp_iter][intervention_param][test_set_index] = \
                            model_metrics_per_run_per_percentage

        return structured_exp_results_dct

    def compose_group_metrics(self, structured_exp_results_dct):
        """
        Create a group metrics dictionary based on the input structured_exp_results_dct.
        The hierarchy of the created dictionary is the following:
            model_name: {
                preprocessing_technique: {
                    exp_iter: {
                        dct_pct_key: pandas df with group metrics for this model, preprocessing technique, and experiment iteration
                    }
                }
            }

        Parameters
        ----------
        structured_exp_results_dct
            A structured dictionary with metrics per each model, preprocessing technique, and experiment iteration.
            Similar to that one returned by ExperimentsComposer.create_exp_subgroup_metrics_dct()

        Returns
        -------
        Return a new structured dictionary with group metrics per each model, preprocessing technique,
         and experiment iteration.

        """
        structured_exp_avg_group_metrics_dct = dict()
        for model_name in structured_exp_results_dct.keys():
            for exp_iter in structured_exp_results_dct[model_name].keys():
                for percentage in structured_exp_results_dct[model_name][exp_iter].keys():
                    model_subgroup_metrics_df = structured_exp_results_dct[model_name][exp_iter][percentage]
                    model_subgroup_metrics_df = model_subgroup_metrics_df.drop(['Bootstrap_Model_Seed', 'Record_Create_Date_Time'], axis=1)
                    metrics_composer = MetricsComposer(
                        {model_name: model_subgroup_metrics_df},
                        self.sensitive_attributes_dct
                    )
                    model_group_metrics_df = metrics_composer.compose_metrics()
                    model_group_metrics_df['Experiment_Iteration'] = exp_iter
                    model_group_metrics_df['Intervention_Param'] = percentage
                    structured_exp_avg_group_metrics_dct.setdefault(model_name, {})\
                        .setdefault(exp_iter, {})[percentage] = model_group_metrics_df

        return structured_exp_avg_group_metrics_dct

    def compose_group_metrics_for_mult_test_sets(self, structured_exp_results_dct):
        structured_exp_avg_group_metrics_dct = dict()
        for model_name in structured_exp_results_dct.keys():
            for exp_iter in structured_exp_results_dct[model_name].keys():
                for intervention_param in structured_exp_results_dct[model_name][exp_iter].keys():
                    for test_set_index in structured_exp_results_dct[model_name][exp_iter][intervention_param].keys():
                        model_subgroup_metrics_df = structured_exp_results_dct[model_name][exp_iter][intervention_param][test_set_index]
                        model_subgroup_metrics_df = model_subgroup_metrics_df.drop(['Bootstrap_Model_Seed', 'Record_Create_Date_Time'], axis=1)
                        metrics_composer = MetricsComposer(
                            {model_name: model_subgroup_metrics_df},
                            self.sensitive_attributes_dct
                        )
                        model_group_metrics_df = metrics_composer.compose_metrics()
                        model_group_metrics_df['Experiment_Iteration'] = exp_iter
                        model_group_metrics_df['Intervention_Param'] = intervention_param
                        model_group_metrics_df['Test_Set_Index'] = test_set_index
                        structured_exp_avg_group_metrics_dct.setdefault(model_name, {}) \
                            .setdefault(exp_iter, {}) \
                            .setdefault(intervention_param, {})[test_set_index] = model_group_metrics_df

        return structured_exp_avg_group_metrics_dct
