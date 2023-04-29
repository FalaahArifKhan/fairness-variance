import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from source.configs.constants import SubgroupMetricsType, GroupMetricsType


class ExperimentsVisualizer:
    def __init__(self, exp_subgroup_metrics_dct: dict, exp_avg_runs_group_metrics_dct: dict,
                 dataset_name: str, model_names: list, sensitive_attrs: list):
        sns.set_theme(style="whitegrid")

        self.exp_subgroup_metrics_dct = exp_subgroup_metrics_dct
        self.exp_avg_runs_group_metrics_dct = exp_avg_runs_group_metrics_dct
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.sensitive_attrs = sensitive_attrs

        # Technical attributes
        self.all_error_subgroup_metrics = [
            'TPR',
            'TNR',
            'FNR',
            'FPR',
            'PPV',
            'Accuracy',
            'F1',
            'Positive-Rate',
            'Selection-Rate',
        ]
        self.all_variance_subgroup_metrics = [
            # 'Mean',
            'Std',
            'IQR',
            # 'Entropy',
            'Jitter',
            # 'Per_Sample_Accuracy',
            'Label_Stability',
        ]
        self.all_group_fairness_metrics_lst = [
            'Accuracy_Parity',
            'Equalized_Odds_TPR',
            'Equalized_Odds_FPR',
            'Disparate_Impact',
            'Statistical_Parity_Difference',
        ]
        self.all_group_variance_metrics_lst = [
            'IQR_Parity',
            'Label_Stability_Ratio',
            'Std_Parity',
            'Std_Ratio',
            'Jitter_Parity',
        ]

        # Create exp_avg_runs_subgroup_metrics_dct
        exp_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.exp_subgroup_metrics_dct.keys():
            for preprocessing_technique in self.exp_subgroup_metrics_dct[model_name].keys():
                for exp_iter in self.exp_subgroup_metrics_dct[model_name][preprocessing_technique].keys():
                    for percentage in self.exp_subgroup_metrics_dct[model_name][preprocessing_technique][exp_iter].keys():
                        multiple_runs_subgroup_metrics_df = self.exp_subgroup_metrics_dct[model_name][preprocessing_technique][exp_iter][percentage]
                        columns_to_group = [col for col in multiple_runs_subgroup_metrics_df.columns
                                            if col not in ('Bootstrap_Model_Seed', 'Run_Number', 'Record_Create_Date_Time')]
                        exp_avg_runs_subgroup_metrics_dct.setdefault(model_name, {})\
                            .setdefault(preprocessing_technique, {})\
                            .setdefault(exp_iter, {})[percentage] = multiple_runs_subgroup_metrics_df[columns_to_group].groupby(
                                                                        ['Metric', 'Model_Name']
                                                                    ).mean().reset_index()

        self.exp_avg_runs_subgroup_metrics_dct = exp_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_runs_subgroup_metrics_dct
        melted_exp_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.exp_avg_runs_subgroup_metrics_dct.keys():
            for preprocessing_technique in self.exp_avg_runs_subgroup_metrics_dct[model_name].keys():
                for exp_iter in self.exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique].keys():
                    for percentage in self.exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique][exp_iter].keys():
                        model_subgroup_metrics_df = self.exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique][exp_iter][percentage]
                        subgroup_names = [col for col in model_subgroup_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
                        melted_model_subgroup_metrics_df = model_subgroup_metrics_df.melt(
                            id_vars=[col for col in model_subgroup_metrics_df.columns if col not in subgroup_names],
                            value_vars=subgroup_names,
                            var_name="Subgroup",
                            value_name="Metric_Value"
                        )
                        melted_exp_avg_runs_subgroup_metrics_dct.setdefault(model_name, {})\
                            .setdefault(preprocessing_technique, {})\
                            .setdefault(exp_iter, {})[percentage] = melted_model_subgroup_metrics_df

        self.melted_exp_avg_runs_subgroup_metrics_dct = melted_exp_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_runs_group_metrics_dct
        melted_exp_avg_runs_group_metrics_dct = dict()
        for model_name in self.exp_avg_runs_group_metrics_dct.keys():
            for preprocessing_technique in self.exp_avg_runs_group_metrics_dct[model_name].keys():
                for exp_iter in self.exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique].keys():
                    for percentage in self.exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique][exp_iter].keys():
                        model_group_metrics_df = self.exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique][exp_iter][percentage]
                        group_names = [col for col in model_group_metrics_df.columns if col not in ('Metric', 'Model_Name')]
                        melted_model_group_metrics_df = model_group_metrics_df.melt(
                            id_vars=[col for col in model_group_metrics_df.columns if col not in group_names],
                            value_vars=group_names,
                            var_name="Group",
                            value_name="Metric_Value"
                        )
                        melted_exp_avg_runs_group_metrics_dct.setdefault(model_name, {})\
                            .setdefault(preprocessing_technique, {})\
                            .setdefault(exp_iter, {})[percentage] = melted_model_group_metrics_df

        self.melted_exp_avg_runs_group_metrics_dct = melted_exp_avg_runs_group_metrics_dct

        # Create melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct
        melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.melted_exp_avg_runs_subgroup_metrics_dct.keys():
            for preprocessing_technique in self.melted_exp_avg_runs_subgroup_metrics_dct[model_name].keys():
                first_exp_iter = list(self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique].keys())[0]

                for percentage in self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique][first_exp_iter].keys():
                    multiple_pct_exp_iters_subgroup_metrics_df = pd.DataFrame()

                    for exp_iter in self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique].keys():
                        multiple_runs_subgroup_metrics_df = self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][preprocessing_technique][exp_iter][percentage]
                        multiple_pct_exp_iters_subgroup_metrics_df = pd.concat([multiple_pct_exp_iters_subgroup_metrics_df, multiple_runs_subgroup_metrics_df])

                    columns_to_group = [col for col in multiple_pct_exp_iters_subgroup_metrics_df.columns
                                        if col not in ('Bootstrap_Model_Seed', 'Run_Number', 'Record_Create_Date_Time',
                                                       'Dataset_Split_Seed', 'Experiment_Iteration', 'Model_Init_Seed',
                                                       'Model_Params')]
                    melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(preprocessing_technique, {})[percentage] = \
                            multiple_pct_exp_iters_subgroup_metrics_df[columns_to_group].groupby(
                                ['Model_Name', 'Test_Set_Index', 'Metric', 'Subgroup']
                            ).mean().reset_index()

        self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct = melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_exp_iters_avg_runs_group_metrics_dct
        melted_exp_avg_exp_iters_avg_runs_group_metrics_dct = dict()
        for model_name in self.melted_exp_avg_runs_group_metrics_dct.keys():
            for preprocessing_technique in self.melted_exp_avg_runs_group_metrics_dct[model_name].keys():
                first_exp_iter = list(self.melted_exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique].keys())[0]

                for percentage in self.melted_exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique][first_exp_iter].keys():
                    multiple_pct_exp_iters_group_metrics_df = pd.DataFrame()

                    for exp_iter in self.melted_exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique].keys():
                        multiple_runs_group_metrics_df = self.melted_exp_avg_runs_group_metrics_dct[model_name][preprocessing_technique][exp_iter][percentage]
                        multiple_pct_exp_iters_group_metrics_df = pd.concat([multiple_pct_exp_iters_group_metrics_df, multiple_runs_group_metrics_df])

                    columns_to_group = [col for col in multiple_pct_exp_iters_group_metrics_df.columns
                                        if col not in ('Bootstrap_Model_Seed', 'Run_Number', 'Record_Create_Date_Time',
                                                       'Dataset_Split_Seed', 'Experiment_Iteration', 'Model_Init_Seed',
                                                       'Model_Params')]
                    melted_exp_avg_exp_iters_avg_runs_group_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(preprocessing_technique, {})[percentage] = \
                        multiple_pct_exp_iters_group_metrics_df[columns_to_group].groupby(
                            ['Model_Name', 'Metric', 'Group']
                        ).mean().reset_index()

        self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct = melted_exp_avg_exp_iters_avg_runs_group_metrics_dct

    def create_subgroup_metrics_box_plot_for_multiple_exp_iters(self, target_percentage: float,
                                                                target_preprocessing_technique: str,
                                                                subgroup_metrics: list = None,
                                                                subgroup_metrics_type: str = None):
        if subgroup_metrics_type is not None and not SubgroupMetricsType.has_value(subgroup_metrics_type):
            raise ValueError(f'subgroup_metrics_type must be in {tuple(SubgroupMetricsType._value2member_map_.keys())}')

        if subgroup_metrics is None:
            if subgroup_metrics_type is None:
                subgroup_metrics = self.all_error_subgroup_metrics + self.all_variance_subgroup_metrics
            else:
                subgroup_metrics = self.all_error_subgroup_metrics if subgroup_metrics_type == SubgroupMetricsType.ERROR.value \
                    else self.all_variance_subgroup_metrics

        subgroup = 'overall'
        all_models_pct_subgroup_metrics_df = pd.DataFrame()
        for model_name in self.exp_avg_runs_subgroup_metrics_dct.keys():
            for exp_iter in self.exp_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique].keys():
                all_models_pct_subgroup_metrics_df = pd.concat([
                    all_models_pct_subgroup_metrics_df,
                    self.exp_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique][exp_iter][target_percentage]
                ])
        all_models_pct_subgroup_metrics_df = all_models_pct_subgroup_metrics_df.reset_index(drop=True)

        to_plot = all_models_pct_subgroup_metrics_df[all_models_pct_subgroup_metrics_df['Metric'].isin(subgroup_metrics)]
        plt.figure(figsize=(15, 10))
        ax = sns.boxplot(x=to_plot['Metric'],
                         y=to_plot[subgroup],
                         hue=to_plot['Model_Name'])

        plt.legend(loc='upper left',
                   ncol=2,
                   fancybox=True,
                   shadow=True)
        plt.xlabel("Metric name")
        plt.ylabel("Metric value")
        fig = ax.get_figure()
        fig.tight_layout()

    def create_subgroup_metrics_box_plot_for_multiple_percentages(self, target_preprocessing_technique: str,
                                                                  subgroup_metrics: list = None,
                                                                  subgroup_metrics_type: str = None):
        if subgroup_metrics_type is not None and not SubgroupMetricsType.has_value(subgroup_metrics_type):
            raise ValueError(f'subgroup_metrics_type must be in {tuple(SubgroupMetricsType._value2member_map_.keys())}')

        if subgroup_metrics is None:
            if subgroup_metrics_type is None:
                subgroup_metrics = self.all_error_subgroup_metrics + self.all_variance_subgroup_metrics
            else:
                subgroup_metrics = self.all_error_subgroup_metrics if subgroup_metrics_type == SubgroupMetricsType.ERROR.value \
                    else self.all_variance_subgroup_metrics

        target_subgroup = 'overall'
        all_models_pct_subgroup_metrics_df = pd.DataFrame()
        # Take all percentages for the specific exp iter
        for model_name in self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct.keys():
            for percentage in self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique].keys():
                pct_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique][percentage]
                pct_metrics_df = pct_metrics_df[pct_metrics_df.Subgroup == target_subgroup]
                all_models_pct_subgroup_metrics_df = pd.concat([
                    all_models_pct_subgroup_metrics_df,
                    pct_metrics_df
                ])
        all_models_pct_subgroup_metrics_df = all_models_pct_subgroup_metrics_df.reset_index(drop=True)

        to_plot = all_models_pct_subgroup_metrics_df[all_models_pct_subgroup_metrics_df['Metric'].isin(subgroup_metrics)]
        plt.figure(figsize=(15, 10))
        ax = sns.boxplot(x=to_plot['Metric'],
                         y=to_plot['Metric_Value'],
                         hue=to_plot['Model_Name'])

        plt.legend(loc='upper left',
                   ncol=2,
                   fancybox=True,
                   shadow=True)
        plt.xlabel("Metric name")
        plt.ylabel("Metric value")
        fig = ax.get_figure()
        fig.tight_layout()

    def create_group_metrics_box_plot_for_multiple_percentages(self, target_preprocessing_technique: str,
                                                               target_group: str, group_metrics: list = None,
                                                               group_metrics_type: str = None):
        if group_metrics_type is not None and not GroupMetricsType.has_value(group_metrics_type):
            raise ValueError(f'group_metrics_type must be in {tuple(GroupMetricsType._value2member_map_.keys())}')

        if group_metrics is None:
            if group_metrics_type is None:
                group_metrics = self.all_group_fairness_metrics_lst + self.all_group_variance_metrics_lst
            else:
                group_metrics = self.all_group_fairness_metrics_lst if group_metrics_type == GroupMetricsType.FAIRNESS.value \
                    else self.all_group_variance_metrics_lst

        all_models_pct_group_metrics_df = pd.DataFrame()
        # Take all percentages for the specific exp iter
        for model_name in self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct.keys():
            for percentage in self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct[model_name][target_preprocessing_technique].keys():
                pct_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct[model_name][target_preprocessing_technique][percentage]
                pct_metrics_df = pct_metrics_df[pct_metrics_df.Group == target_group]
                all_models_pct_group_metrics_df = pd.concat([
                    all_models_pct_group_metrics_df,
                    pct_metrics_df
                ])
        all_models_pct_group_metrics_df = all_models_pct_group_metrics_df.reset_index(drop=True)

        to_plot = all_models_pct_group_metrics_df[all_models_pct_group_metrics_df['Metric'].isin(group_metrics)]
        plt.figure(figsize=(15, 10))
        ax = sns.boxplot(x=to_plot['Metric'],
                         y=to_plot['Metric_Value'],
                         hue=to_plot['Model_Name'])

        plt.legend(loc='upper left',
                   ncol=2,
                   fancybox=True,
                   shadow=True)
        plt.xlabel("Metric name")
        plt.ylabel("Metric value")
        fig = ax.get_figure()
        fig.tight_layout()

    def create_subgroups_grid_pct_lines_plot(self, model_name: str, target_preprocessing_technique: str = None,
                                             subgroup_metrics: list = None, subgroups: list = None,
                                             subgroup_metrics_type = None):
        if subgroup_metrics_type is not None and not SubgroupMetricsType.has_value(subgroup_metrics_type):
            raise ValueError(f'subgroup_metrics_type must be in {tuple(SubgroupMetricsType._value2member_map_.keys())}')

        if subgroups is None:
            subgroups = [attr + '_priv' for attr in self.sensitive_attrs] + \
                        [attr + '_dis' for attr in self.sensitive_attrs] + ['overall']

        if target_preprocessing_technique is None:
            target_preprocessing_technique = list(self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name].keys())[0]

        if subgroup_metrics is None:
            if subgroup_metrics_type is None:
                subgroup_metrics = self.all_error_subgroup_metrics + self.all_variance_subgroup_metrics
            else:
                subgroup_metrics = self.all_error_subgroup_metrics if subgroup_metrics_type == SubgroupMetricsType.ERROR.value \
                    else self.all_variance_subgroup_metrics

        # Create a grid framing
        row_len = 3
        subgroup_metrics_len = len(subgroup_metrics)
        div_val, mod_val = divmod(subgroup_metrics_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        all_percentage_subgroup_metrics_df = pd.DataFrame()
        for pct in self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique].keys():
            percentage_subgroup_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique][pct]
            percentage_subgroup_metrics_df['Percentage'] = pct
            all_percentage_subgroup_metrics_df = pd.concat(
                [all_percentage_subgroup_metrics_df, percentage_subgroup_metrics_df]
            )

        all_percentage_subgroup_metrics_df = all_percentage_subgroup_metrics_df.reset_index(drop=True)

        grid_chart = alt.vconcat()
        metric_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                metric_idx += 1
                subplot_metrics_df = all_percentage_subgroup_metrics_df[
                    (all_percentage_subgroup_metrics_df.Metric == subgroup_metrics[metric_idx]) &
                    (all_percentage_subgroup_metrics_df.Subgroup.isin(subgroups))
                ]
                base = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Percentage', type='quantitative', title='Percentage of Affected Rows'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=subgroup_metrics[metric_idx]),
                    color='Subgroup:N',
                    strokeWidth=alt.condition(
                        "datum.Subgroup == 'overall'",
                        alt.value(4),
                        alt.value(2)
                    ),
                ).properties(
                    width=250, height=250
                )

                row |= base

            grid_chart &= row

        grid_chart = (
            grid_chart.configure_axis(
                labelFontSize=15,
                titleFontSize=15
            ).configure_legend(
                titleFontSize=15,
                labelFontSize=13,
                symbolStrokeWidth=10,
            )
        )
        return grid_chart

    def create_groups_grid_pct_lines_plot(self, model_name: str, target_preprocessing_technique: str = None,
                                          group_metrics: list = None, groups: list = None, group_metrics_type = None):
        if group_metrics_type is not None and not GroupMetricsType.has_value(group_metrics_type):
            raise ValueError(f'group_metrics_type must be in {tuple(GroupMetricsType._value2member_map_.keys())}')

        if groups is None:
            groups = [attr for attr in self.sensitive_attrs]

        if group_metrics is None:
            if group_metrics_type is None:
                group_metrics = self.all_group_fairness_metrics_lst + self.all_group_variance_metrics_lst
            else:
                group_metrics = self.all_group_fairness_metrics_lst if group_metrics_type == GroupMetricsType.FAIRNESS.value \
                    else self.all_group_variance_metrics_lst

        # Create a grid framing
        row_len = 3
        group_metrics_len = len(group_metrics)
        div_val, mod_val = divmod(group_metrics_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        all_percentage_group_metrics_df = pd.DataFrame()
        for pct in self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct[model_name][target_preprocessing_technique].keys():
            percentage_group_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct[model_name][target_preprocessing_technique][pct]
            percentage_group_metrics_df['Percentage'] = pct
            all_percentage_group_metrics_df = pd.concat(
                [all_percentage_group_metrics_df, percentage_group_metrics_df]
            )

        all_percentage_group_metrics_df = all_percentage_group_metrics_df.reset_index(drop=True)

        grid_chart = alt.vconcat()
        metric_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                metric_idx += 1
                subplot_metrics_df = all_percentage_group_metrics_df[
                    (all_percentage_group_metrics_df.Metric == group_metrics[metric_idx]) &
                    (all_percentage_group_metrics_df.Group.isin(groups))
                    ]
                base = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Percentage', type='quantitative', title='Percentage of Affected Rows'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=group_metrics[metric_idx]),
                    color='Group:N',
                ).properties(
                    width=250, height=250
                )

                row |= base

            grid_chart &= row

        grid_chart = (
            grid_chart.configure_axis(
                labelFontSize=15,
                titleFontSize=15
            ).configure_legend(
                titleFontSize=15,
                labelFontSize=13,
                symbolStrokeWidth=10,
            )
        )
        return grid_chart

    def create_subgroups_grid_pct_lines_per_model_plot(self, subgroup_metric: str, target_preprocessing_technique: str,
                                                       model_names: list = None, subgroups: list = None):
        if subgroups is None:
            subgroups = [attr + '_priv' for attr in self.sensitive_attrs] + \
                        [attr + '_dis' for attr in self.sensitive_attrs] + ['overall']

        if model_names is None:
            model_names = self.model_names

        # Create a grid framing
        row_len = 3
        subgroup_models_len = len(model_names)
        div_val, mod_val = divmod(subgroup_models_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        all_models_percentage_subgroup_metrics_df = pd.DataFrame()
        for model_name in model_names:
            for pct in self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique].keys():
                percentage_subgroup_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique][pct]
                percentage_subgroup_metrics_df['Percentage'] = pct
                all_models_percentage_subgroup_metrics_df = pd.concat(
                    [all_models_percentage_subgroup_metrics_df, percentage_subgroup_metrics_df]
                )

        all_models_percentage_subgroup_metrics_df = all_models_percentage_subgroup_metrics_df.reset_index(drop=True)

        grid_chart = alt.vconcat()
        model_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                model_idx += 1
                subplot_metrics_df = all_models_percentage_subgroup_metrics_df[
                    (all_models_percentage_subgroup_metrics_df.Metric == subgroup_metric) &
                    (all_models_percentage_subgroup_metrics_df.Model_Name == model_names[model_idx]) &
                    (all_models_percentage_subgroup_metrics_df.Subgroup.isin(subgroups))
                    ]
                base = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Percentage', type='quantitative', title='Percentage of Affected Rows'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=subgroup_metric),
                    color='Subgroup:N',
                    strokeWidth=alt.condition(
                        "datum.Subgroup == 'overall'",
                        alt.value(4),
                        alt.value(2)
                    ),
                ).properties(
                    width=250, height=250, title = model_names[model_idx]
                )

                row |= base

            grid_chart &= row

        grid_chart = (
            grid_chart.configure_axis(
                labelFontSize=15,
                titleFontSize=15
            ).configure_legend(
                titleFontSize=15,
                labelFontSize=13,
                symbolStrokeWidth=10,
            )
        )
        return grid_chart

    def create_groups_grid_pct_lines_per_model_plot(self, group_metric: str, target_preprocessing_technique: str = None,
                                                    model_names: list = None, groups: list = None):
        if groups is None:
            groups = [attr for attr in self.sensitive_attrs]

        if model_names is None:
            model_names = self.model_names

        # Create a grid framing
        row_len = 3
        group_models_len = len(model_names)
        div_val, mod_val = divmod(group_models_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        all_models_percentage_group_metrics_df = pd.DataFrame()
        for model_name in model_names:
            for pct in self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct[model_name][target_preprocessing_technique].keys():
                percentage_group_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct[model_name][target_preprocessing_technique][pct]
                percentage_group_metrics_df['Percentage'] = pct
                all_models_percentage_group_metrics_df = pd.concat(
                    [all_models_percentage_group_metrics_df, percentage_group_metrics_df]
                )
        all_models_percentage_group_metrics_df = all_models_percentage_group_metrics_df.reset_index(drop=True)

        grid_chart = alt.vconcat()
        model_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                model_idx += 1
                subplot_metrics_df = all_models_percentage_group_metrics_df[
                    (all_models_percentage_group_metrics_df.Metric == group_metric) &
                    (all_models_percentage_group_metrics_df.Model_Name == model_names[model_idx]) &
                    (all_models_percentage_group_metrics_df.Group.isin(groups))
                    ]
                base = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Percentage', type='quantitative', title='Percentage of Affected Rows'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=group_metric),
                    color='Group:N',
                ).properties(
                    width=250, height=250, title=model_names[model_idx]
                )

                row |= base

            grid_chart &= row

        grid_chart = (
            grid_chart.configure_axis(
                labelFontSize=15,
                titleFontSize=15
            ).configure_legend(
                titleFontSize=15,
                labelFontSize=13,
                symbolStrokeWidth=10,
            )
        )
        return grid_chart

    def create_subgroups_grid_pct_lines_per_model_and_preprocessing_plot(self, subgroup_metric: str, model_name: str,
                                                                         subgroups: list = None):
        if subgroups is None:
            subgroups = [attr + '_priv' for attr in self.sensitive_attrs] + \
                        [attr + '_dis' for attr in self.sensitive_attrs] + ['overall']

        preprocessing_techniques = list(self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name].keys())

        # Create a grid framing
        row_len = 3
        subgroup_grid_len = len(preprocessing_techniques)
        div_val, mod_val = divmod(subgroup_grid_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        all_models_percentage_subgroup_metrics_df = pd.DataFrame()
        for technique in preprocessing_techniques:
            for pct in self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][technique].keys():
                percentage_subgroup_metrics_df = self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct[model_name][technique][pct]
                percentage_subgroup_metrics_df['Percentage'] = pct
                percentage_subgroup_metrics_df['Preprocessing_Technique'] = technique
                all_models_percentage_subgroup_metrics_df = pd.concat(
                    [all_models_percentage_subgroup_metrics_df, percentage_subgroup_metrics_df]
                )

        all_models_percentage_subgroup_metrics_df = all_models_percentage_subgroup_metrics_df.reset_index(drop=True)

        grid_chart = alt.vconcat()
        technique_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                technique_idx += 1
                subplot_metrics_df = all_models_percentage_subgroup_metrics_df[
                    (all_models_percentage_subgroup_metrics_df.Metric == subgroup_metric) &
                    (all_models_percentage_subgroup_metrics_df.Preprocessing_Technique == preprocessing_techniques[technique_idx]) &
                    (all_models_percentage_subgroup_metrics_df.Subgroup.isin(subgroups))
                ]
                base = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Percentage', type='quantitative', title='Percentage of Affected Rows'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=subgroup_metric),
                    color='Subgroup:N',
                    strokeWidth=alt.condition(
                        "datum.Subgroup == 'overall'",
                        alt.value(4),
                        alt.value(2)
                    ),
                ).properties(
                    width=250, height=250, title = preprocessing_techniques[technique_idx]
                )

                row |= base

            grid_chart &= row

        grid_chart = (
            grid_chart.configure_axis(
                labelFontSize=15,
                titleFontSize=15
            ).configure_legend(
                titleFontSize=15,
                labelFontSize=13,
                symbolStrokeWidth=10,
            )
        )
        return grid_chart
