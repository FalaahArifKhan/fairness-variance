import os
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone

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
            'Mean',
            'Std',
            'IQR',
            'Entropy',
            'Jitter',
            'Per_Sample_Accuracy',
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
            for exp_iter in self.exp_subgroup_metrics_dct[model_name].keys():
                for percentage in self.exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                    multiple_runs_subgroup_metrics_df = self.exp_subgroup_metrics_dct[model_name][exp_iter][percentage]
                    columns_to_group = [col for col in multiple_runs_subgroup_metrics_df.columns
                                        if col not in ('Model_Seed', 'Run_Number')]
                    exp_avg_runs_subgroup_metrics_dct.setdefault(model_name, {}).setdefault(exp_iter, {})[percentage] = \
                        multiple_runs_subgroup_metrics_df[columns_to_group].groupby(['Metric', 'Model_Name']).mean().reset_index()

        self.exp_avg_runs_subgroup_metrics_dct = exp_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_runs_subgroup_metrics_dct
        melted_exp_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.exp_avg_runs_subgroup_metrics_dct.keys():
            for exp_iter in self.exp_avg_runs_subgroup_metrics_dct[model_name].keys():
                for percentage in self.exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter].keys():
                    model_subgroup_metrics_df = self.exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter][percentage]
                    subgroup_names = [col for col in model_subgroup_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
                    melted_model_subgroup_metrics_df = model_subgroup_metrics_df.melt(
                        id_vars=[col for col in model_subgroup_metrics_df.columns if col not in subgroup_names],
                        value_vars=subgroup_names,
                        var_name="Subgroup",
                        value_name="Metric_Value"
                    )
                    melted_exp_avg_runs_subgroup_metrics_dct.setdefault(model_name, {}).setdefault(exp_iter, {})[percentage] = \
                        melted_model_subgroup_metrics_df

        self.melted_exp_avg_runs_subgroup_metrics_dct = melted_exp_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_runs_group_metrics_dct
        melted_exp_avg_runs_group_metrics_dct = dict()
        for model_name in self.exp_avg_runs_group_metrics_dct.keys():
            for exp_iter in self.exp_avg_runs_group_metrics_dct[model_name].keys():
                for percentage in self.exp_avg_runs_group_metrics_dct[model_name][exp_iter].keys():
                    model_group_metrics_df = self.exp_avg_runs_group_metrics_dct[model_name][exp_iter][percentage]
                    group_names = [col for col in model_group_metrics_df.columns if col not in ('Metric', 'Model_Name')]
                    melted_model_group_metrics_df = model_group_metrics_df.melt(
                        id_vars=[col for col in model_group_metrics_df.columns if col not in group_names],
                        value_vars=group_names,
                        var_name="Group",
                        value_name="Metric_Value"
                    )
                    melted_exp_avg_runs_group_metrics_dct.setdefault(model_name, {}).setdefault(exp_iter, {})[percentage] = \
                        melted_model_group_metrics_df

        self.melted_exp_avg_runs_group_metrics_dct = melted_exp_avg_runs_group_metrics_dct

    def create_subgroups_grid_pct_lines_plot(self, model_name: str, exp_iter: str,
                                             subgroup_metrics: list = None, subgroups: list = None,
                                             subgroup_metrics_type = None):
        if subgroup_metrics_type is not None and not SubgroupMetricsType.has_value(subgroup_metrics_type):
            raise ValueError(f'subgroup_metrics_type must be in {tuple(SubgroupMetricsType._value2member_map_.keys())}')

        if subgroups is None:
            subgroups = [attr + '_priv' for attr in self.sensitive_attrs] + \
                        [attr + '_dis' for attr in self.sensitive_attrs] + ['overall']

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
        for pct in self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter].keys():
            percentage_subgroup_metrics_df = self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter][pct]
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
                    x='Percentage:Q',
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

    def create_groups_grid_pct_lines_plot(self, model_name: str, exp_iter: str,
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
        for pct in self.melted_exp_avg_runs_group_metrics_dct[model_name][exp_iter].keys():
            percentage_group_metrics_df = self.melted_exp_avg_runs_group_metrics_dct[model_name][exp_iter][pct]
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
                    x='Percentage:Q',
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
