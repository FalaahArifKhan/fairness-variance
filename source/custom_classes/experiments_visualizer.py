import altair as alt
import pandas as pd
import seaborn as sns


class ExperimentsVisualizer:
    def __init__(self, exp_subgroup_metrics_dct: dict, exp_group_metrics_dct: dict,
                 dataset_name: str, model_names: list, sensitive_attrs: list):
        sns.set_theme(style="whitegrid")

        self.exp_subgroup_metrics_dct = exp_subgroup_metrics_dct
        self.exp_group_metrics_dct = exp_group_metrics_dct
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
        self.all_stability_subgroup_metrics = [
            'Std',
            'IQR',
            'Jitter',
            'Label_Stability',
            'Aleatoric_Uncertainty',
            'Overall_Uncertainty',
        ]
        self.all_group_fairness_metrics_lst = [
            'Accuracy_Parity',
            'Equalized_Odds_TPR',
            'Equalized_Odds_FPR',
            'Equalized_Odds_FNR',
            'Disparate_Impact',
            'Statistical_Parity_Difference',
        ]
        self.all_group_stability_metrics_lst = [
            'IQR_Parity',
            'Label_Stability_Ratio',
            'Std_Parity',
            'Std_Ratio',
            'Jitter_Parity',
        ]

        # Create melted_exp_subgroup_metrics_dct
        melted_exp_subgroup_metrics_dct = dict()
        for model_name in self.exp_subgroup_metrics_dct.keys():
            for exp_iter in self.exp_subgroup_metrics_dct[model_name].keys():
                for percentage in self.exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                    model_subgroup_metrics_df = self.exp_subgroup_metrics_dct[model_name][exp_iter][percentage]
                    subgroup_names = [col for col in model_subgroup_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
                    melted_model_subgroup_metrics_df = model_subgroup_metrics_df.melt(
                        id_vars=[col for col in model_subgroup_metrics_df.columns if col not in subgroup_names],
                        value_vars=subgroup_names,
                        var_name="Subgroup",
                        value_name="Metric_Value"
                    )
                    melted_exp_subgroup_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {})[percentage] = melted_model_subgroup_metrics_df

        self.melted_exp_subgroup_metrics_dct = melted_exp_subgroup_metrics_dct

        # Create a dict where a key is a model name and a value is a dataframe with
        # all subgroup metrics, percentages, and exp iterations per each model
        melted_all_subgroup_metrics_per_model_dct = dict()
        for model_name in self.melted_exp_subgroup_metrics_dct.keys():
            all_subgroup_metrics_per_model_df = pd.DataFrame()
            for exp_iter in self.melted_exp_subgroup_metrics_dct[model_name].keys():
                for percentage in self.melted_exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                    all_subgroup_metrics_per_model_df = pd.concat([
                        all_subgroup_metrics_per_model_df,
                        melted_exp_subgroup_metrics_dct[model_name][exp_iter][percentage],
                    ])
            melted_all_subgroup_metrics_per_model_dct[model_name] = all_subgroup_metrics_per_model_df

        self.melted_all_subgroup_metrics_per_model_dct = melted_all_subgroup_metrics_per_model_dct

        # Create melted_exp_group_metrics_dct
        melted_exp_group_metrics_dct = dict()
        for model_name in self.exp_group_metrics_dct.keys():
            for exp_iter in self.exp_group_metrics_dct[model_name].keys():
                for percentage in self.exp_group_metrics_dct[model_name][exp_iter].keys():
                    model_group_metrics_df = self.exp_group_metrics_dct[model_name][exp_iter][percentage]
                    # All other columns in model_group_metrics_df
                    # except 'Metric', 'Model_Name', 'Intervention_Param', 'Experiment_Iteration' are group names
                    group_names = [col for col in model_group_metrics_df.columns
                                   if col not in ('Metric', 'Model_Name', 'Intervention_Param', 'Experiment_Iteration')]
                    melted_model_group_metrics_df = model_group_metrics_df.melt(
                        id_vars=[col for col in model_group_metrics_df.columns if col not in group_names],
                        value_vars=group_names,
                        var_name="Group",
                        value_name="Metric_Value"
                    )
                    melted_exp_group_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {})[percentage] = melted_model_group_metrics_df

        self.melted_exp_group_metrics_dct = melted_exp_group_metrics_dct

        # Create a dict where a key is a model name and a value is a dataframe with
        # all group metrics, percentages, and exp iterations per each model
        melted_all_group_metrics_per_model_dct = dict()
        for model_name in self.melted_exp_group_metrics_dct.keys():
            all_group_metrics_per_model_df = pd.DataFrame()
            for exp_iter in self.melted_exp_group_metrics_dct[model_name].keys():
                for percentage in self.melted_exp_group_metrics_dct[model_name][exp_iter].keys():
                    all_group_metrics_per_model_df = pd.concat([
                        all_group_metrics_per_model_df,
                        melted_exp_group_metrics_dct[model_name][exp_iter][percentage],
                    ])
            melted_all_group_metrics_per_model_dct[model_name] = all_group_metrics_per_model_df

        self.melted_all_group_metrics_per_model_dct = melted_all_group_metrics_per_model_dct

    def create_subgroup_metrics_line_band_plot(self, model_name: str, subgroup_metrics: list, subgroup: str = 'overall'):
        subgroup_metrics_df = self.melted_all_subgroup_metrics_per_model_dct[model_name]

        # Create a grid framing
        row_len = 3
        subgroup_metrics_len = len(subgroup_metrics)
        div_val, mod_val = divmod(subgroup_metrics_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        subgroups_grid_chart = alt.vconcat()
        metric_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                metric_idx += 1
                subplot_metrics_df = subgroup_metrics_df[
                    (subgroup_metrics_df.Metric == subgroup_metrics[metric_idx]) &
                    (subgroup_metrics_df.Subgroup == subgroup)
                ]

                line = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
                    y=alt.Y('mean(Metric_Value)', type='quantitative', title=subgroup_metrics[metric_idx], scale=alt.Scale(zero=False)),
                )

                band = alt.Chart(subplot_metrics_df).mark_errorband(extent='ci').encode(
                    x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=subgroup_metrics[metric_idx], scale=alt.Scale(zero=False))
                )

                base = (band + line).properties(
                    width=280, height=280
                )

                row |= base

            subgroups_grid_chart &= row

        base_font_size = 20
        final_grid_chart = (
            subgroups_grid_chart.configure_axis(
                labelFontSize=base_font_size + 2,
                titleFontSize=base_font_size + 4,
                labelFontWeight='normal',
                titleFontWeight='normal',
            ).configure_title(
                fontSize=base_font_size + 2
            ).configure_legend(
                titleFontSize=base_font_size + 4,
                labelFontSize=base_font_size + 2,
                symbolStrokeWidth=10,
            ).properties(
                title=alt.TitleParams(f'{model_name} Model', fontSize=base_font_size + 5, anchor='middle', dy=-10),
            )
        )

        return final_grid_chart

    def create_line_bands_per_group_metric_plot(self, model_name: str, group_metric: str, group: str, metric_type: str):
        """
        :param metric_type: 'group' or 'subgroup'

        """
        if metric_type == 'subgroup':
            group_metrics_df = self.melted_all_subgroup_metrics_per_model_dct[model_name]
            group_metrics_df['Group'] = group_metrics_df['Subgroup']
        else:
            group_metrics_df = self.melted_all_group_metrics_per_model_dct[model_name]

        groups = [group + suffix for suffix in ['_priv_correct', '_dis_correct', '_priv_incorrect', '_dis_incorrect']]
        subplot_metrics_df = group_metrics_df[
            (group_metrics_df.Metric == group_metric) &
            (group_metrics_df.Group.isin(groups))
        ]

        line_chart = alt.Chart(subplot_metrics_df).mark_line().encode(
            x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
            y=alt.Y('mean(Metric_Value)', type='quantitative', title=group_metric, scale=alt.Scale(zero=False)),
            color='Group:N',
        )

        base_font_size = 20
        final_grid_chart = (
            line_chart.configure_axis(
                labelFontSize=base_font_size + 4,
                titleFontSize=base_font_size + 6,
                labelFontWeight='normal',
                titleFontWeight='normal',
            ).configure_title(
                fontSize=base_font_size + 2
            ).configure_legend(
                titleFontSize=base_font_size + 4,
                labelFontSize=base_font_size + 2,
                symbolStrokeWidth=10,
                labelLimit=300,
                titleLimit=300,
                # columns=1,
                columns=2,
                orient='top',
                # orient='none',
                # legendX=-90, legendY=-100,
                direction='horizontal',
                titleAnchor='middle'
            ).properties(
                width=300,
                height=300
            )
        )

        return final_grid_chart

    def create_group_metrics_line_band_plot(self, group: str, model_name: str, group_metrics: list):
        group_metrics_df = self.melted_all_group_metrics_per_model_dct[model_name]

        # Create a grid framing
        row_len = 3
        group_metrics_len = len(group_metrics)
        div_val, mod_val = divmod(group_metrics_len, row_len)
        grid_framing = [row_len] * div_val + [mod_val] if mod_val != 0 else [row_len] * div_val

        groups_grid_chart = alt.vconcat()
        metric_idx = -1
        for num_subplots in grid_framing:
            row = alt.hconcat()
            for i in range(num_subplots):
                metric_idx += 1
                subplot_metrics_df = group_metrics_df[
                    (group_metrics_df.Metric == group_metrics[metric_idx]) &
                    (group_metrics_df.Group == group)
                    ]

                line = alt.Chart(subplot_metrics_df).mark_line().encode(
                    x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
                    y=alt.Y('mean(Metric_Value)', type='quantitative', title=group_metrics[metric_idx], scale=alt.Scale(zero=False)),
                )

                band = alt.Chart(subplot_metrics_df).mark_errorband(extent='ci').encode(
                    x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
                    y=alt.Y(field='Metric_Value', type='quantitative', title=group_metrics[metric_idx], scale=alt.Scale(zero=False))
                )

                base = (band + line).properties(
                    width=280, height=280
                )

                row |= base

            groups_grid_chart &= row

        base_font_size = 25
        final_grid_chart = (
            groups_grid_chart.configure_axis(
                labelFontSize=base_font_size + 2,
                titleFontSize=base_font_size + 4,
                labelFontWeight='normal',
                titleFontWeight='normal',
            ).configure_title(
                fontSize=base_font_size + 2
            ).configure_legend(
                titleFontSize=base_font_size + 4,
                labelFontSize=base_font_size + 2,
                symbolStrokeWidth=10,
            ).properties(
                title=alt.TitleParams(f'{model_name} Model', fontSize=base_font_size + 5, anchor='middle', dy=-10),
            )
        )

        return final_grid_chart
