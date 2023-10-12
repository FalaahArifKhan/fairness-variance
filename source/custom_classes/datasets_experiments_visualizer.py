import altair as alt
import pandas as pd
import seaborn as sns

from altair.utils.schemapi import Undefined

from source.utils.data_vis_utils import create_melted_subgroup_and_group_dicts


class DatasetsExperimentsVisualizer:
    def __init__(self, datasets_exp_metrics_dct: dict, dataset_names: list, model_names: list,
                 datasets_sensitive_attrs: dict):
        sns.set_theme(style="whitegrid")

        self.dataset_names = dataset_names
        self.model_names = model_names
        self.datasets_sensitive_attrs = datasets_sensitive_attrs

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

        # Create a dict with melted subgroup and group dataframes per dataset per model
        melted_all_metrics_per_dataset_per_model_dct = dict()
        for dataset_name in datasets_exp_metrics_dct.keys():
            exp_subgroup_metrics_dct = datasets_exp_metrics_dct[dataset_name]['subgroup_metrics']
            exp_group_metrics_dct = datasets_exp_metrics_dct[dataset_name]['group_metrics']
            melted_all_subgroup_metrics_per_model_dct, melted_all_group_metrics_per_model_dct = (
                create_melted_subgroup_and_group_dicts(exp_subgroup_metrics_dct, exp_group_metrics_dct))
            melted_all_metrics_per_dataset_per_model_dct[dataset_name] = {
                'subgroup_metrics': melted_all_subgroup_metrics_per_model_dct,
                'group_metrics': melted_all_group_metrics_per_model_dct,
            }

        # Create a dict with melted subgroup and group dataframes for all datasets per model
        melted_all_datasets_subgroup_metrics_per_model_dct = dict()
        melted_all_datasets_group_metrics_per_model_dct = dict()
        for model_name in model_names:
            all_datasets_model_subgroup_metrics = pd.DataFrame()
            all_datasets_model_group_metrics = pd.DataFrame()
            for dataset_name in melted_all_metrics_per_dataset_per_model_dct.keys():
                exp_subgroup_metrics_df = melted_all_metrics_per_dataset_per_model_dct[dataset_name]['subgroup_metrics'][model_name]
                exp_subgroup_metrics_df['Dataset_Name'] = dataset_name
                exp_group_metrics_df = melted_all_metrics_per_dataset_per_model_dct[dataset_name]['group_metrics'][model_name]
                exp_group_metrics_df['Dataset_Name'] = dataset_name

                all_datasets_model_subgroup_metrics = (
                    pd.concat([all_datasets_model_subgroup_metrics, exp_subgroup_metrics_df])
                )
                all_datasets_model_group_metrics = (
                    pd.concat([all_datasets_model_group_metrics, exp_group_metrics_df])
                )

            melted_all_datasets_subgroup_metrics_per_model_dct[model_name] = all_datasets_model_subgroup_metrics
            melted_all_datasets_group_metrics_per_model_dct[model_name] = all_datasets_model_group_metrics

        self.melted_all_datasets_subgroup_metrics_per_model_dct = melted_all_datasets_subgroup_metrics_per_model_dct
        self.melted_all_datasets_group_metrics_per_model_dct = melted_all_datasets_group_metrics_per_model_dct

    def create_overall_metric_line_bands_per_dataset_plot(self, model_name: str, metric_name: str,
                                                          dataset_names: list = None, ylim=Undefined, with_band=True):
        group = 'overall'
        if dataset_names is None:
            dataset_names = self.dataset_names

        group_metrics_df = self.melted_all_datasets_subgroup_metrics_per_model_dct[model_name]
        subplot_metrics_df = group_metrics_df[
            (group_metrics_df.Metric == metric_name) &
            (group_metrics_df.Subgroup == group) &
            (group_metrics_df.Dataset_Name.isin(dataset_names))
            ]

        line_chart = alt.Chart(subplot_metrics_df).mark_line().encode(
            x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
            y=alt.Y('mean(Metric_Value)', type='quantitative', title='', scale=alt.Scale(zero=False, domain=ylim)),
            color='Dataset_Name:N',
        )
        if with_band:
            band_chart = alt.Chart(subplot_metrics_df).mark_errorband(extent='ci').encode(
                x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
                y=alt.Y(field='Metric_Value', type='quantitative', title='', scale=alt.Scale(zero=False, domain=ylim)),
                color='Dataset_Name:N',
            )
            base_chart = (band_chart + line_chart)
        else:
            base_chart = line_chart

        base_font_size = 20
        final_grid_chart = (
            base_chart.configure_axis(
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
                labelLimit=400,
                titleLimit=300,
                columns=2,
                orient='top',
                direction='horizontal',
                titleAnchor='middle'
            ).properties(
                width=300,
                height=300
            )
        )

        return final_grid_chart
