import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


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
        self.all_variance_subgroup_metrics = [
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
        self.all_group_variance_metrics_lst = [
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

    def create_subgroup_metrics_line_band(self, model_name: str, subgroup_metric):
        subgroup = 'overall'
        subgroup_metrics_df = self.melted_all_subgroup_metrics_per_model_dct[model_name]
        subplot_metrics_df = subgroup_metrics_df[
            (subgroup_metrics_df.Metric == subgroup_metric) &
            (subgroup_metrics_df.Subgroup == subgroup)
        ]

        line = alt.Chart(subplot_metrics_df).mark_line().encode(
            x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
            y=alt.Y('mean(Metric_Value)', type='quantitative', title=subgroup_metric, scale=alt.Scale(zero=False)),
        )

        band = alt.Chart(subplot_metrics_df).mark_errorband(extent='ci').encode(
            x=alt.X(field='Intervention_Param', type='quantitative', title='Alpha'),
            y=alt.Y(field='Metric_Value', type='quantitative', title=subgroup_metric, scale=alt.Scale(zero=False))
        )

        final_chart = (
            (band + line).configure_axis(
                labelFontSize=15 + 2,
                titleFontSize=15 + 4,
                labelFontWeight='normal',
                titleFontWeight='normal',
            ).configure_title(
                fontSize=15 + 2
            ).properties(
                title=alt.TitleParams(f'{model_name} Model', fontSize=16 + 4, anchor='middle', dy=-10),
                width=250,
                height=250,
            )
        )

        return final_chart
