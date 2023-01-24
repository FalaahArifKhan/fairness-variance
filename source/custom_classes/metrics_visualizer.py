import os
import altair as alt
import pandas as pd
import seaborn as sns

from source.custom_classes.metrics_composer import MetricsComposer


class MetricsVisualizer:
    def __init__(self, metrics_path, dataset_name, model_names, sensitive_attributes_dct):
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.sensitive_attributes_dct = sensitive_attributes_dct

        # Read models metrics dfs
        metrics_filenames = [filename for filename in os.listdir(metrics_path)]
        models_metrics_dct = dict()
        models_average_metrics_dct = dict()
        for model_name in model_names:
            for filename in metrics_filenames:
                if dataset_name in filename and model_name in filename:
                    models_metrics_dct[model_name] = pd.read_csv(f'{metrics_path}/{filename}')
                    columns_to_group = [col for col in models_metrics_dct[model_name].columns
                                        if col not in ('Model_Seed', 'Run_Number')]
                    models_average_metrics_dct[model_name] = models_metrics_dct[model_name][columns_to_group].groupby(['Metric', 'Model_Name']).mean().reset_index()
                    break

        # Create one average metrics df with all model_dfs
        models_average_metrics_df = pd.DataFrame()
        for model_name in models_average_metrics_dct.keys():
            model_average_metrics_df = models_average_metrics_dct[model_name]
            models_average_metrics_df = pd.concat([models_average_metrics_df, model_average_metrics_df])

        # Create one metrics df with all model_dfs
        all_models_metrics_df = pd.DataFrame()
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

        # Create a composed metrics df
        models_composed_metrics_df = pd.DataFrame()
        for model_name in models_average_metrics_dct.keys():
            metrics_composer = MetricsComposer(sensitive_attributes_dct, models_average_metrics_dct[model_name])
            model_composed_metrics_df = metrics_composer.compose_metrics()
            model_composed_metrics_df['Model_Name'] = model_name
            models_composed_metrics_df = pd.concat([models_composed_metrics_df, model_composed_metrics_df])

        self.models_metrics_dct = models_metrics_dct
        self.models_average_metrics_dct = models_average_metrics_dct
        self.all_models_metrics_df = all_models_metrics_df
        self.models_average_metrics_df = models_average_metrics_df
        self.models_composed_metrics_df = models_composed_metrics_df
        self.melted_models_composed_metrics_df = self.models_composed_metrics_df.melt(id_vars=["Metric", "Model_Name"],
                                                                                      var_name="Subgroup",
                                                                                      value_name="Value")

    def visualize_overall_metrics(self, metrics_names, reversed_metrics_names=None, x_label="Prediction Metrics"):
        if reversed_metrics_names is None:
            reversed_metrics_names = []
        metrics_names = set(metrics_names + reversed_metrics_names)

        overall_metrics_df = pd.DataFrame()
        for model_name in self.models_average_metrics_dct.keys():
            model_average_results_df = self.models_average_metrics_dct[model_name].copy(deep=True)
            model_average_results_df = model_average_results_df.loc[model_average_results_df['Metric'].isin(metrics_names)]

            overall_model_metrics_df = pd.DataFrame()
            overall_model_metrics_df['overall'] = model_average_results_df['overall']
            overall_model_metrics_df['metric'] = model_average_results_df['Metric']
            overall_model_metrics_df['model_name'] = model_name
            overall_metrics_df = pd.concat([overall_metrics_df, overall_model_metrics_df])

        overall_metrics_df.loc[overall_metrics_df['metric'].isin(reversed_metrics_names), 'overall'] = \
            1 - overall_metrics_df.loc[overall_metrics_df['metric'].isin(reversed_metrics_names), 'overall']

        # Draw a nested barplot
        height = 9 if len(metrics_names) >= 7 else 6
        g = sns.catplot(
            data=overall_metrics_df, kind="bar",
            x="overall", y="metric", hue="model_name",
            # errorbar="sd",
            palette="tab20",
            alpha=.8, height=height
        )
        g.despine(left=True)
        g.set_axis_labels("", x_label)
        g.legend.set_title("")

    def create_models_metrics_bar_chart(self, metrics_lst, metrics_group_name, default_plot_metric=None):
        if default_plot_metric is None:
            default_plot_metric = metrics_lst[0]

        df_for_model_metrics_chart = self.melted_models_composed_metrics_df.loc[self.melted_models_composed_metrics_df['Metric'].isin(metrics_lst)]

        radio_select = alt.selection_single(fields=['Metric'], init={'Metric': default_plot_metric}, empty="none")
        color_condition = alt.condition(radio_select,
                                        alt.Color('Metric:N', legend=None, scale=alt.Scale(scheme="tableau20")),
                                        alt.value('lightgray'))

        models_metrics_chart = (
            alt.Chart(df_for_model_metrics_chart)
            .mark_bar()
            .transform_filter(radio_select)
            .encode(
                x='Value:Q',
                y=alt.Y('Model_Name:N', axis=None),
                color=alt.Color(
                    'Model_Name:N',
                    scale=alt.Scale(scheme="tableau20")
                ),
                row='Subgroup:N',
            )
        )

        select_metric_legend = (
            alt.Chart(df_for_model_metrics_chart)
            .mark_circle(size=200)
            .encode(
                y=alt.Y("Metric:N", axis=alt.Axis(title=f"Select {metrics_group_name} Metric", titleFontSize=15)),
                color=color_condition,
            )
            .add_selection(radio_select)
        )

        color_legend = (
            alt.Chart(df_for_model_metrics_chart)
            .mark_circle(size=200)
            .encode(
                y=alt.Y("Model_Name:N", axis=alt.Axis(title="Model Name", titleFontSize=15)),
                color=alt.Color("Model_Name:N", scale=alt.Scale(scheme="tableau20")),
            )
        )

        return models_metrics_chart, select_metric_legend, color_legend
