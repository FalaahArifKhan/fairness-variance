import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from source.preprocessing import create_models_in_range_dct


def preprocess_metrics(exp_subgroup_metrics_dct, exp_group_metrics_dct):
    # Create melted_exp_subgroup_metrics_dct
    melted_exp_subgroup_metrics_dct = dict()
    for model_name in exp_subgroup_metrics_dct.keys():
        for exp_iter in exp_subgroup_metrics_dct[model_name].keys():
            for intervention_param in exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                for test_set_index in exp_subgroup_metrics_dct[model_name][exp_iter][intervention_param].keys():
                    model_subgroup_metrics_df = exp_subgroup_metrics_dct[model_name][exp_iter][intervention_param][test_set_index]
                    subgroup_names = [col for col in model_subgroup_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
                    melted_model_subgroup_metrics_df = model_subgroup_metrics_df.melt(
                        id_vars=[col for col in model_subgroup_metrics_df.columns if col not in subgroup_names],
                        value_vars=subgroup_names,
                        var_name="Subgroup",
                        value_name="Metric_Value"
                    )
                    melted_exp_subgroup_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {}).setdefault(intervention_param, {})[test_set_index] = melted_model_subgroup_metrics_df

    # Create a dict where a key is a model name and a value is a dataframe with
    # all subgroup metrics, intervention_params, test_set_index, and exp iterations per each model
    melted_all_subgroup_metrics_per_model_dct = dict()
    for model_name in melted_exp_subgroup_metrics_dct.keys():
        all_subgroup_metrics_per_model_df = pd.DataFrame()
        for exp_iter in melted_exp_subgroup_metrics_dct[model_name].keys():
            for intervention_param in melted_exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                for test_set_index in melted_exp_subgroup_metrics_dct[model_name][exp_iter][intervention_param].keys():
                    all_subgroup_metrics_per_model_df = pd.concat([
                        all_subgroup_metrics_per_model_df,
                        melted_exp_subgroup_metrics_dct[model_name][exp_iter][intervention_param][test_set_index],
                    ])
        melted_all_subgroup_metrics_per_model_dct[model_name] = all_subgroup_metrics_per_model_df

    # Create melted_exp_group_metrics_dct
    melted_exp_group_metrics_dct = dict()
    for model_name in exp_group_metrics_dct.keys():
        for exp_iter in exp_group_metrics_dct[model_name].keys():
            for intervention_param in exp_group_metrics_dct[model_name][exp_iter].keys():
                for test_set_index in exp_group_metrics_dct[model_name][exp_iter][intervention_param].keys():
                    model_group_metrics_df = exp_group_metrics_dct[model_name][exp_iter][intervention_param][test_set_index]
                    # All other columns in model_group_metrics_df
                    # except 'Metric', 'Model_Name', 'Intervention_Param', 'Experiment_Iteration' are group names
                    group_names = [col for col in model_group_metrics_df.columns
                                   if col not in ('Metric', 'Model_Name', 'Intervention_Param', 'Test_Set_Index', 'Experiment_Iteration')]
                    melted_model_group_metrics_df = model_group_metrics_df.melt(
                        id_vars=[col for col in model_group_metrics_df.columns if col not in group_names],
                        value_vars=group_names,
                        var_name="Group",
                        value_name="Metric_Value"
                    )
                    melted_exp_group_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {}).setdefault(intervention_param, {})[test_set_index] = melted_model_group_metrics_df

    # Create a dict where a key is a model name and a value is a dataframe with
    # all group metrics, intervention_params, test_set_index, and exp iterations per each model
    melted_all_group_metrics_per_model_dct = dict()
    for model_name in melted_exp_group_metrics_dct.keys():
        all_group_metrics_per_model_df = pd.DataFrame()
        for exp_iter in melted_exp_group_metrics_dct[model_name].keys():
            for intervention_param in melted_exp_group_metrics_dct[model_name][exp_iter].keys():
                for test_set_index in melted_exp_group_metrics_dct[model_name][exp_iter][intervention_param].keys():
                    all_group_metrics_per_model_df = pd.concat([
                        all_group_metrics_per_model_df,
                        melted_exp_group_metrics_dct[model_name][exp_iter][intervention_param][test_set_index],
                    ])
        melted_all_group_metrics_per_model_dct[model_name] = all_group_metrics_per_model_df


    return melted_all_subgroup_metrics_per_model_dct, melted_all_group_metrics_per_model_dct


def create_group_base_and_fair_models_box_plot(all_group_metrics_per_model_dct: dict, metric_names: list, group: str = 'overall',
                                               ylim: tuple = None, test_set_index: int = 0, vals_to_replace: dict = None):
    sns.set_style("whitegrid")

    # Create one metrics df with all model_dfs
    all_models_metrics_df = pd.DataFrame()
    for model_name in all_group_metrics_per_model_dct.keys():
        model_metrics_df = all_group_metrics_per_model_dct[model_name]
        all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

    all_models_metrics_df = all_models_metrics_df.reset_index(drop=True)
    if vals_to_replace is not None:
        all_models_metrics_df = all_models_metrics_df.replace(vals_to_replace)

    group_col_name = 'Subgroup' if group == 'overall' else 'Group'
    to_plot = all_models_metrics_df[
        (all_models_metrics_df['Metric'].isin(metric_names)) &
        (all_models_metrics_df[group_col_name] == group) &
        (all_models_metrics_df['Test_Set_Index'] == test_set_index)
    ]

    plt.figure(figsize=(12, 6))
    g = sns.catplot(kind = "box",
                    data=to_plot,
                    x='Model_Name',
                    y='Metric_Value',
                    hue='Intervention_Param',
                    col='Metric',
                    col_order=metric_names,
                    legend=False)
    # Extra configs for the FacetGrid
    font_increase = 4 if len(metric_names) >= 3 else 6
    g.set_xlabels("")
    g.set_ylabels("Metric Value", fontsize=16 + font_increase)
    g.set_titles(size=14 + font_increase)
    g.tick_params(labelsize=14 + font_increase)
    g.set(ylim=ylim)
    g.despine(left=True)
    g.add_legend(title='Alpha',
                 ncol=1,
                 fancybox=True,
                 shadow=True,
                 fontsize=13 + font_increase)
    plt.setp(g._legend.get_title(), fontsize=14 + font_increase)


def create_group_models_box_plot_per_test_set(all_group_metrics_per_model_dct: dict, metric_name: str, group: str = 'overall',
                                              ylim: tuple = None, vals_to_replace: dict = None):
    sns.set_style("whitegrid")

    # Create one metrics df with all model_dfs
    all_models_metrics_df = pd.DataFrame()
    for model_name in all_group_metrics_per_model_dct.keys():
        model_metrics_df = all_group_metrics_per_model_dct[model_name]
        all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

    all_models_metrics_df = all_models_metrics_df.reset_index(drop=True)
    if vals_to_replace is not None:
        all_models_metrics_df = all_models_metrics_df.replace(vals_to_replace)

    group_col_name = 'Subgroup' if group == 'overall' else 'Group'
    to_plot = all_models_metrics_df[
        (all_models_metrics_df['Metric'] == metric_name) &
        (all_models_metrics_df[group_col_name] == group)
        ]

    plt.figure(figsize=(12, 6))
    g = sns.catplot(kind = "box",
                    data=to_plot,
                    x='Model_Name',
                    y='Metric_Value',
                    hue='Intervention_Param',
                    col='Test_Set_Index',
                    legend=False)
    # Extra configs for the FacetGrid
    font_increase = 8
    g.set_xlabels("")
    g.set_ylabels(metric_name, fontsize=16 + font_increase)
    g.set_titles(size=14 + font_increase)
    g.tick_params(labelsize=14 + font_increase)
    g.set(ylim=ylim)
    g.despine(left=True)
    g.add_legend(title='Alpha',
                 ncol=1,
                 fancybox=True,
                 shadow=True,
                 fontsize=13 + font_increase)
    plt.setp(g._legend.get_title(), fontsize=14 + font_increase)


def create_scatter_plot(all_group_metrics_per_model_dct: dict, group: str,
                        fairness_metric_name: str, stability_metric_name: str, test_set_index: int = 0):
    # Create one metrics df with all model_dfs
    all_models_metrics_df = pd.DataFrame()
    for model_name in all_group_metrics_per_model_dct.keys():
        model_metrics_df = all_group_metrics_per_model_dct[model_name]
        all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

    all_models_metrics_df = all_models_metrics_df.reset_index(drop=True)
    all_models_metrics_df = all_models_metrics_df[
        (all_models_metrics_df['Group'] == group) &
        (all_models_metrics_df['Test_Set_Index'] == test_set_index)
    ]

    models_fairness_metric_df = all_models_metrics_df[
        (all_models_metrics_df['Metric'] == fairness_metric_name)
    ].reset_index(drop=True)
    models_stability_metric_df = all_models_metrics_df[
        (all_models_metrics_df['Metric'] == stability_metric_name)
    ].reset_index(drop=True)

    to_plot = models_fairness_metric_df
    to_plot = to_plot.drop(columns=['Metric', 'Metric_Value'])
    to_plot[fairness_metric_name] = models_fairness_metric_df['Metric_Value']
    to_plot[stability_metric_name] = models_stability_metric_df['Metric_Value']

    chart = alt.Chart(to_plot).mark_point(size=60).encode(
        x=stability_metric_name,
        y=fairness_metric_name,
        color=alt.Color(field='Intervention_Param', type='nominal'),
        shape='Model_Name',
        size=alt.value(80),
    )

    final_chart = (
        chart.configure_axis(
            labelFontSize=15 + 2,
            titleFontSize=15 + 4,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=15 + 2
        ).configure_legend(
            titleFontSize=17 + 2,
            labelFontSize=15 + 2,
            symbolStrokeWidth=4,
            labelLimit=200,
            titleLimit=220,
        ).properties(
            width=500, height=300
        )
    )

    return final_chart


def create_bar_plot_for_model_selection(all_subgroup_metrics_per_model_dct: dict, all_group_metrics_per_model_dct: dict,
                                        metrics_value_range_dct: dict, group: str):
    # Compute the number of models that satisfy the conditions
    models_in_range_df = create_models_in_range_dct(all_subgroup_metrics_per_model_dct, all_group_metrics_per_model_dct,
                                                     metrics_value_range_dct, group)
    # Replace metric groups on their aliases
    metric_name_to_alias_dct = {
        # C1
        'TPR': 'C1',
        'TNR': 'C1',
        'FNR': 'C1',
        'FPR': 'C1',
        'PPV': 'C1',
        'Accuracy': 'C1',
        'F1': 'C1',
        # C2
        'Equalized_Odds_TPR': 'C2',
        'Equalized_Odds_FPR': 'C2',
        'Equalized_Odds_FNR': 'C2',
        'Disparate_Impact': 'C2',
        'Statistical_Parity_Difference': 'C2',
        # C3
        'Std': 'C3',
        'IQR': 'C3',
        'Jitter': 'C3',
        'Label_Stability': 'C3',
        # C4
        'IQR_Parity': 'C4',
        'Label_Stability_Ratio': 'C4',
        'Std_Parity': 'C4',
        'Std_Ratio': 'C4',
        'Jitter_Parity': 'C4',
    }

    def get_column_alias(metric_group):
        if '&' not in metric_group:
            alias = metric_name_to_alias_dct[metric_group]
        else:
            metrics = metric_group.split('&')
            alias = None
            for idx, metric in enumerate(metrics):
                if idx == 0:
                    alias = metric_name_to_alias_dct[metric]
                else:
                    alias += ' & ' + metric_name_to_alias_dct[metric]

        return alias

    models_in_range_df['Alias'] = models_in_range_df['Metric_Group'].apply(get_column_alias)
    models_in_range_df['Title'] = models_in_range_df['Alias']
    # models_in_range_df['Title'] = models_in_range_df.apply(
    #     lambda row: f'{row["Alias"]} = {row["Metric_Group"]}' if '&' not in row["Alias"] else row["Alias"], axis=1
    # )

    base_font_size = 25
    bar_plot = alt.Chart(models_in_range_df).mark_bar().encode(
        x=alt.X("Title", type="nominal", title='Metric Group', axis=alt.Axis(labelAngle=-30),
                sort=alt.Sort(order='ascending')),
        y=alt.Y("Number_of_Models", title="Number of Models", type="quantitative"),
        color=alt.Color('Model_Name', legend=alt.Legend(title='Model Name'))
    ).configure_axis(
        labelFontSize=base_font_size + 2,
        titleFontSize=base_font_size + 4,
        labelFontWeight='normal',
        titleFontWeight='normal',
        labelLimit=300,
    ).configure_title(
        fontSize=base_font_size + 2
    ).configure_legend(
        titleFontSize=base_font_size + 2,
        labelFontSize=base_font_size,
        symbolStrokeWidth=4,
        labelLimit=300,
        titleLimit=220,
        orient='none',
        legendX=345, legendY=10,
    ).properties(width=650, height=450)

    return bar_plot
