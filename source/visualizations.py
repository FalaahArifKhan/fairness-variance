import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def create_group_base_and_fair_models_box_plot(all_group_metrics_per_model_dct: dict, metric_name: str, group: str = 'overall',
                                               legend_location: str = 'upper left', test_set_index: int = 0):
    sns.set_style("darkgrid")

    # Create one metrics df with all model_dfs
    all_models_metrics_df = pd.DataFrame()
    for model_name in all_group_metrics_per_model_dct.keys():
        model_metrics_df = all_group_metrics_per_model_dct[model_name]
        all_models_metrics_df = pd.concat([all_models_metrics_df, model_metrics_df])

    all_models_metrics_df = all_models_metrics_df.reset_index(drop=True)

    group_col_name = 'Subgroup' if group == 'overall' else 'Group'
    to_plot = all_models_metrics_df[
        (all_models_metrics_df['Metric'] == metric_name) &
        (all_models_metrics_df[group_col_name] == group) &
        (all_models_metrics_df['Test_Set_Index'] == test_set_index)
    ]

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x=to_plot['Model_Name'],
                     y=to_plot['Metric_Value'],
                     hue=to_plot['Intervention_Param'])

    plt.legend(loc=legend_location,
               title='Alpha',
               title_fontsize=14,
               ncol=2,
               fancybox=True,
               shadow=True,
               fontsize=13)
    plt.xlabel("Model name", fontsize=16)
    plt.ylabel("Metric value", fontsize=16)
    ax.tick_params(labelsize=14)
    fig = ax.get_figure()
    fig.tight_layout()
