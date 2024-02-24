import pandas as pd
import altair as alt

from altair.utils.schemapi import Undefined
from virny.utils.custom_initializers import create_models_metrics_dct_from_database_df

from source.utils.db_functions import connect_to_mongodb
from source.utils.db_functions import read_model_metric_dfs_from_db
from source.custom_classes.experiments_composer import ExperimentsComposer
from source.visualizations import preprocess_metrics


def create_exp_metrics_dicts_for_mult_train_set_sizes(datasets_db_config: dict, experiment_names: list,
                                                      db_collection_name: str, sensitive_attrs: list):
    # Extract experimental data for the defined dataset from MongoDB
    client, collection_obj, _ = connect_to_mongodb(db_collection_name)

    metrics_per_exp_dct = dict()
    for exp_name in experiment_names:
        subgroup_metrics_per_train_set_df = pd.DataFrame()
        group_metrics_per_train_set_df = pd.DataFrame()
        for train_set_size in datasets_db_config[exp_name].keys():
            experiment_session_uuid = datasets_db_config[exp_name][train_set_size]
            model_metric_dfs = read_model_metric_dfs_from_db(collection_obj, experiment_session_uuid)
            model_metric_dfs = model_metric_dfs[model_metric_dfs['Metric'] != 'Sample_Size']  # Remove non-metric
            models_metrics_dct = create_models_metrics_dct_from_database_df(model_metric_dfs)

            # Compose disparity metrics for the defined dataset
            exp_composer = ExperimentsComposer(models_metrics_dct, sensitive_attrs)
            exp_subgroup_metrics_dct = exp_composer.create_exp_subgroup_metrics_dct_for_mult_test_sets()
            exp_group_metrics_dct = exp_composer.compose_group_metrics_for_mult_test_sets(exp_subgroup_metrics_dct)

            melted_all_subgroup_metrics_per_model_dct, melted_all_group_metrics_per_model_dct = (
                preprocess_metrics(exp_subgroup_metrics_dct, exp_group_metrics_dct))

            # Concat metrics for all models into a common df
            subgroup_metrics_for_train_set_df = pd.DataFrame()
            group_metrics_for_train_set_df = pd.DataFrame()
            for model_name in melted_all_subgroup_metrics_per_model_dct.keys():
                subgroup_metrics_for_train_set_df = pd.concat([subgroup_metrics_for_train_set_df,
                                                               melted_all_subgroup_metrics_per_model_dct[model_name]])
                group_metrics_for_train_set_df = pd.concat([group_metrics_for_train_set_df,
                                                            melted_all_group_metrics_per_model_dct[model_name]])

            subgroup_metrics_for_train_set_df['Train_Set_Size'] = train_set_size
            group_metrics_for_train_set_df['Train_Set_Size'] = train_set_size

            subgroup_metrics_per_train_set_df = pd.concat([subgroup_metrics_per_train_set_df, subgroup_metrics_for_train_set_df])
            group_metrics_per_train_set_df = pd.concat([group_metrics_per_train_set_df, group_metrics_for_train_set_df])

            print(f'Extracted metrics for {exp_name} and {train_set_size} train set size')

        print('\n')
        metrics_per_exp_dct[exp_name] = {
            'subgroup_metrics': subgroup_metrics_per_train_set_df,
            'group_metrics': group_metrics_per_train_set_df,
        }

    client.close()

    return metrics_per_exp_dct


def get_line_bands_plot_for_exp_metrics(exp_metrics_dct: dict, model_name: str, group_name: str,
                                        metric_type: str, metric_name: str, title: str, base_font_size: int,
                                        ylim=Undefined, with_band=True):
    if metric_type == 'subgroup':
        metrics_per_exp_df = exp_metrics_dct['subgroup_metrics']
        metrics_per_exp_df['Group'] = metrics_per_exp_df['Subgroup']
    else:
        metrics_per_exp_df = exp_metrics_dct['group_metrics']

    # Create epistemic uncertainty based on aleatoric and overall uncertainty
    if metric_name == 'Epistemic_Uncertainty':
        temp_metrics_df = metrics_per_exp_df[(metrics_per_exp_df['Model_Name'] == model_name) &
                                             (metrics_per_exp_df['Metric'].isin(['Aleatoric_Uncertainty', 'Overall_Uncertainty'])) &
                                             (metrics_per_exp_df['Group'] == group_name)]

        # Create columns based on values in the Subgroup column
        subplot_metrics_df = temp_metrics_df.pivot(columns='Metric', values='Metric_Value',
                                                   index=[col for col in temp_metrics_df.columns
                                                          if col not in ('Metric', 'Metric_Value')]).reset_index()
        subplot_metrics_df = subplot_metrics_df.rename_axis(None, axis=1)
        subplot_metrics_df['Epistemic_Uncertainty'] = subplot_metrics_df['Overall_Uncertainty'] - subplot_metrics_df['Aleatoric_Uncertainty']
        subplot_metrics_df['Metric_Value'] = subplot_metrics_df['Epistemic_Uncertainty'] # Added to align with the downstream code
    # Find any other uncertainty except Epistemic and Aleatoric uncertainties
    elif metric_name == 'Other_Uncertainty':
        temp_metrics_df = metrics_per_exp_df[(metrics_per_exp_df['Model_Name'] == model_name) &
                                             (metrics_per_exp_df['Metric'].isin(['Aleatoric_Uncertainty', 'Overall_Uncertainty', 'Std'])) &
                                             (metrics_per_exp_df['Group'] == group_name)]

        # Create columns based on values in the Subgroup column
        subplot_metrics_df = temp_metrics_df.pivot(columns='Metric', values='Metric_Value',
                                                   index=[col for col in temp_metrics_df.columns
                                                          if col not in ('Metric', 'Metric_Value')]).reset_index()
        subplot_metrics_df = subplot_metrics_df.rename_axis(None, axis=1)
        subplot_metrics_df['Other_Uncertainty'] = subplot_metrics_df['Overall_Uncertainty'] - (subplot_metrics_df['Aleatoric_Uncertainty'] + subplot_metrics_df['Std'] ** 2)
        subplot_metrics_df['Metric_Value'] = subplot_metrics_df['Other_Uncertainty'] # Added to align with the downstream code
    else:
        subplot_metrics_df = metrics_per_exp_df[(metrics_per_exp_df['Model_Name'] == model_name) &
                                                (metrics_per_exp_df['Metric'] == metric_name) &
                                                (metrics_per_exp_df['Group'] == group_name)]

    # Set an extended model name for plot colors
    subplot_metrics_df['Extended_Model_Name'] = None
    subplot_metrics_df['Extended_Model_Name'].loc[(subplot_metrics_df['Intervention_Param'] == 0.0) &
                                                  (subplot_metrics_df['Test_Set_Index'] == 0)] = 'In-domain Unfair Model'
    subplot_metrics_df['Extended_Model_Name'].loc[(subplot_metrics_df['Intervention_Param'] == 0.7) &
                                                  (subplot_metrics_df['Test_Set_Index'] == 0)] = 'In-domain Fair Model'
    subplot_metrics_df['Extended_Model_Name'].loc[(subplot_metrics_df['Intervention_Param'] == 0.0) &
                                                  (subplot_metrics_df['Test_Set_Index'] == 1)] = 'Out-of-domain Unfair Model'
    subplot_metrics_df['Extended_Model_Name'].loc[(subplot_metrics_df['Intervention_Param'] == 0.7) &
                                                  (subplot_metrics_df['Test_Set_Index'] == 1)] = 'Out-of-domain Fair Model'

    train_set_sizes = metrics_per_exp_df['Train_Set_Size'].unique().tolist()
    min_train_set_size = str(min(train_set_sizes))
    line_chart = alt.Chart(subplot_metrics_df).mark_line().encode(
        x=alt.X(field='Train_Set_Size', type='quantitative', title='Train Set Size',
                scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 5000) || (datum.value == 10000) ? '{min_train_set_size[0] + ',' + min_train_set_size[1:]}' : datum.label")),
        y=alt.Y('mean(Metric_Value)', type='quantitative', title=metric_name.replace('Parity', 'Difference'), scale=alt.Scale(zero=False, domain=ylim)),
        color=alt.Color('Extended_Model_Name:N', title=None),
    )
    if with_band:
        band_chart = alt.Chart(subplot_metrics_df).mark_errorband(extent="ci").encode(
            x=alt.X(field='Train_Set_Size', type='quantitative', title='Train Set Size',
                    scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 5000) || (datum.value == 10000) ? '{min_train_set_size[0] + ',' + min_train_set_size[1:]}' : datum.label")),
            y=alt.Y(field='Metric_Value', type='quantitative', title=metric_name.replace('Parity', 'Difference'), scale=alt.Scale(zero=False, domain=ylim)),
            color='Extended_Model_Name:N',
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    base_chart = base_chart.properties(
        width=300, height=300,
        title=alt.TitleParams(text=title, fontSize=base_font_size + 6),
    )

    return base_chart


def create_group_metric_line_bands_per_dataset_plot(metrics_per_exp_dct: dict, experiment_names: list,
                                                    model_name: str, group_name: str, metric_type: str, metric_name: str,
                                                    ylim=Undefined, with_band=True):
    base_font_size = 20
    exp_name_to_title_dct = {
        'rich_experiment': 'Trained on a high-income set',
        'poor_experiment': 'Trained on a low-income set',
    }
    base_chart1 = get_line_bands_plot_for_exp_metrics(exp_metrics_dct=metrics_per_exp_dct[experiment_names[0]],
                                                      model_name=model_name,
                                                      group_name=group_name,
                                                      metric_type=metric_type,
                                                      metric_name=metric_name,
                                                      title=exp_name_to_title_dct[experiment_names[0]],
                                                      base_font_size=base_font_size,
                                                      ylim=ylim,
                                                      with_band=with_band)
    base_chart2 = get_line_bands_plot_for_exp_metrics(exp_metrics_dct=metrics_per_exp_dct[experiment_names[1]],
                                                      model_name=model_name,
                                                      group_name=group_name,
                                                      metric_type=metric_type,
                                                      metric_name=metric_name,
                                                      title=exp_name_to_title_dct[experiment_names[1]],
                                                      base_font_size=base_font_size,
                                                      ylim=ylim,
                                                      with_band=with_band)
    # Concatenate two base charts
    main_base_chart = alt.vconcat()
    row = alt.hconcat()
    row |= base_chart1
    row |= base_chart2
    main_base_chart &= row

    final_grid_chart = (
        main_base_chart.configure_axis(
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
        )
    )

    return final_grid_chart


def create_plot_for_diff_bootstrap_sizes(all_models_metrics_df: pd.DataFrame, dataset_name: str,
                                         model_name: str, metric_name: str, group: str = 'overall',
                                         ylim=Undefined, vals_to_replace: dict = None, with_band: bool = True):
    if vals_to_replace is not None:
        all_models_metrics_df = all_models_metrics_df.replace(vals_to_replace)

    group_col_name = 'Subgroup' if group == 'overall' else 'Group'
    to_plot = all_models_metrics_df[
        (all_models_metrics_df['Dataset_Name'] == dataset_name) &
        (all_models_metrics_df['Model_Name'] == model_name) &
        (all_models_metrics_df['Metric'] == metric_name) &
        (all_models_metrics_df[group_col_name] == group)
        ]

    line_chart = alt.Chart(to_plot).mark_line().encode(
        x=alt.X(field='Num_Estimators', type='quantitative', title='Number of Estimators',
                scale=alt.Scale(nice=False, zero=False, domain=[50, 200])),
        y=alt.Y('mean(Metric_Value)', type='quantitative', title=metric_name, scale=alt.Scale(zero=False, domain=ylim)),
    )
    if with_band:
        band_chart = alt.Chart(to_plot).mark_errorband(extent="ci").encode(
            x=alt.X(field='Num_Estimators', type='quantitative', title='Number of Estimators',
                    scale=alt.Scale(nice=False, zero=False, domain=[50, 200])),
            y=alt.Y(field='Metric_Value', type='quantitative', title=metric_name, scale=alt.Scale(zero=False, domain=ylim)),
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    base_font_size = 20
    base_chart = (
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
        )
    )

    return base_chart


def create_plot_for_diff_bootstrap_sizes_and_exp_run(all_models_metrics_df: pd.DataFrame, dataset_name: str,
                                                     model_name: str, metric_name: str, group: str = 'overall',
                                                     ylim=Undefined, vals_to_replace: dict = None):
    if vals_to_replace is not None:
        all_models_metrics_df = all_models_metrics_df.replace(vals_to_replace)

    group_col_name = 'Subgroup' if group == 'overall' else 'Group'
    to_plot = all_models_metrics_df[
        (all_models_metrics_df['Dataset_Name'] == dataset_name) &
        (all_models_metrics_df['Model_Name'] == model_name) &
        (all_models_metrics_df['Metric'] == metric_name) &
        (all_models_metrics_df[group_col_name] == group)
        ]

    line_chart = alt.Chart(to_plot).mark_line().encode(
        x=alt.X(field='Num_Estimators', type='quantitative', title='Number of Estimators',
                scale=alt.Scale(nice=False, zero=False, domain=[50, 200])),
        y=alt.Y('mean(Metric_Value)', type='quantitative', title=metric_name, scale=alt.Scale(zero=False, domain=ylim)),
        color='Experiment_Iteration:N',
    )

    base_font_size = 20
    base_chart = line_chart
    base_chart = (
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
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle'
        )
    )

    return base_chart


def get_line_bands_per_correct_incorrect_group(metrics_per_exp_df: pd.DataFrame, model_name: str, group_name: str,
                                               metric_name: str, intervention_param: float, test_set_index: int,
                                               title: str, base_font_size: int, ylim=Undefined, with_band=True):
    groups = [group_name + suffix for suffix in ['_priv_correct', '_dis_correct', '_priv_incorrect', '_dis_incorrect']]

    # Create epistemic uncertainty based on aleatoric and overall uncertainty
    if metric_name == 'Epistemic_Uncertainty':
        temp_metrics_df = metrics_per_exp_df[(metrics_per_exp_df['Model_Name'] == model_name) &
                                             (metrics_per_exp_df['Metric'].isin(['Aleatoric_Uncertainty', 'Overall_Uncertainty'])) &
                                             (metrics_per_exp_df['Group'].isin(groups)) &
                                             (metrics_per_exp_df['Intervention_Param'] == intervention_param) &
                                             (metrics_per_exp_df['Test_Set_Index'] == test_set_index)]

        # Create columns based on values in the Subgroup column
        subplot_metrics_df = temp_metrics_df.pivot(columns='Metric', values='Metric_Value',
                                                   index=[col for col in temp_metrics_df.columns
                                                          if col not in ('Metric', 'Metric_Value')]).reset_index()
        subplot_metrics_df = subplot_metrics_df.rename_axis(None, axis=1)
        subplot_metrics_df['Epistemic_Uncertainty'] = subplot_metrics_df['Overall_Uncertainty'] - subplot_metrics_df['Aleatoric_Uncertainty']
        subplot_metrics_df['Metric_Value'] = subplot_metrics_df['Epistemic_Uncertainty'] # Added to align with the downstream code
    else:
        subplot_metrics_df = metrics_per_exp_df[(metrics_per_exp_df['Model_Name'] == model_name) &
                                                (metrics_per_exp_df['Metric'] == metric_name) &
                                                (metrics_per_exp_df['Group'].isin(groups)) &
                                                (metrics_per_exp_df['Intervention_Param'] == intervention_param) &
                                                (metrics_per_exp_df['Test_Set_Index'] == test_set_index)]

    train_set_sizes = metrics_per_exp_df['Train_Set_Size'].unique().tolist()
    min_train_set_size = str(min(train_set_sizes))
    line_chart = alt.Chart(subplot_metrics_df).mark_line().encode(
        x=alt.X(field='Train_Set_Size', type='quantitative', title='Train Set Size',
                scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"datum.value == 10000 ? '{min_train_set_size[0] + ',' + min_train_set_size[1:]}' : datum.label")),
        y=alt.Y('mean(Metric_Value)', type='quantitative', title=metric_name, scale=alt.Scale(zero=False, domain=ylim)),
        color='Group:N',
    )
    if with_band:
        band_chart = alt.Chart(subplot_metrics_df).mark_errorband(extent="ci").encode(
            x=alt.X(field='Train_Set_Size', type='quantitative', title='Train Set Size',
                    scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"datum.value == 10000 ? '{min_train_set_size[0] + ',' + min_train_set_size[1:]}' : datum.label")),
            y=alt.Y(field='Metric_Value', type='quantitative', title=metric_name, scale=alt.Scale(zero=False, domain=ylim)),
            color='Group:N',
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    base_chart = base_chart.properties(
        width=275, height=275,
        title=alt.TitleParams(text=title, fontSize=base_font_size + 4),
    )

    return base_chart


def create_line_bands_per_correct_incorrect_group(exp_metrics_dct: dict, model_name: str, group_name: str,
                                                  metric_type: str, metric_name: str, ylim=Undefined, with_band=True):
    if metric_type == 'subgroup':
        metrics_per_exp_df = exp_metrics_dct['subgroup_metrics']
        metrics_per_exp_df['Group'] = metrics_per_exp_df['Subgroup']
    else:
        metrics_per_exp_df = exp_metrics_dct['group_metrics']

    extended_model_name_to_params = {
        'In-domain Unfair Model': {'Intervention_Param': 0.0, 'Test_Set_Index': 0},
        'In-domain Fair Model': {'Intervention_Param': 0.7, 'Test_Set_Index': 0},
        'Out-of-domain Unfair Model': {'Intervention_Param': 0.0, 'Test_Set_Index': 1},
        'Out-of-domain Fair Model': {'Intervention_Param': 0.7, 'Test_Set_Index': 1},
    }
    extended_model_names = list(extended_model_name_to_params.keys())

    base_font_size = 20
    num_rows, num_cols = 2, 2
    subgroups_grid_chart = alt.vconcat()
    extended_model_name_idx = -1
    for _ in range(num_rows):
        row = alt.hconcat()
        for i in range(num_cols):
            extended_model_name_idx += 1
            cur_extended_model_name = extended_model_names[extended_model_name_idx]

            base_chart = get_line_bands_per_correct_incorrect_group(
                metrics_per_exp_df=metrics_per_exp_df,
                model_name=model_name,
                group_name=group_name,
                metric_name=metric_name,
                intervention_param=extended_model_name_to_params[cur_extended_model_name]['Intervention_Param'],
                test_set_index=extended_model_name_to_params[cur_extended_model_name]['Test_Set_Index'],
                title=cur_extended_model_name,
                base_font_size=base_font_size,
                ylim=ylim,
                with_band=with_band
            )

            row |= base_chart

        subgroups_grid_chart &= row

    final_grid_chart = (
        subgroups_grid_chart.configure_axis(
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
            # columns=2,
            # orient='top',
            # direction='horizontal',
            # titleAnchor='middle'
        )
    )

    return final_grid_chart
