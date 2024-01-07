import pandas as pd
from virny.utils.custom_initializers import create_models_metrics_dct_from_database_df
from virny.custom_classes.metrics_composer import MetricsComposer

from source.utils.db_functions import connect_to_mongodb
from source.utils.db_functions import read_model_metric_dfs_from_db
from source.custom_classes.experiments_composer import ExperimentsComposer


def create_ext_subgroup_and_group_metrics_dicts_for_datasets(datasets_db_config: dict, dataset_names: list):
    datasets_exp_metrics_dct = dict()
    for dataset_name in dataset_names:
        # Extract experimental data for the defined dataset from MongoDB
        client, collection_obj, db_writer_func = connect_to_mongodb(datasets_db_config[dataset_name]['db_collection_name'])
        model_metric_dfs = read_model_metric_dfs_from_db(collection_obj, datasets_db_config[dataset_name]['experiment_session_uuid'])
        models_metrics_dct = create_models_metrics_dct_from_database_df(model_metric_dfs)
        client.close()

        # Compose disparity metrics for the defined dataset
        exp_composer = ExperimentsComposer(models_metrics_dct, datasets_db_config[dataset_name]['sensitive_attrs'])
        exp_subgroup_metrics_dct = exp_composer.create_exp_subgroup_metrics_dct()
        exp_group_metrics_dct = exp_composer.compose_group_metrics(exp_subgroup_metrics_dct)

        datasets_exp_metrics_dct[dataset_name] = {
            'subgroup_metrics': exp_subgroup_metrics_dct,
            'group_metrics': exp_group_metrics_dct,
        }
        print(f'Extracted metrics for {dataset_name} dataset')

    return datasets_exp_metrics_dct


def create_metrics_dicts_for_diff_fairness_interventions(datasets_db_config: dict, datasets_sensitive_attrs_dct: dict,
                                                         db_collection_name: str):
    client, collection_obj, db_writer_func = connect_to_mongodb(db_collection_name)
    all_subgroup_metrics_df = pd.DataFrame()
    all_group_metrics_df = pd.DataFrame()
    for dataset_name in datasets_db_config.keys():
        for fairness_intervention in datasets_db_config[dataset_name].keys():
            # Extract experimental data for the defined dataset from MongoDB
            model_metric_df = read_model_metric_dfs_from_db(collection_obj, datasets_db_config[dataset_name][fairness_intervention])
            model_metric_df = model_metric_df.drop(columns=['Model_Params', 'Tag', 'Model_Init_Seed'])
            if dataset_name == 'Student_Performance_Por':
                if fairness_intervention == 'Baseline':
                    model_metric_df = model_metric_df[model_metric_df['intervention_param'] == 0.0]
                    print('Filtered to alpha = 0.0 for baseline')
                elif fairness_intervention == 'DIR':
                    model_metric_df = model_metric_df[model_metric_df['intervention_param'] == 0.7]
                    print('Filtered to alpha = 0.7 for DIR')

            model_metric_df['Fairness_Intervention'] = fairness_intervention
            all_subgroup_metrics_df = pd.concat([all_subgroup_metrics_df, model_metric_df])

            # Compose disparity metrics for the defined dataset
            models_metrics_dct = create_models_metrics_dct_from_database_df(model_metric_df)
            cur_sensitive_attrs_dct = {attr: None for attr in datasets_sensitive_attrs_dct[dataset_name]}
            metrics_composer = MetricsComposer(models_metrics_dct, cur_sensitive_attrs_dct)
            model_composed_metrics_df = metrics_composer.compose_metrics()
            model_composed_metrics_df['Dataset_Name'] = dataset_name
            model_composed_metrics_df['Fairness_Intervention'] = fairness_intervention

            # Unpivot group columns to align with visualizations API
            unpivot_composed_metrics_df = unpivot_group_metrics(model_composed_metrics_df, datasets_sensitive_attrs_dct[dataset_name])
            all_group_metrics_df = pd.concat([all_group_metrics_df, unpivot_composed_metrics_df])

            print(f'Extracted metrics for {dataset_name} dataset and {fairness_intervention} intervention')

    client.close()
    return all_subgroup_metrics_df, all_group_metrics_df


def unpivot_group_metrics(model_composed_metrics_df, sensitive_attrs):
    id_vars = [col for col in model_composed_metrics_df.columns if col not in sensitive_attrs]
    return pd.melt(model_composed_metrics_df,
                   id_vars=id_vars,
                   value_vars=sensitive_attrs,
                   var_name='Group',
                   value_name='Metric_Value')


def create_metrics_df_for_diff_dataset_groups(group_metrics_df, metric_name, dataset_groups_dct, dataset_names):
    subplot_metrics_df = pd.DataFrame()
    for dataset_name in dataset_names:
        dataset_metrics_df = group_metrics_df[
            (group_metrics_df.Metric == metric_name) &
            (group_metrics_df.Group == dataset_groups_dct[dataset_name]) &
            (group_metrics_df.Dataset_Name == dataset_name)
            ]
        subplot_metrics_df = pd.concat([subplot_metrics_df, dataset_metrics_df])

    return subplot_metrics_df


def create_melted_subgroup_and_group_dicts(exp_subgroup_metrics_dct, exp_group_metrics_dct):
    # Create melted_exp_subgroup_metrics_dct
    melted_exp_subgroup_metrics_dct = dict()
    for model_name in exp_subgroup_metrics_dct.keys():
        for exp_iter in exp_subgroup_metrics_dct[model_name].keys():
            for percentage in exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                model_subgroup_metrics_df = exp_subgroup_metrics_dct[model_name][exp_iter][percentage]
                subgroup_names = [col for col in model_subgroup_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
                melted_model_subgroup_metrics_df = model_subgroup_metrics_df.melt(
                    id_vars=[col for col in model_subgroup_metrics_df.columns if col not in subgroup_names],
                    value_vars=subgroup_names,
                    var_name="Subgroup",
                    value_name="Metric_Value"
                )
                melted_exp_subgroup_metrics_dct.setdefault(model_name, {}) \
                    .setdefault(exp_iter, {})[percentage] = melted_model_subgroup_metrics_df

    melted_exp_subgroup_metrics_dct = melted_exp_subgroup_metrics_dct

    # Create a dict where a key is a model name and a value is a dataframe with
    # all subgroup metrics, percentages, and exp iterations per each model
    melted_all_subgroup_metrics_per_model_dct = dict()
    for model_name in melted_exp_subgroup_metrics_dct.keys():
        all_subgroup_metrics_per_model_df = pd.DataFrame()
        for exp_iter in melted_exp_subgroup_metrics_dct[model_name].keys():
            for percentage in melted_exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                all_subgroup_metrics_per_model_df = pd.concat([
                    all_subgroup_metrics_per_model_df,
                    melted_exp_subgroup_metrics_dct[model_name][exp_iter][percentage],
                ])
        melted_all_subgroup_metrics_per_model_dct[model_name] = all_subgroup_metrics_per_model_df

    melted_all_subgroup_metrics_per_model_dct = melted_all_subgroup_metrics_per_model_dct

    # Create melted_exp_group_metrics_dct
    melted_exp_group_metrics_dct = dict()
    for model_name in exp_group_metrics_dct.keys():
        for exp_iter in exp_group_metrics_dct[model_name].keys():
            for percentage in exp_group_metrics_dct[model_name][exp_iter].keys():
                model_group_metrics_df = exp_group_metrics_dct[model_name][exp_iter][percentage]
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

    melted_exp_group_metrics_dct = melted_exp_group_metrics_dct

    # Create a dict where a key is a model name and a value is a dataframe with
    # all group metrics, percentages, and exp iterations per each model
    melted_all_group_metrics_per_model_dct = dict()
    for model_name in melted_exp_group_metrics_dct.keys():
        all_group_metrics_per_model_df = pd.DataFrame()
        for exp_iter in melted_exp_group_metrics_dct[model_name].keys():
            for percentage in melted_exp_group_metrics_dct[model_name][exp_iter].keys():
                all_group_metrics_per_model_df = pd.concat([
                    all_group_metrics_per_model_df,
                    melted_exp_group_metrics_dct[model_name][exp_iter][percentage],
                ])
        melted_all_group_metrics_per_model_dct[model_name] = all_group_metrics_per_model_df

    melted_all_group_metrics_per_model_dct = melted_all_group_metrics_per_model_dct

    return melted_all_subgroup_metrics_per_model_dct, melted_all_group_metrics_per_model_dct
