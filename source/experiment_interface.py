import os
import copy
from pprint import pprint
from tqdm.notebook import tqdm
from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer

from virny.user_interfaces.metrics_computation_interfaces import compute_metrics_multiple_runs_with_db_writer, \
    compute_metrics_multiple_runs_with_multiple_test_sets
from virny.utils.custom_initializers import create_models_config_from_tuned_params_df
from virny.preprocessing.basic_preprocessing import preprocess_dataset

from source.utils.model_tuning_utils import tune_ML_models
from source.utils.custom_logger import get_logger
from source.preprocessing import remove_correlation, remove_correlation_for_mult_test_sets, \
    remove_disparate_impact, get_preprocessor_for_diabetes, preprocess_mult_data_loaders_for_disp_imp, \
    remove_disparate_impact_with_mult_sets, get_simple_preprocessor


def run_exp_iter_with_preprocessing_intervention(data_loader, experiment_seed, test_set_fraction,
                                                 db_writer_func, fair_intervention_params_lst,
                                                 column_transformer: ColumnTransformer, models_params_for_tuning,
                                                 metrics_computation_config, custom_table_fields_dct,
                                                 with_tuning: bool = False, save_results_dir_path: str = None,
                                                 tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                                                 verbose: bool = False):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Set seeds for metrics computation
    metrics_computation_config.runs_seed_lst = [experiment_seed + i for i in range(1, metrics_computation_config.num_runs + 1)]

    # Preprocess the dataset using the defined preprocessor
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
    if verbose:
        logger.info("The dataset is preprocessed")
        print("Top indexes of an X_test in a base flow dataset: ", base_flow_dataset.X_test.index[:20])
        print("Top indexes of an y_test in a base flow dataset: ", base_flow_dataset.y_test.index[:20])

    for intervention_idx, intervention_param in tqdm(enumerate(fair_intervention_params_lst),
                                                     total=len(fair_intervention_params_lst),
                                                     desc="Multiple alphas",
                                                     colour="#40E0D0"):
        print('intervention_param: ', intervention_param)
        custom_table_fields_dct['intervention_param'] = intervention_param

        # Fair preprocessing
        cur_base_flow_dataset = remove_correlation(base_flow_dataset, alpha=intervention_param)
        # Tune model parameters if needed
        if with_tuning:
            # Tune models and create a models config for metrics computation
            tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, cur_base_flow_dataset,
                                                            metrics_computation_config.dataset_name,
                                                            n_folds=num_folds_for_tuning)

            # Create models_config from the saved tuned_params_df for higher reliability
            date_time_str = datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")
            os.makedirs(save_results_dir_path, exist_ok=True)
            tuned_df_path = os.path.join(save_results_dir_path,
                                         f'tuning_results_{metrics_computation_config.dataset_name}_alpha_{intervention_param}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_multiple_runs_with_db_writer(dataset=cur_base_flow_dataset,
                                                     config=metrics_computation_config,
                                                     models_config=models_config,
                                                     custom_tbl_fields_dct=custom_table_fields_dct,
                                                     db_writer_func=db_writer_func,
                                                     verbose=0)

    logger.info("Experiment run was successful!")


def run_exp_iter_with_disparate_impact(data_loader, experiment_seed, test_set_fraction, db_writer_func,
                                       fair_intervention_params_lst, models_params_for_tuning,
                                       metrics_computation_config, custom_table_fields_dct,
                                       with_tuning: bool = False, save_results_dir_path: str = None,
                                       tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                                       verbose: bool = False, dataset_name: str = 'ACSIncomeDataset'):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Set seeds for metrics computation
    metrics_computation_config.runs_seed_lst = [experiment_seed + i for i in range(1, metrics_computation_config.num_runs + 1)]

    # Add RACE column for DisparateImpactRemover and remove 'SEX', 'RAC1P' to create a blind estimator
    init_data_loader = copy.deepcopy(data_loader)

    if dataset_name in ('ACSIncomeDataset', 'ACSPublicCoverageDataset'):
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        data_loader.X_data['RACE'] = data_loader.X_data['RAC1P'].apply(lambda x: 1 if x == '1' else 0)
        data_loader.full_df = data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

        # Preprocess the dataset using the defined preprocessor
        column_transformer = get_simple_preprocessor(data_loader)
        base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
        base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
        base_flow_dataset.X_train_val['RACE'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'RACE']
        base_flow_dataset.X_test['RACE'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'RACE']

    elif dataset_name == 'DiabetesDataset':
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('gender', 'race')]
        data_loader.X_data['race_binary'] = data_loader.X_data['race'].apply(lambda x: 1 if x == 'Caucasian' else 0)
        data_loader.full_df = data_loader.full_df.drop(['gender', 'race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['gender', 'race'], axis=1)

        # Preprocess the dataset using the defined preprocessor
        column_transformer = get_preprocessor_for_diabetes(data_loader)
        base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
        base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
        base_flow_dataset.X_train_val['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'race_binary']
        base_flow_dataset.X_test['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'race_binary']

    elif dataset_name == 'RicciDataset':
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('Race')]
        data_loader.X_data['race_binary'] = data_loader.X_data['Race'].apply(lambda x: 1 if x == 'White' else 0)
        data_loader.full_df = data_loader.full_df.drop(['Race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['Race'], axis=1)

        # Preprocess the dataset using the defined preprocessor
        column_transformer = get_simple_preprocessor(data_loader)
        base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
        base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
        base_flow_dataset.X_train_val['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'race_binary']
        base_flow_dataset.X_test['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'race_binary']

    elif dataset_name == 'LawSchoolDataset':
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('male', 'race')]
        data_loader.X_data['race_binary'] = data_loader.X_data['race'].apply(lambda x: 1 if x == 'White' else 0)
        data_loader.full_df = data_loader.full_df.drop(['male', 'race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['male', 'race'], axis=1)

        # Preprocess the dataset using the defined preprocessor
        column_transformer = get_simple_preprocessor(data_loader)
        base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
        base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
        base_flow_dataset.X_train_val['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, 'race_binary']
        base_flow_dataset.X_test['race_binary'] = data_loader.X_data.loc[base_flow_dataset.X_test.index, 'race_binary']

    if verbose:
        logger.info("The dataset is preprocessed")
        print("Top indexes of an X_test in a base flow dataset: ", base_flow_dataset.X_test.index[:20])
        print("Top indexes of an y_test in a base flow dataset: ", base_flow_dataset.y_test.index[:20])

    for intervention_idx, intervention_param in tqdm(enumerate(fair_intervention_params_lst),
                                                     total=len(fair_intervention_params_lst),
                                                     desc="Multiple alphas",
                                                     colour="#40E0D0"):
        print('intervention_param: ', intervention_param)
        custom_table_fields_dct['intervention_param'] = intervention_param

        # Fair preprocessing
        cur_base_flow_dataset = remove_disparate_impact(base_flow_dataset, alpha=intervention_param)

        # Tune model parameters if needed
        if with_tuning:
            # Tune models and create a models config for metrics computation
            tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, cur_base_flow_dataset,
                                                            metrics_computation_config.dataset_name,
                                                            n_folds=num_folds_for_tuning)

            # Create models_config from the saved tuned_params_df for higher reliability
            date_time_str = datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")
            os.makedirs(save_results_dir_path, exist_ok=True)
            tuned_df_path = os.path.join(save_results_dir_path,
                                         f'tuning_results_{metrics_computation_config.dataset_name}_alpha_{intervention_param}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_multiple_runs_with_db_writer(dataset=cur_base_flow_dataset,
                                                     config=metrics_computation_config,
                                                     models_config=models_config,
                                                     custom_tbl_fields_dct=custom_table_fields_dct,
                                                     db_writer_func=db_writer_func,
                                                     verbose=0)

    logger.info("Experiment run was successful!")


def run_exp_iter_with_disparate_impact_and_mult_sets(data_loader, extra_data_loaders, experiment_seed, test_set_fraction,
                                                     db_writer_func, fair_intervention_params_lst, models_params_for_tuning,
                                                     metrics_computation_config, custom_table_fields_dct,
                                                     with_tuning: bool = False, save_results_dir_path: str = None,
                                                     tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                                                     verbose: bool = False):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Set seeds for metrics computation
    metrics_computation_config.runs_seed_lst = [experiment_seed + i for i in range(1, metrics_computation_config.num_runs + 1)]

    init_base_flow_dataset, extra_base_flow_datasets = \
        preprocess_mult_data_loaders_for_disp_imp(data_loaders=[data_loader] + extra_data_loaders,
                                                  test_set_fraction=test_set_fraction,
                                                  experiment_seed=experiment_seed)
    if verbose:
        logger.info("The dataset is preprocessed")
        print("Top indexes of an X_test in a base flow dataset: ", init_base_flow_dataset.X_test.index[:20])
        print("Top indexes of an y_test in a base flow dataset: ", init_base_flow_dataset.y_test.index[:20])

    for intervention_idx, intervention_param in tqdm(enumerate(fair_intervention_params_lst),
                                                     total=len(fair_intervention_params_lst),
                                                     desc="Multiple alphas",
                                                     colour="#40E0D0"):
        print('intervention_param: ', intervention_param)
        custom_table_fields_dct['intervention_param'] = intervention_param

        # Fair preprocessing
        cur_base_flow_dataset, cur_extra_test_sets =\
            remove_disparate_impact_with_mult_sets(init_base_flow_dataset, alpha=intervention_param,
                                                   init_extra_base_flow_datasets=extra_base_flow_datasets)

        # Tune model parameters if needed
        if with_tuning:
            # Tune models and create a models config for metrics computation
            tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, cur_base_flow_dataset,
                                                            metrics_computation_config.dataset_name,
                                                            n_folds=num_folds_for_tuning)

            # Create models_config from the saved tuned_params_df for higher reliability
            date_time_str = datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")
            os.makedirs(save_results_dir_path, exist_ok=True)
            tuned_df_path = os.path.join(save_results_dir_path,
                                         f'tuning_results_{metrics_computation_config.dataset_name}_alpha_{intervention_param}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_multiple_runs_with_multiple_test_sets(dataset=cur_base_flow_dataset,
                                                              extra_test_sets_lst=cur_extra_test_sets,
                                                              config=metrics_computation_config,
                                                              models_config=models_config,
                                                              custom_tbl_fields_dct=custom_table_fields_dct,
                                                              db_writer_func=db_writer_func,
                                                              verbose=0)

    logger.info("Experiment run was successful!")


def run_exp_iter_with_mult_set_and_preprocessing_intervention(data_loader, experiment_seed, test_set_fraction,
                                                              db_writer_func, fair_intervention_params_lst,
                                                              extra_test_sets: list,
                                                              column_transformer: ColumnTransformer, models_params_for_tuning,
                                                              metrics_computation_config, custom_table_fields_dct,
                                                              with_tuning: bool = False, save_results_dir_path: str = None,
                                                              tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                                                              verbose: bool = False):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Legacy, ignore for now
    metrics_computation_config.runs_seed_lst = [experiment_seed]

    # Preprocess the dataset using the defined preprocessor
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
    if verbose:
        logger.info("The dataset is preprocessed")
        print("Top indexes of an X_test in a base flow dataset: ", base_flow_dataset.X_test.index[:20])
        print("Top indexes of an y_test in a base flow dataset: ", base_flow_dataset.y_test.index[:20])

    for intervention_idx, intervention_param in tqdm(enumerate(fair_intervention_params_lst),
                                                     total=len(fair_intervention_params_lst),
                                                     desc="Multiple alphas",
                                                     colour="#40E0D0"):
        print('intervention_param: ', intervention_param)
        custom_table_fields_dct['intervention_param'] = intervention_param

        # Fair preprocessing
        cur_base_flow_dataset, preprocessed_extra_test_sets = \
            remove_correlation_for_mult_test_sets(base_flow_dataset, alpha=intervention_param, extra_test_sets=extra_test_sets)

        # Tune model parameters if needed
        if with_tuning:
            # Tune models and create a models config for metrics computation
            tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, cur_base_flow_dataset,
                                                            metrics_computation_config.dataset_name,
                                                            n_folds=num_folds_for_tuning)

            # Create models_config from the saved tuned_params_df for higher reliability
            date_time_str = datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")
            os.makedirs(save_results_dir_path, exist_ok=True)
            tuned_df_path = os.path.join(save_results_dir_path,
                                         f'tuning_results_{metrics_computation_config.dataset_name}_alpha_{intervention_param}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
#             print(f'{list(models_config.keys())[1]}: ', models_config[list(models_config.keys())[1]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_multiple_runs_with_multiple_test_sets(dataset=cur_base_flow_dataset,
                                                              extra_test_sets_lst=preprocessed_extra_test_sets,
                                                              config=metrics_computation_config,
                                                              models_config=models_config,
                                                              custom_tbl_fields_dct=custom_table_fields_dct,
                                                              db_writer_func=db_writer_func,
                                                              verbose=0)

    logger.info("Experiment run was successful!")
