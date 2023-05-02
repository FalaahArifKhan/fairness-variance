import os
from pprint import pprint
from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer

from virny.user_interfaces.metrics_computation_interfaces import (
    compute_metrics_multiple_runs_with_db_writer,
    compute_metrics_multiple_runs_with_multiple_test_sets
)
from virny.utils.custom_initializers import create_models_config_from_tuned_params_df
from virny.preprocessing.basic_preprocessing import preprocess_dataset

from source.preprocessing.basic_preprocessing import preprocess_experiment_dataset, create_stress_testing_sets, \
    create_stress_testing_sets_using_columns, create_stress_testing_sets_using_cols_importance
from source.utils.model_tuning_utils import tune_ML_models
from source.custom_classes.custom_logger import get_logger


def run_exp_iteration(data_loader, experiment_seed, test_set_fraction, db_writer_func,
                      preprocessor: ColumnTransformer, models_params_for_tuning,
                      metrics_computation_config, custom_table_fields_dct,
                      with_tuning: bool = False, save_results_dir_path: str = None,
                      tuned_params_df_path: str = None, num_folds_for_tuning: int = 3,
                      verbose: bool = False):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Set seeds for metrics computation
    metrics_computation_config.runs_seed_lst = [experiment_seed + i for i in range(1, metrics_computation_config.num_runs + 1)]

    # Preprocess the dataset using the defined preprocessor
    base_flow_dataset = preprocess_dataset(data_loader, preprocessor, test_set_fraction, experiment_seed)
    if verbose:
        logger.info("The dataset is preprocessed")
        print("Top indexes of an X_test in a base flow dataset: ", base_flow_dataset.X_test.index[:20])
        print("Top indexes of an y_test in a base flow dataset: ", base_flow_dataset.y_test.index[:20])

    # Tune model parameters if needed
    if with_tuning:
        # Tune models and create a models config for metrics computation
        tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset,
                                                        metrics_computation_config.dataset_name, n_folds=num_folds_for_tuning)

        # Create models_config from the saved tuned_params_df for higher reliability
        date_time_str = datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")
        models_tuning_results_dir = os.path.join(save_results_dir_path, 'models_tuning')
        os.makedirs(models_tuning_results_dir, exist_ok=True)
        tuned_df_path = os.path.join(models_tuning_results_dir, f'tuning_results_{metrics_computation_config.dataset_name}_{date_time_str}.csv')
        tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
        logger.info("Models are tuned and saved to a file")
    else:
        models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_path)
        logger.info("Models config is loaded from the input file")

    # Compute metrics for tuned models
    multiple_run_metrics_dct = compute_metrics_multiple_runs_with_db_writer(base_flow_dataset, metrics_computation_config, models_config,
                                                                            custom_table_fields_dct, db_writer_func, verbose=0)
    logger.info("Metrics are computed")

    return multiple_run_metrics_dct


def run_exp_iter_with_models_stress_testing(data_loader, experiment_seed, test_set_fraction, db_writer_func,
                                            error_injector, injector_config_lst,
                                            preprocessor: ColumnTransformer, models_params_for_tuning,
                                            metrics_computation_config, custom_table_fields_dct,
                                            with_tuning: bool = False, save_results_dir_path: str = None,
                                            tuned_params_df_path: str = None, num_folds_for_tuning: int = 3,
                                            mode='rows_pct',
                                            verbose: bool = False):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['injector_config_lst'] = str(injector_config_lst)

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Set seeds for metrics computation
    metrics_computation_config.runs_seed_lst = [experiment_seed + i for i in range(1, metrics_computation_config.num_runs + 1)]

    # Preprocess the dataset using the defined preprocessor
    base_flow_dataset, train_test_sets, fitted_column_transformer = \
        preprocess_experiment_dataset(data_loader, preprocessor, test_set_fraction, experiment_seed)
    if verbose:
        logger.info("The dataset is preprocessed")
        print("Top indexes of an X_test in a base flow dataset: ", base_flow_dataset.X_test.index[:20])
        print("Top indexes of an y_test in a base flow dataset: ", base_flow_dataset.y_test.index[:20])

    # Create extra stress testing sets
    original_X_train_val, original_X_test, original_y_train_val, original_y_test = train_test_sets
    if mode == 'max_num_columns':
        print('Creating test sets based on max_num_columns...')
        extra_test_sets_lst = create_stress_testing_sets_using_columns(original_X_test, original_y_test,
                                                                       error_injector, injector_config_lst,
                                                                       fitted_column_transformer)
    elif mode == 'column_importance':
        print('Creating test sets based on column_importance...')
        extra_test_sets_lst = create_stress_testing_sets_using_cols_importance(original_X_test, original_y_test,
                                                                               error_injector, injector_config_lst,
                                                                               fitted_column_transformer)
    else:
        extra_test_sets_lst = create_stress_testing_sets(original_X_test, original_y_test,
                                                         error_injector, injector_config_lst,
                                                         fitted_column_transformer)

    # Tune model parameters if needed
    if with_tuning:
        # Tune models and create a models config for metrics computation
        tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset,
                                                        metrics_computation_config.dataset_name,
                                                        n_folds=num_folds_for_tuning)

        # Create models_config from the saved tuned_params_df for higher reliability
        date_time_str = datetime.now(timezone.utc).strftime("%Y%m%d__%H%M%S")
        models_tuning_results_dir = os.path.join(save_results_dir_path, 'models_tuning')
        os.makedirs(models_tuning_results_dir, exist_ok=True)
        tuned_df_path = os.path.join(models_tuning_results_dir,
                                     f'tuning_results_{metrics_computation_config.dataset_name}_{custom_table_fields_dct["experiment_iteration"].lower()}_{date_time_str}.csv')
        tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
        logger.info("Models are tuned and saved to a file")
    else:
        models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_path)
        print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
        # print(f'{list(models_config.keys())[1]}: ', models_config[list(models_config.keys())[1]].get_params())
        logger.info("Models config is loaded from the input file")

    # Compute metrics for tuned models
    compute_metrics_multiple_runs_with_multiple_test_sets(dataset=base_flow_dataset,
                                                          extra_test_sets_lst=extra_test_sets_lst,
                                                          config=metrics_computation_config,
                                                          models_config=models_config,
                                                          custom_tbl_fields_dct=custom_table_fields_dct,
                                                          db_writer_func=db_writer_func,
                                                          verbose=0)
    logger.info("Experiment run was successful!")
