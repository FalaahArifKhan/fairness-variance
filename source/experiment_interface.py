import os
from pprint import pprint
from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer

from virny.user_interfaces.metrics_computation_interfaces import compute_metrics_multiple_runs_with_db_writer
from virny.utils.custom_initializers import create_models_config_from_tuned_params_df
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.utils.model_tuning_utils import tune_ML_models

from configs.constants import TEST_SET_FRACTION, NUM_TUNING_FOLDS
from source.custom_logger import get_logger
from source.db_functions import connect_to_mongodb


def run_exp_iteration(data_loader, experiment_seed, db_collection_name, preprocessor: ColumnTransformer,
                      models_params_for_tuning, metrics_computation_config, custom_table_fields_dct,
                      with_tuning: bool = False, save_results_dir_path: str = None, tuned_params_df_path: str = None):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed

    logger = get_logger()
    logger.info(f"Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n')

    # Set seeds for metrics computation
    metrics_computation_config.runs_seed_lst = [experiment_seed + i for i in range(1, metrics_computation_config.num_runs + 1)]

    # Preprocess the dataset using the defined preprocessor
    base_flow_dataset = preprocess_dataset(data_loader, preprocessor, TEST_SET_FRACTION, experiment_seed)
    logger.info("The dataset is preprocessed")

    # Tune model parameters if needed
    if with_tuning:
        # Tune models and create a models config for metrics computation
        tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset,
                                                        metrics_computation_config.dataset_name, n_folds=NUM_TUNING_FOLDS)

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
    client, collection, db_writer_func = connect_to_mongodb(db_collection_name)
    logger.info("Connected to MongoDB")
    multiple_run_metrics_dct = compute_metrics_multiple_runs_with_db_writer(base_flow_dataset, metrics_computation_config, models_config,
                                                                            custom_table_fields_dct, db_writer_func, verbose=0)
    logger.info("Metrics are computed")
    client.close()

    return multiple_run_metrics_dct
