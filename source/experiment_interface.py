import os
from datetime import datetime, timezone

from sklearn.compose import ColumnTransformer

from virny.user_interfaces.metrics_computation_interfaces import compute_metrics_multiple_runs_with_db_writer
from virny.utils.custom_initializers import create_models_config_from_tuned_params_df
from virny.preprocessing.basic_preprocessing import preprocess_dataset
from virny.utils.model_tuning_utils import tune_ML_models

from configs.constants import TEST_SET_FRACTION
from source.db_functions import connect_to_mongodb


def run_experiment(data_loader, experiment_seed, preprocessor: ColumnTransformer, models_params_for_tuning,
                   metrics_computation_config, custom_table_fields_dct,
                   with_tuning: bool = False, save_results_dir_path: str = None, tuned_params_df_path: str = None):
    # Preprocess the dataset using the defined preprocessor
    base_flow_dataset = preprocess_dataset(data_loader, preprocessor, TEST_SET_FRACTION, experiment_seed)

    # Tune model parameters if needed
    if with_tuning:
        # Tune models and create a models config for metrics computation
        tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset,
                                                        metrics_computation_config.dataset_name, n_folds=3)

        # Create models_config from the saved tuned_params_df for higher reliability
        now = datetime.now(timezone.utc)
        date_time_str = now.strftime("%Y%m%d__%H%M%S")
        tuned_df_path = os.path.join(save_results_dir_path, 'models_tuning',
                                     f'tuning_results_{metrics_computation_config.dataset_name}_{date_time_str}.csv')
        tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
    else:
        models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_path)

    # Compute metrics for tuned models
    client, collection, db_writer_func = connect_to_mongodb()
    multiple_run_metrics_dct = compute_metrics_multiple_runs_with_db_writer(base_flow_dataset, metrics_computation_config, models_config,
                                                                            custom_table_fields_dct, db_writer_func, debug_mode=False)

    return multiple_run_metrics_dct
