import os
import sys
from pathlib import Path

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent.parent))
print('Current location: ', os.getcwd())

# Import dependencies
import uuid
import ast
import copy
import argparse
import warnings
from datetime import datetime
from dotenv import load_dotenv
from IPython.display import display

from virny.utils.custom_initializers import create_config_obj
from virny.datasets import ACSIncomeDataset

from configs.constants import TEST_SET_FRACTION, EXPERIMENT_SEEDS
from configs.models_config_for_tuning import get_model_params_for_mult_repair_levels, get_model_params_for_mult_repair_levels_dummy
from source.utils.db_functions import connect_to_mongodb
from source.experiment_interface import run_exp_iter_with_eqq_odds_postprocessing


# Define configurable variables
EXPERIMENT_NAME = 'bootstrap_sizes_experiment_income_GA'
DB_COLLECTION_NAME = 'bootstrap_sizes_experiment'
CUSTOM_TABLE_FIELDS_DCT = {
    #'session_uuid': str(uuid.uuid4())
    'session_uuid': 'test'
}

# Define input variables
ROOT_DIR = os.getcwd()
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'scripts', 'results', EXPERIMENT_NAME)
os.makedirs(SAVE_RESULTS_DIR_PATH, exist_ok=True)
config_yaml_path = os.path.join(ROOT_DIR, 'scripts', EXPERIMENT_NAME, 'folk_ga_2018_config.yaml')
METRICS_COMPUTATION_CONFIG = create_config_obj(config_yaml_path=config_yaml_path)
CLIENT, COLLECTION_OBJ, DB_WRITER_FUNC = connect_to_mongodb(DB_COLLECTION_NAME)


def preconfigurate_experiment(env_file_path='./configs/secrets.env'):
    warnings.filterwarnings('ignore')
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Load env variables
    load_dotenv(env_file_path)
    print('\n\nDB_NAME:', os.getenv("DB_NAME"))


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nums", type=str, help="a list of experiment run numbers", required=True)
    parser.add_argument("--bootstrap_fraction", type=float, help="fraction of dataset to use during bootstrap", required=True)
    parser.add_argument("--fairness_intervention_params", type=str,
                       help="a list of fairness intervention params", default='[]')
    parser.add_argument("--tuned_params_filenames", type=str,
                       help="a list of filenames with tuned model hyper-parameters", default='[]')
    args = parser.parse_args()

    run_nums = ast.literal_eval(args.run_nums)
    bootstrap_fraction = args.bootstrap_fraction
    fairness_intervention_params = ast.literal_eval(args.fairness_intervention_params)
    tuned_params_filenames = ast.literal_eval(args.tuned_params_filenames)

    print(
        f"Experiment name: {EXPERIMENT_NAME}\n"
        f"Current session uuid: {CUSTOM_TABLE_FIELDS_DCT['session_uuid']}\n"
        f"Experiment run numbers: {run_nums}\n"
        f"Bootstrap fraction: {bootstrap_fraction}"
        f"Fairness intervention params: {fairness_intervention_params}\n"
        f"Tuned params filenames: {tuned_params_filenames}\n"
    )

    return run_nums, bootstrap_fraction, fairness_intervention_params, tuned_params_filenames


def run_experiment(exp_run_num, data_loader):
    # Configs for an experiment iteration
    experiment_seed = EXPERIMENT_SEEDS[exp_run_num - 1]
    CUSTOM_TABLE_FIELDS_DCT['experiment_iteration'] = f'Exp_iter_{exp_run_num}'

    exp_iter_data_loader = copy.deepcopy(data_loader)  # Add deepcopy to avoid data leakage
    models_params_for_tuning = get_model_params_for_mult_repair_levels(experiment_seed)
    #models_params_for_tuning = get_model_params_for_mult_repair_levels_dummy(experiment_seed)

    tuned_params_df_paths = [os.path.join(SAVE_RESULTS_DIR_PATH, p) for p in os.listdir(SAVE_RESULTS_DIR_PATH)]
    
    if (exp_run_num == 1) and (len(tuned_params_df_paths) == 0):
        with_tuning = True
        tuned_params_df_path = None
        print('Enable hyper-params tuning')
    else:
        with_tuning = False
        
        assert len(tuned_params_df_paths) == 1, "Len of tuned_params_df_paths should be 1!"
        tuned_params_df_path = tuned_params_df_paths[0]

    run_exp_iter_with_eqq_odds_postprocessing(data_loader=exp_iter_data_loader,
                                              experiment_seed=experiment_seed,
                                              test_set_fraction=TEST_SET_FRACTION,
                                              db_writer_func=DB_WRITER_FUNC,
                                              models_params_for_tuning=models_params_for_tuning,
                                              metrics_computation_config=METRICS_COMPUTATION_CONFIG,
                                              custom_table_fields_dct=CUSTOM_TABLE_FIELDS_DCT,
                                              with_tuning=with_tuning,
                                              tuned_params_df_path=tuned_params_df_path,
                                              save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                              verbose=True,
                                              dataset_name='ACSIncomeDataset')
    
    del exp_iter_data_loader


if __name__ == '__main__':
    start_time = datetime.now()
    preconfigurate_experiment()
    run_nums, bootstrap_fraction, fairness_intervention_params, tuned_params_filenames = parse_input_args()
    METRICS_COMPUTATION_CONFIG.bootstrap_fraction = bootstrap_fraction

    # Initialize custom objects
    data_loader = ACSIncomeDataset(
                        state=['GA'], year=2018, with_nulls=False, 
                        subsample_size=15_000, subsample_seed=42
                        )
    
    print('data_loader_rich.X_data.shape:', data_loader.X_data.shape)
    print('data_loader_rich.X_data.head()')
    display(data_loader.X_data.head())

    # Execute each experiment run
    for run_num in run_nums:
        print(f"{'#'*40} Experiment iteration {run_num} {'#'*40}", flush=True)
        run_experiment(exp_run_num=run_num,
                       data_loader=data_loader
                       )
        print('\n\n\n', flush=True)

    end_time = datetime.now()
    print(f'The script is successfully executed. Run time: {end_time - start_time}\n'
          f'Experiment uuid: {CUSTOM_TABLE_FIELDS_DCT["session_uuid"]}')
    

    CLIENT.close()
