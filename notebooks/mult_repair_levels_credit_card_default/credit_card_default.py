import os
import warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

cur_folder_name = os.getcwd().split('\\')[-1]
if cur_folder_name != "fairness-variance":
    os.chdir("../..")

print('Current location: ', os.getcwd())

import copy
from dotenv import load_dotenv

load_dotenv('./configs/secrets.env')

from virny.utils.custom_initializers import create_config_obj
from virny.datasets import CreditCardDefaultDataset

from configs.constants import TEST_SET_FRACTION, EXPERIMENT_SEEDS
from configs.models_config_for_tuning import get_model_params_for_mult_repair_levels, get_dummy_model_params

from source.experiment_interface import run_exp_iter_with_disparate_impact

ROOT_DIR = os.getcwd()
EXPERIMENT_NAME = 'mult_repair_levels_credit_card_default'
DB_COLLECTION_NAME = 'exp_mult_repair_levels'
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME)
FAIR_INTERVENTION_PARAMS_LST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#FAIR_INTERVENTION_PARAMS_LST = [0.0, 0.2]

config_yaml_path = os.path.join(ROOT_DIR, 'notebooks', EXPERIMENT_NAME, 'credit_card_default_config.yaml')
metrics_computation_config = create_config_obj(config_yaml_path=config_yaml_path)

print(metrics_computation_config)

from source.utils.db_functions import connect_to_mongodb

client, collection_obj, db_writer_func = connect_to_mongodb(DB_COLLECTION_NAME)

import uuid

custom_table_fields_dct = {
    #'session_uuid': str(uuid.uuid4()),
    'session_uuid': '989f86af-ac68-4c31-972b-dad9cb5d6887',
}
print('Current session uuid: ', custom_table_fields_dct['session_uuid'])

data_loader = CreditCardDefaultDataset()
data_loader.X_data.head()

tuned_params_dir = os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME)
tuned_params_filenames = sorted([p for p in os.listdir(tuned_params_dir) if 'tuning_results' in p])
tuned_params_df_paths = [os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME, tuned_params_filename)
                         for tuned_params_filename in tuned_params_filenames]


# Configs for an experiment iteration
exp_iter_num = 2
experiment_seed = EXPERIMENT_SEEDS[exp_iter_num - 1]
custom_table_fields_dct['experiment_iteration'] = f'Exp_iter_{exp_iter_num}'

exp_iter_data_loader = copy.deepcopy(data_loader)  # Add deepcopy to avoid data leakage

run_exp_iter_with_disparate_impact(data_loader=exp_iter_data_loader,
                                   experiment_seed=experiment_seed,
                                   test_set_fraction=TEST_SET_FRACTION,
                                   db_writer_func=db_writer_func,
                                   fair_intervention_params_lst=FAIR_INTERVENTION_PARAMS_LST,
                                   models_params_for_tuning=models_params_for_tuning,
                                   metrics_computation_config=metrics_computation_config,
                                   custom_table_fields_dct=custom_table_fields_dct,
                                   with_tuning=False,
                                   tuned_params_df_paths=tuned_params_df_paths,
                                   save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                   verbose=True,
                                   dataset_name="CreditCardDefaultDataset")