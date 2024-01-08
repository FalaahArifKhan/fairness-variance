import os
import copy
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow._api.v2.compat.v1 as tf
from pprint import pprint
from tqdm.notebook import tqdm
from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer
from IPython.display import display
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification
from aif360.metrics import ClassificationMetric

from virny.user_interfaces.multiple_models_with_db_writer_api import compute_metrics_with_db_writer
from virny.user_interfaces.multiple_models_with_multiple_test_sets_api import compute_metrics_with_multiple_test_sets
from virny.utils.custom_initializers import create_models_config_from_tuned_params_df
from virny.preprocessing.basic_preprocessing import preprocess_dataset

from source.custom_classes.adversarial_debiasing_wrapper import AdversarialDebiasingWrapper
from source.utils.model_tuning_utils import tune_ML_models
from source.utils.custom_logger import get_logger
from source.preprocessing import (remove_correlation, remove_correlation_for_mult_test_sets,
                                  remove_disparate_impact, get_preprocessor_for_diabetes, preprocess_mult_data_loaders_for_disp_imp,
                                  remove_disparate_impact_with_mult_sets, get_simple_preprocessor, apply_lfr)


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
        compute_metrics_with_db_writer(dataset=cur_base_flow_dataset,
                                       config=metrics_computation_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=db_writer_func,
                                       notebook_logs_stdout=True,
                                       verbose=0)

    logger.info("Experiment run was successful!")


def run_exp_iter_with_inprocessor(data_loader, experiment_seed, test_set_fraction, db_writer_func,
                                  fair_intervention_params_lst, metrics_computation_config, custom_table_fields_dct,
                                  verbose: bool = False, dataset_name: str = 'ACSIncomeDataset',
                                  inprocessor_name: str = 'AdversarialDebiasing'):
    tf.disable_eager_execution()
    # tf.disable_v2_behavior()

    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info("Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # ACS Income: Add SEX&RAC1P_binary column for LFR and remove 'SEX', 'RAC1P' to create a blind estimator.
    # Do similarly for other datasets.
    init_data_loader = copy.deepcopy(data_loader)
    sensitive_attrs_dct = metrics_computation_config.sensitive_attributes_dct
    if dataset_name in ('ACSIncomeDataset', 'ACSPublicCoverageDataset'):
        sensitive_attr_for_intervention = 'SEX&RAC1P_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['SEX'] == sensitive_attrs_dct['SEX'] and row['RAC1P'] in sensitive_attrs_dct['RAC1P']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

    elif dataset_name == 'StudentPerformancePortugueseDataset':
        sensitive_attr_for_intervention = 'sex_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col != 'sex']
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data['sex'].apply(lambda x: 1 if x == 'M' else 0)
        data_loader.full_df = data_loader.full_df.drop(['sex'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['sex'], axis=1)

    elif dataset_name == 'LawSchoolDataset':
        sensitive_attr_for_intervention = 'male&race_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('male', 'race')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['male'] == sensitive_attrs_dct['male'] and row['race'] == sensitive_attrs_dct['race']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['male', 'race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['male', 'race'], axis=1)

    else:
        raise ValueError('Incorrect dataset name')

    # Preprocess the dataset using the defined preprocessor
    column_transformer = get_simple_preprocessor(data_loader)
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
    base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
    # Align indexes of base_flow_dataset with data_loader for sensitive_attr_for_intervention column
    base_flow_dataset.X_train_val[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, sensitive_attr_for_intervention]
    base_flow_dataset.X_test[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_test.index, sensitive_attr_for_intervention]

    for intervention_idx, intervention_option in tqdm(enumerate(fair_intervention_params_lst),
                                                      total=len(fair_intervention_params_lst),
                                                      desc="Multiple alphas",
                                                      colour="#40E0D0"):
        print('intervention_option: ', intervention_option)
        custom_table_fields_dct['intervention_param'] = str(intervention_option)
        cur_base_flow_dataset = copy.deepcopy(base_flow_dataset)
        if verbose:
            logger.info("The dataset is preprocessed")
            print("cur_base_flow_dataset.X_train_val.columns: ", cur_base_flow_dataset.X_train_val.columns)
            print("Top indexes of an X_test in the current base flow dataset: ", cur_base_flow_dataset.X_test.index[:20])
            print("Top indexes of an y_test in the current base flow dataset: ", cur_base_flow_dataset.y_test.index[:20])

        # Define a postprocessor
        privileged_groups = [{sensitive_attr_for_intervention: 1}]
        unprivileged_groups = [{sensitive_attr_for_intervention: 0}]
        sess = tf.Session()
        if inprocessor_name == 'ExponentiatedGradientReduction':
            print('Using ExponentiatedGradientReduction postprocessor')
            debiased_model = None

        elif inprocessor_name == 'AdversarialDebiasing':
            print('Using AdversarialDebiasing postprocessor')
            debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                                  unprivileged_groups=unprivileged_groups,
                                                  scope_name=intervention_option,
                                                  debias=True,
                                                  num_epochs=200,
                                                  sess=sess)
        else:
            raise ValueError('inprocessor_name is unknown. Please, select one of the above defined options.')

        models_config = {
            inprocessor_name: AdversarialDebiasingWrapper(inprocessor=debiased_model,
                                                          sensitive_attr_for_intervention=sensitive_attr_for_intervention)
        }
        # Compute metrics for tuned models
        compute_metrics_with_db_writer(dataset=cur_base_flow_dataset,
                                       config=metrics_computation_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=db_writer_func,
                                       notebook_logs_stdout=True,
                                       verbose=0)

        sess.close()

    tf.reset_default_graph()
    logger.info("Experiment run was successful!")


def run_exp_iter_with_eq_odds(data_loader, experiment_seed, test_set_fraction, db_writer_func,
                              fair_intervention_params_lst, models_params_for_tuning,
                              metrics_computation_config, custom_table_fields_dct,
                              with_tuning: bool = False, save_results_dir_path: str = None,
                              tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                              verbose: bool = False, dataset_name: str = 'ACSIncomeDataset',
                              postprocessor_name: str = 'EqOddsPostprocessing'):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info("Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # ACS Income: Add SEX&RAC1P_binary column for LFR and remove 'SEX', 'RAC1P' to create a blind estimator.
    # Do similarly for other datasets.
    init_data_loader = copy.deepcopy(data_loader)
    sensitive_attrs_dct = metrics_computation_config.sensitive_attributes_dct
    sensitive_attr_for_intervention = metrics_computation_config.postprocessing_sensitive_attribute
    if dataset_name in ('ACSIncomeDataset', 'ACSPublicCoverageDataset'):
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['SEX'] == sensitive_attrs_dct['SEX'] and row['RAC1P'] in sensitive_attrs_dct['RAC1P']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

    elif dataset_name == 'StudentPerformancePortugueseDataset':
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col != 'sex']
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data['sex'].apply(lambda x: 1 if x == 'M' else 0)
        data_loader.full_df = data_loader.full_df.drop(['sex'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['sex'], axis=1)

    elif dataset_name == 'LawSchoolDataset':
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('male', 'race')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['male'] == sensitive_attrs_dct['male'] and row['race'] == sensitive_attrs_dct['race']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['male', 'race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['male', 'race'], axis=1)

    else:
        raise ValueError('Incorrect dataset name')

    # Preprocess the dataset using the defined preprocessor
    column_transformer = get_simple_preprocessor(data_loader)
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
    base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
    # Align indexes of base_flow_dataset with data_loader for sensitive_attr_for_intervention column
    base_flow_dataset.X_train_val[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, sensitive_attr_for_intervention]
    base_flow_dataset.X_test[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_test.index, sensitive_attr_for_intervention]

    for intervention_idx, intervention_option in tqdm(enumerate(fair_intervention_params_lst),
                                                      total=len(fair_intervention_params_lst),
                                                      desc="Multiple alphas",
                                                      colour="#40E0D0"):
        print('intervention_option: ', intervention_option)
        if intervention_option is not True:
            print('Skipping...')
            continue
        custom_table_fields_dct['intervention_param'] = str(intervention_option)
        cur_base_flow_dataset = copy.deepcopy(base_flow_dataset)

        # Define a postprocessor
        privileged_groups = [{sensitive_attr_for_intervention: 1}]
        unprivileged_groups = [{sensitive_attr_for_intervention: 0}]
        if postprocessor_name == 'ROC':
            print('Using ROC postprocessor')
            postprocessor = RejectOptionClassification(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                low_class_thresh=0.01, high_class_thresh=0.99,
                num_class_thresh=100, num_ROC_margin=50,
                metric_name='Statistical parity difference',
                metric_ub=0.05, metric_lb=-0.05
            )
        else:
            postprocessor = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                                 unprivileged_groups=unprivileged_groups,
                                                 seed=42)

        if verbose:
            logger.info("The dataset is preprocessed")
            print("cur_base_flow_dataset.X_train_val.columns: ", cur_base_flow_dataset.X_train_val.columns)
            print("Top indexes of an X_test in the current base flow dataset: ", cur_base_flow_dataset.X_test.index[:20])
            print("Top indexes of an y_test in the current base flow dataset: ", cur_base_flow_dataset.y_test.index[:20])

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
                                         f'tuning_results_{metrics_computation_config.dataset_name}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_with_db_writer(dataset=cur_base_flow_dataset,
                                       config=metrics_computation_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=db_writer_func,
                                       postprocessor=postprocessor,
                                       notebook_logs_stdout=True,
                                       verbose=0)

    logger.info("Experiment run was successful!")


def run_exp_iter_with_LFR(data_loader, experiment_seed, test_set_fraction, db_writer_func,
                          fair_intervention_params_lst, models_params_for_tuning,
                          metrics_computation_config, custom_table_fields_dct,
                          with_tuning: bool = False, save_results_dir_path: str = None,
                          tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                          verbose: bool = False, dataset_name: str = 'ACSIncomeDataset'):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info("Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # ACS Income: Add SEX&RAC1P_binary column for LFR and remove 'SEX', 'RAC1P' to create a blind estimator.
    # Do similarly for other datasets.
    init_data_loader = copy.deepcopy(data_loader)
    sensitive_attrs_dct = metrics_computation_config.sensitive_attributes_dct
    if dataset_name in ('ACSIncomeDataset', 'ACSPublicCoverageDataset'):
        sensitive_attr_for_intervention = 'SEX&RAC1P_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['SEX'] == sensitive_attrs_dct['SEX'] and row['RAC1P'] in sensitive_attrs_dct['RAC1P']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

    elif dataset_name == 'StudentPerformancePortugueseDataset':
        sensitive_attr_for_intervention = 'sex_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col != 'sex']
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data['sex'].apply(lambda x: 1 if x == 'M' else 0)
        data_loader.full_df = data_loader.full_df.drop(['sex'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['sex'], axis=1)

    elif dataset_name == 'LawSchoolDataset':
        sensitive_attr_for_intervention = 'male&race_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('male', 'race')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data['race'].apply(lambda x: 1 if x == 'White' else 0)
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['male'] == sensitive_attrs_dct['male'] and row['race'] == sensitive_attrs_dct['race']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['male', 'race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['male', 'race'], axis=1)

    else:
        raise ValueError('Incorrect dataset name')

    # Preprocess the dataset using the defined preprocessor
    column_transformer = get_simple_preprocessor(data_loader)
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
    base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
    # Align indexes of base_flow_dataset with data_loader for sensitive_attr_for_intervention column
    base_flow_dataset.X_train_val[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, sensitive_attr_for_intervention]
    base_flow_dataset.X_test[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_test.index, sensitive_attr_for_intervention]

    for intervention_idx, intervention_options in tqdm(enumerate(fair_intervention_params_lst),
                                                       total=len(fair_intervention_params_lst),
                                                       desc="Multiple alphas",
                                                       colour="#40E0D0"):
        print('intervention_options: ', intervention_options)
        custom_table_fields_dct['intervention_param'] = str(intervention_options)

        # Fair preprocessing
        cur_base_flow_dataset = apply_lfr(base_flow_dataset,
                                          intervention_options=intervention_options,
                                          sensitive_attribute=sensitive_attr_for_intervention)
        if verbose:
            logger.info("The dataset is preprocessed")
            print("cur_base_flow_dataset.X_train_val.columns: ", cur_base_flow_dataset.X_train_val.columns)
            print("Top indexes of an X_test in the current base flow dataset: ", cur_base_flow_dataset.X_test.index[:20])
            print("Top indexes of an y_test in the current base flow dataset: ", cur_base_flow_dataset.y_test.index[:20])

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
                                         f'tuning_results_{metrics_computation_config.dataset_name}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_with_db_writer(dataset=cur_base_flow_dataset,
                                       config=metrics_computation_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=db_writer_func,
                                       notebook_logs_stdout=True,
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
    logger.info("Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    # Add SEX&RAC1P_binary column for DisparateImpactRemover and remove 'SEX', 'RAC1P' to create a blind estimator
    init_data_loader = copy.deepcopy(data_loader)
    sensitive_attr_for_intervention = None
    sensitive_attrs_dct = metrics_computation_config.sensitive_attributes_dct
    if dataset_name in ('ACSIncomeDataset', 'ACSPublicCoverageDataset'):
        sensitive_attr_for_intervention = 'SEX&RAC1P_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['SEX'] == sensitive_attrs_dct['SEX'] and row['RAC1P'] in sensitive_attrs_dct['RAC1P']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

    elif dataset_name == 'LawSchoolDataset':
        sensitive_attr_for_intervention = 'male&race_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('male', 'race')]
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data.apply(
            lambda row: 0 if (row['male'] == sensitive_attrs_dct['male'] and row['race'] == sensitive_attrs_dct['race']) else 1,
            axis=1
        )
        data_loader.full_df = data_loader.full_df.drop(['male', 'race'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['male', 'race'], axis=1)

    elif dataset_name == 'StudentPerformancePortugueseDataset':
        sensitive_attr_for_intervention = 'sex_binary'
        data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col != 'sex']
        data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data['sex'].apply(lambda x: 1 if x == 'M' else 0)
        data_loader.full_df = data_loader.full_df.drop(['sex'], axis=1)
        data_loader.X_data = data_loader.X_data.drop(['sex'], axis=1)

    # Preprocess the dataset using the defined preprocessor
    column_transformer = get_simple_preprocessor(data_loader)
    base_flow_dataset = preprocess_dataset(data_loader, column_transformer, test_set_fraction, experiment_seed)
    base_flow_dataset.init_features_df = init_data_loader.full_df.drop(init_data_loader.target, axis=1, errors='ignore')
    # Align indexes of base_flow_dataset with data_loader for sensitive_attr_for_intervention column
    base_flow_dataset.X_train_val[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_train_val.index, sensitive_attr_for_intervention]
    base_flow_dataset.X_test[sensitive_attr_for_intervention] = data_loader.X_data.loc[base_flow_dataset.X_test.index, sensitive_attr_for_intervention]

    for intervention_idx, intervention_param in tqdm(enumerate(fair_intervention_params_lst),
                                                     total=len(fair_intervention_params_lst),
                                                     desc="Multiple alphas",
                                                     colour="#40E0D0"):
        print('intervention_param: ', intervention_param)
        custom_table_fields_dct['intervention_param'] = intervention_param

        # Fair preprocessing
        cur_base_flow_dataset = remove_disparate_impact(base_flow_dataset,
                                                        alpha=intervention_param,
                                                        sensitive_attribute=sensitive_attr_for_intervention)
        if verbose:
            logger.info("The dataset is preprocessed")
            print("cur_base_flow_dataset.X_train_val.columns: ", cur_base_flow_dataset.X_train_val.columns)
            print("Top indexes of an X_test in the current base flow dataset: ", cur_base_flow_dataset.X_test.index[:20])
            print("Top indexes of an y_test in the current base flow dataset: ", cur_base_flow_dataset.y_test.index[:20])

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
        compute_metrics_with_db_writer(dataset=cur_base_flow_dataset,
                                       config=metrics_computation_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=db_writer_func,
                                       notebook_logs_stdout=True,
                                       verbose=0)

    logger.info("Experiment run was successful!")


def run_exp_iter_with_disparate_impact_and_mult_sets(data_loader, extra_data_loaders, experiment_seed, test_set_fraction,
                                                     db_writer_func, fair_intervention_params_lst, models_params_for_tuning,
                                                     metrics_computation_config, custom_table_fields_dct,
                                                     with_tuning: bool = False, save_results_dir_path: str = None,
                                                     tuned_params_df_paths: list = None, num_folds_for_tuning: int = 3,
                                                     train_set_subsample_size: int = None,
                                                     verbose: bool = False):
    custom_table_fields_dct['dataset_split_seed'] = experiment_seed
    custom_table_fields_dct['model_init_seed'] = experiment_seed
    custom_table_fields_dct['fair_intervention_params_lst'] = str(fair_intervention_params_lst)

    logger = get_logger()
    logger.info("Start an experiment iteration for the following custom params:")
    pprint(custom_table_fields_dct)
    print('\n', flush=True)

    logger.info("Start dataset preprocessing")
    init_base_flow_dataset, extra_base_flow_datasets = \
        preprocess_mult_data_loaders_for_disp_imp(main_data_loader=data_loader,
                                                  extra_data_loaders=extra_data_loaders,
                                                  test_set_fraction=test_set_fraction,
                                                  experiment_seed=experiment_seed,
                                                  train_set_subsample_size=train_set_subsample_size,
                                                  metrics_computation_config=metrics_computation_config)
    logger.info("The dataset is preprocessed")

    for intervention_idx, intervention_param in tqdm(enumerate(fair_intervention_params_lst),
                                                     total=len(fair_intervention_params_lst),
                                                     desc="Multiple alphas",
                                                     colour="#40E0D0"):
        print('intervention_param: ', intervention_param)
        custom_table_fields_dct['intervention_param'] = intervention_param

        # Fair preprocessing
        logger.info("Start fairness intervention")
        cur_base_flow_dataset, cur_extra_test_sets =\
            remove_disparate_impact_with_mult_sets(init_base_flow_dataset,
                                                   alpha=intervention_param,
                                                   init_extra_base_flow_datasets=extra_base_flow_datasets)
        logger.info("Fairness intervention is completed")
        if verbose:
            print('Number of rows in the in-domain X_test', cur_base_flow_dataset.X_test.shape[0], flush=True)
            print("Top indexes of an X_test in an in-domain base flow dataset: ", cur_base_flow_dataset.X_test.index[:20], flush=True)
            print("Top indexes of an y_test in an in-domain base flow dataset: ", cur_base_flow_dataset.y_test.index[:20], flush=True)
            display(cur_base_flow_dataset.X_test.head())
            print('\n\n', flush=True)
            print('Number of rows in the out-of-domain X_test', cur_extra_test_sets[0][0].shape[0], flush=True)
            print("Top indexes of an X_test in an out-of-domain base flow dataset: ", cur_extra_test_sets[0][0].index[:20], flush=True)
            print("Top indexes of an y_test in an out-of-domain base flow dataset: ", cur_extra_test_sets[0][1].index[:20], flush=True)
            display(cur_extra_test_sets[0][0].head())

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
                                         f'tuning_results_{metrics_computation_config.dataset_name}_alpha_{intervention_param}_{custom_table_fields_dct["experiment_iteration"].lower()}_{date_time_str}.csv')
            tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
            logger.info("Models are tuned and saved to a file")
        else:
            print('Path for tuned params: ', tuned_params_df_paths[intervention_idx])
            models_config = create_models_config_from_tuned_params_df(models_params_for_tuning, tuned_params_df_paths[intervention_idx])
            print(f'{list(models_config.keys())[0]}: ', models_config[list(models_config.keys())[0]].get_params())
            logger.info("Models config is loaded from the input file")

        # Compute metrics for tuned models
        compute_metrics_with_multiple_test_sets(dataset=cur_base_flow_dataset,
                                                extra_test_sets_lst=cur_extra_test_sets,
                                                config=metrics_computation_config,
                                                models_config=models_config,
                                                custom_tbl_fields_dct=custom_table_fields_dct,
                                                db_writer_func=db_writer_func,
                                                notebook_logs_stdout=True,
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
        compute_metrics_with_multiple_test_sets(dataset=cur_base_flow_dataset,
                                                extra_test_sets_lst=preprocessed_extra_test_sets,
                                                config=metrics_computation_config,
                                                models_config=models_config,
                                                custom_tbl_fields_dct=custom_table_fields_dct,
                                                db_writer_func=db_writer_func,
                                                notebook_logs_stdout=True,
                                                verbose=0)

    logger.info("Experiment run was successful!")
