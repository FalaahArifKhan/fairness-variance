import os
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from IPython.display import display

from configs.constants import ModelSetting
from configs.models_config_for_tuning import reset_model_seed
from source.utils.custom_initializers import create_base_pipeline
from source.custom_classes.base_dataset import BaseDataset
from source.analyzers.subgroups_variance_analyzer import SubgroupsVarianceAnalyzer
from source.utils.common_helpers import save_metrics_to_file
from source.analyzers.subgroups_statistical_bias_analyzer import SubgroupsStatisticalBiasAnalyzer


def compute_model_metrics_with_config(base_model, model_name, dataset, config, save_results_dir_path: str,
                                      model_seed: int = None, save_results=True, debug_mode=False) -> pd.DataFrame:
    if model_seed is None:
        model_seed = random.randint(1, 1000)

    return compute_model_metrics(base_model, config.n_estimators,
                                 dataset, config.test_set_fraction,
                                 config.bootstrap_fraction, config.sensitive_attributes_dct,
                                 model_seed=model_seed,
                                 dataset_name=config.dataset_name,
                                 base_model_name=model_name,
                                 save_results=save_results,
                                 save_results_dir_path=save_results_dir_path,
                                 debug_mode=debug_mode)


def compute_model_metrics(base_model, n_estimators, dataset, test_set_fraction: float, bootstrap_fraction: float,
                          sensitive_attributes_dct, model_seed, dataset_name, base_model_name,
                          save_results=True, save_results_dir_path=None, debug_mode=False):
    base_model = reset_model_seed(base_model, model_seed)
    print('Model random_state: ', base_model.get_params().get('random_state', None))

    base_pipeline = create_base_pipeline(dataset, sensitive_attributes_dct, model_seed, test_set_fraction)
    if debug_mode:
        print('\nProtected groups splits:')
        for g in base_pipeline.test_groups.keys():
            print(g, base_pipeline.test_groups[g].shape)

        print('\n\nTop rows of processed X train + validation set: ')
        display(base_pipeline.X_train_val.head(10))

    # Compute variance metrics for subgroups
    subgroups_variance_analyzer = SubgroupsVarianceAnalyzer(ModelSetting.BATCH, n_estimators, base_model, base_model_name,
                                                            bootstrap_fraction, base_pipeline, dataset_name)

    y_preds, variance_metrics_df = subgroups_variance_analyzer.compute_metrics(save_results=False,
                                                                               result_filename=None,
                                                                               save_dir_path=None,
                                                                               make_plots=False)

    # Compute bias metrics for subgroups
    bias_analyzer = SubgroupsStatisticalBiasAnalyzer(base_pipeline.X_test, base_pipeline.y_test,
                                                     base_pipeline.sensitive_attributes_dct, base_pipeline.test_groups)
    dtc_res = bias_analyzer.compute_subgroups_metrics(y_preds,
                                                      save_results=False,
                                                      result_filename=None,
                                                      save_dir_path=None)
    bias_metrics_df = pd.DataFrame(dtc_res)

    metrics_df = pd.concat([variance_metrics_df, bias_metrics_df])
    metrics_df = metrics_df.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Metric"})
    metrics_df['Model_Seed'] = model_seed

    if save_results:
        # Save metrics
        result_filename = f'Metrics_{dataset_name}_{base_model_name}'
        save_metrics_to_file(metrics_df, result_filename, save_results_dir_path)

    return metrics_df


def run_metrics_computation_with_config(dataset: BaseDataset, config, models_config: dict, save_results_dir_path: str,
                                        run_seed: int = None, debug_mode: bool = False) -> dict:
    """
    Find variance and statistical bias metrics for each model in models_config.
    Save results in `save_results_dir_path` folder.

    Returns a dictionary where keys are model names, and values are metrics for sensitive attributes defined in config.

    Parameters
    ----------
    dataset
        Dataset object that contains all needed attributes like target, features, numerical_columns etc
    config
        Object that contains test_set_fraction, bootstrap_fraction, dataset_name,
         n_estimators, sensitive_attributes_dct attributes
    models_config
        Dictionary where keys are model names, and values are initialized models
    save_results_dir_path
        Location where to save result files with metrics
    run_seed
        Base seed for this run
    debug_mode
        Enable or disable extra logs

    """
    if run_seed is None:
        run_seed = random.randint(1, 1000)
    # Create a directory for results if not exists
    os.makedirs(save_results_dir_path, exist_ok=True)
    # Parse config and execute the main run_metrics_computation function
    return run_metrics_computation(dataset, config.test_set_fraction, config.bootstrap_fraction,
                                   config.dataset_name, models_config, config.n_estimators,
                                   config.sensitive_attributes_dct,
                                   model_seed=run_seed,
                                   save_results_dir_path=save_results_dir_path,
                                   save_results=True,
                                   debug_mode=debug_mode)


def run_metrics_computation(dataset, test_set_fraction, bootstrap_fraction, dataset_name,
                            models_config, n_estimators, sensitive_attributes_dct, model_seed: int = None,
                            save_results=True, save_results_dir_path=None, debug_mode=False) -> dict:
    """
    Find variance and bias metrics for each model in config.MODELS_CONFIG.
    Save results in results/config.MODELS_CONFIG folder.

    :param exp_num: the number of experiment; is used to name the result file with metrics
    """
    models_metrics_dct = dict()
    num_models = len(models_config)
    for model_idx, model_name in tqdm(enumerate(models_config.keys()),
                                      total=num_models,
                                      desc="Analyze models in one run",
                                      colour="red"):
        print('#' * 30, f' [Model {model_idx + 1} / {num_models}] Analyze {model_name} ', '#' * 30)
        model_seed += 1
        try:
            base_model = models_config[model_name]
            model_metrics_df = compute_model_metrics(base_model, n_estimators, dataset, test_set_fraction,
                                                     bootstrap_fraction, sensitive_attributes_dct,
                                                     model_seed=model_seed,
                                                     dataset_name=dataset_name,
                                                     base_model_name=model_name,
                                                     save_results=save_results,
                                                     save_results_dir_path=save_results_dir_path,
                                                     debug_mode=debug_mode)
            model_metrics_df['Model_Name'] = model_name
            models_metrics_dct[f'Model_{model_idx + 1}_{model_name}'] = model_metrics_df
            if debug_mode:
                print(f'\n[{model_name}] Metrics matrix:')
                display(model_metrics_df)
        except Exception as err:
            print(f'ERROR with {model_name}: ', err)

        print('\n\n\n')

    return models_metrics_dct


def compute_metrics_multiple_runs(dataset, config, models_config, save_results_dir_path: str, debug_mode=False) -> dict:
    start_datetime = datetime.now(timezone.utc)
    os.makedirs(save_results_dir_path, exist_ok=True)

    multiple_runs_metrics_dct = dict()
    for run_num, run_seed in tqdm(enumerate(config.runs_seed_lst),
                                  total=len(config.runs_seed_lst),
                                  desc="Multiple runs progress",
                                  colour="green"):
        models_metrics_dct = run_metrics_computation(dataset, config.test_set_fraction, config.bootstrap_fraction,
                                                     config.dataset_name, models_config, config.n_estimators,
                                                     config.sensitive_attributes_dct, run_seed,
                                                     save_results=False, debug_mode=debug_mode)

        # Concatenate with previous results and save them in an overwrite mode each time for backups
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            model_metrics_df['Run_Number'] = f'Run_{run_num + 1}'

            if multiple_runs_metrics_dct.get(model_name) is None:
                multiple_runs_metrics_dct[model_name] = model_metrics_df
            else:
                multiple_runs_metrics_dct[model_name] = pd.concat([multiple_runs_metrics_dct[model_name], model_metrics_df])

            result_filename = f'Metrics_{config.dataset_name}_{model_name}_{config.n_estimators}_Estimators_{start_datetime.strftime("%Y%m%d__%H%M%S")}.csv'
            multiple_runs_metrics_dct[model_name].to_csv(f'{save_results_dir_path}/{result_filename}', index=False, mode='w')

    return multiple_runs_metrics_dct
