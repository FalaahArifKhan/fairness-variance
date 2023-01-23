import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from IPython.display import display

from configs.constants import ModelSetting
from utils.custom_initializers import create_base_pipeline, create_tuned_base_model
from utils.analyzers.subgroups_variance_analyzer import SubgroupsVarianceAnalyzer
from utils.common_helpers import save_metrics_to_file
from utils.analyzers.subgroups_statistical_bias_analyzer import SubgroupsStatisticalBiasAnalyzer


# TODO: create MetricsCalculator
def compute_model_metrics(base_model, n_estimators, dataset, test_set_fraction, sensitive_attributes, priv_values,
                          model_seed, dataset_name, base_model_name,
                          save_results=True, save_results_dir_path=None, debug_mode=False):
    base_pipeline = create_base_pipeline(dataset, sensitive_attributes, priv_values, model_seed, test_set_fraction)
    if debug_mode:
        print('\nProtected groups splits:')
        for g in base_pipeline.test_groups.keys():
            print(g, base_pipeline.test_groups[g].shape)

        print('\n\nX train and validation set: ')
        display(base_pipeline.X_train_val.head(10))

    # Compute variance metrics for subgroups
    stability_fairness_analyzer = SubgroupsVarianceAnalyzer(ModelSetting.BATCH, n_estimators, base_model, base_model_name,
                                                            base_pipeline.X_train_val, base_pipeline.y_train_val,
                                                            base_pipeline.X_test, base_pipeline.y_test,
                                                            base_pipeline.sensitive_attributes, base_pipeline.priv_values,
                                                            base_pipeline.test_groups,
                                                            base_pipeline.target, dataset_name)

    y_preds, variance_metrics_df = stability_fairness_analyzer.compute_metrics(save_results=save_results,
                                                                               result_filename=None,
                                                                               save_dir_path=None,
                                                                               make_plots=False)

    # Compute bias metrics for subgroups
    bias_analyzer = SubgroupsStatisticalBiasAnalyzer(base_pipeline.X_test, base_pipeline.y_test,
                                                     base_pipeline.sensitive_attributes, base_pipeline.priv_values,
                                                     base_pipeline.test_groups)
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


def run_metrics_computation(dataset, test_set_fraction, dataset_name, model_seed: int,
                            config, models_tuned_params_df, n_estimators, sensitive_attributes, priv_values,
                            save_results=True, save_results_dir_path=None, debug_mode=False) -> dict:
    """
    Find variance and bias metrics for each model in config.MODELS_CONFIG.
    Save results in results/config.MODELS_CONFIG folder.

    :param exp_num: the number of experiment; is used to name the result file with metrics
    """
    models_metrics_dct = dict()
    num_models = len(config.MODELS_CONFIG)
    for model_idx in tqdm(range(len(config.MODELS_CONFIG))):
        model_name = config.MODELS_CONFIG[model_idx]["model_name"]
        print('#' * 30, f' [Model {model_idx + 1} / {num_models}] Analyze {model_name} ', '#' * 30)
        model_seed += 1
        try:
            base_model = create_tuned_base_model(config.MODELS_CONFIG[model_idx]['model'],
                                                 model_name,
                                                 models_tuned_params_df)
            model_metrics_df = compute_model_metrics(base_model, n_estimators, dataset, test_set_fraction,
                                                     sensitive_attributes, priv_values,
                                                     model_seed=model_seed,
                                                     dataset_name=dataset_name,
                                                     base_model_name=model_name,
                                                     save_results=save_results,
                                                     save_results_dir_path=save_results_dir_path,
                                                     debug_mode=debug_mode)
            model_metrics_df['Model_Name'] = model_name
            models_metrics_dct[f'Model_{model_idx + 1}_{model_name}'] = model_metrics_df
            if debug_mode:
                print(f'\n[{model_name}] Metrics confusion matrix:')
                display(model_metrics_df)
        except Exception as err:
            print(f'ERROR with {model_name}: ', err)

        print('\n\n\n')

    return models_metrics_dct


def compute_metrics_multiple_runs(dataset, test_set_fraction, dataset_name,
                                  config, models_tuned_params_df, n_estimators, sensitive_attributes, priv_values,
                                  runs_seed_lst: list, save_results_dir_path: str, debug_mode=False):
    start_datetime = datetime.now(timezone.utc)
    os.makedirs(save_results_dir_path, exist_ok=True)

    multiple_runs_metrics_dct = dict()
    for run_num, run_seed in enumerate(runs_seed_lst):
        models_metrics_dct = run_metrics_computation(dataset, test_set_fraction, dataset_name, run_seed,
                                                     config, models_tuned_params_df, n_estimators, sensitive_attributes, priv_values,
                                                     save_results=False, debug_mode=debug_mode)

        # Concatenate with previous results and save them in an overwrite mode each time for backups
        for model_name in models_metrics_dct.keys():
            model_metrics_df = models_metrics_dct[model_name]
            model_metrics_df['Run_Number'] = f'Run_{run_num + 1}'

            if multiple_runs_metrics_dct.get(model_name) is None:
                multiple_runs_metrics_dct[model_name] = model_metrics_df
            else:
                multiple_runs_metrics_dct[model_name] = pd.concat([multiple_runs_metrics_dct[model_name], model_metrics_df])

            result_filename = f'Metrics_{dataset_name}_{model_name}_{start_datetime.strftime("%Y%m%d__%H%M%S")}.csv'
            multiple_runs_metrics_dct[model_name].to_csv(f'{save_results_dir_path}/{result_filename}', index=False, mode='w')

    return multiple_runs_metrics_dct
