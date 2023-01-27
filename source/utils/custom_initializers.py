import yaml
from munch import DefaultMunch

from source.custom_classes.generic_pipeline import GenericPipeline
from source.utils.common_helpers import validate_config


__all__ = []


def create_config_obj(config_yaml_path):
    with open(config_yaml_path) as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader)

    config_obj = DefaultMunch.fromDict(config_dct)
    validate_config(config_obj)

    return config_obj


def create_models_config_from_tuned_params_df(models_config_for_tuning, models_tuned_params_df):
    experiment_models_config = dict()
    for model_idx in range(len(models_config_for_tuning)):
        model_name = models_config_for_tuning[model_idx]["model_name"]
        base_model = create_tuned_base_model(models_config_for_tuning[model_idx]['model'], model_name, models_tuned_params_df)
        experiment_models_config[model_name] = base_model

    return experiment_models_config


def create_base_pipeline(dataset, sensitive_attributes_dct, model_seed, test_set_fraction):
    base_pipeline = GenericPipeline(dataset, sensitive_attributes_dct)
    _ = base_pipeline.create_preprocessed_train_test_split(dataset, test_set_fraction, seed=model_seed)

    return base_pipeline


def create_tuned_base_model(init_model, model_name, models_tuned_params_df):
    model_params = eval(models_tuned_params_df.loc[models_tuned_params_df['Model_Name'] == model_name,
                                                   'Model_Best_Params'].iloc[0])
    return init_model.set_params(**model_params)
