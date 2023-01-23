from utils.custom_classes.generic_pipeline import GenericPipeline


def create_base_pipeline(dataset, sensitive_attributes, priv_values, model_seed, test_set_fraction):
    base_pipeline = GenericPipeline(dataset, sensitive_attributes, priv_values)
    _ = base_pipeline.create_preprocessed_train_test_split(dataset, test_set_fraction, seed=model_seed)

    return base_pipeline


def create_tuned_base_model(init_model, model_name, models_tuned_params_df):
    model_params = eval(models_tuned_params_df.loc[models_tuned_params_df['Model_Name'] == model_name,
                                                   'Model_Best_Params'].iloc[0])
    return init_model.set_params(**model_params)
