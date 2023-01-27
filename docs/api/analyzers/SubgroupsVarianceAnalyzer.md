# SubgroupsVarianceAnalyzer

SubgroupsVarianceAnalyzer description.



## Parameters

- **model_setting**

    Constant from configs.constants.ModelSetting

- **n_estimators** (*int*)

    Number of estimators for bootstrap

- **base_model**

    Initialized base model to analyze

- **base_model_name** (*str*)

    Model name

- **bootstrap_fraction** (*float*)

    [0-1], fraction from train_pd_dataset for fitting an ensemble of base models

- **base_pipeline** (*source.custom_classes.generic_pipeline.GenericPipeline*)

    Initialized object of GenericPipeline class

- **dataset_name** (*str*)

    Name of dataset, used for correct results naming




## Methods

???- note "compute_metrics"

    Measure variance metrics for subgroups for the base model. Display stability plots for analysis if needed.  Save results to a .csv file if needed.

    :param save_results: bool if we need to save metrics in a file :param make_plots: bool, if display plots for analysis

    **Parameters**

    - **save_results**    
    - **result_filename**    
    - **save_dir_path**    
    - **make_plots**     – defaults to `True`    
    
