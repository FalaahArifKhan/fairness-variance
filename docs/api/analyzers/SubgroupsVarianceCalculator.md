# SubgroupsVarianceCalculator

SubgroupsVarianceCalculator description.



## Parameters

- **X_test**

    Processed features test set

- **y_test**

    Targets test set

- **sensitive_attributes_dct**

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these subgroups

- **test_groups** – defaults to `None`

    A dictionary where keys are sensitive attributes, and values input dataset rows  that are correspondent to these sensitive attributes




## Methods

???- note "compute_subgroups_metrics"

    Compute variance metrics for subgroups

    :param models_predictions: dict of lists, where key is a model index and value is model predictions based on X_test :return: dict of dicts, where key is 'overall' or a subgroup name, and value is a dict of metrics for this subgroup

    **Parameters**

    - **models_predictions**    
    - **save_results**    
    - **result_filename**    
    - **save_dir_path**    
    
???- note "save_metrics_to_file"

???- note "set_overall_stability_metrics"

