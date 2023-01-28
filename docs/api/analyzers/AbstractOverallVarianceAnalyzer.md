# AbstractOverallVarianceAnalyzer

AbstractOverallVarianceAnalyzer description.



## Parameters

- **base_model**

    Base model for stability measuring

- **base_model_name** (*str*)

    Model name like 'HoeffdingTreeClassifier' or 'LogisticRegression'

- **bootstrap_fraction** (*float*)

    [0-1], fraction from train_pd_dataset for fitting an ensemble of base models

- **X_train** (*pandas.core.frame.DataFrame*)

    Processed features train set

- **y_train** (*pandas.core.frame.DataFrame*)

    Targets train set

- **X_test** (*pandas.core.frame.DataFrame*)

    Processed features test set

- **y_test** (*pandas.core.frame.DataFrame*)

    Targets test set

- **dataset_name** (*str*)

    Name of dataset, used for correct results naming

- **n_estimators** (*int*)

    Number of estimators in ensemble to measure base_model stability




## Methods

???- note "UQ_by_boostrap"

    Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples.

    **Parameters**

    - **boostrap_size**     (*int*)    
    - **with_replacement**     (*bool*)    
    
    **Returns**

    Dictionary where keys are models indexes,
    
???- note "compute_metrics"

    Measure metrics for the base model. Display plots for analysis if needed. Save results to a .pkl file

    **Parameters**

    - **make_plots**     (*bool*)     – defaults to `False`    
    - **save_results**     (*bool*)     – defaults to `True`    
    
???- note "get_metrics_dict"

???- note "print_metrics"

???- note "save_metrics_to_file"

