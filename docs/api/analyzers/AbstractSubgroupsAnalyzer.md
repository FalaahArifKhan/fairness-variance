# AbstractSubgroupsAnalyzer

AbstractSubgroupsAnalyzer description.



## Parameters

- **X_test** (*pandas.core.frame.DataFrame*)

    Processed features test set

- **y_test** (*pandas.core.frame.DataFrame*)

    Targets test set

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attributes names (including attributes intersections),  and values are privilege values for these subgroups

- **test_groups** (*dict*)

    A dictionary where keys are sensitive attributes, and values input dataset rows  that are correspondent to these sensitive attributes




## Methods

???- note "compute_subgroups_metrics"

???- note "save_metrics_to_file"

