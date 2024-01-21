# fairness-variance

Studying model uncertainty and instability for different demographic groups.


## Structure of the repo

* `configs` folder includes all general configurations used for experiments.
* `data` folder includes datasets used in experiments. Rest of the datasets are present in the _Virny_ library.
* `results` folder contains tuned hyper-parameters for all models used in experiments.
* `source` folder includes Python modules for experimental pipelines and visualizations.
* `notebooks` folder includes Jupyter notebooks used for all experiments. It has the following subfolders:
  * `notebooks/diff_fairness_interventions_exp` contains notebooks for the experiment with different fairness interventions.
  * `notebooks/diff_train_set_sizes_exp` contains notebooks for the experiment with different train set sizes and in-domain / out-of-domain settings.
  * `notebooks/mult_repair_levels_exp` contains notebooks for the experiment with different repair levels for Disparate Impact Remover.
  * `notebooks/visualizations_for_all_datasets` contains notebooks with visualizations for all experiments aggregated over all datasets and model types.


## Table with used parameters for each dataset and fairness intervention pair

|                                        | ACS Income                                  | ACS PublicCoverage                            | Law School                                    | Student Performance                           |
|:---------------------------------------|:--------------------------------------------|:----------------------------------------------|:----------------------------------------------|:----------------------------------------------|
| _Disparate Impact Remover (DIR)_         | repair\_level = 0.7                         | repair\_level = 0.6                           | repair\_level = 0.6                           | repair\_level = 0.7                           |
| _Learning Fair Representations (LFR)_    | {'k': 5, 'Ax': 0.01, 'Ay': 1.0, 'Az': 50.0} | {'k': 10, ‘Ax’: 0.1, ‘Ay’: 1.0, 'Az': 2.0}    | {'k': 5, 'Ax': 0.01, 'Ay': 1.0, ‘Az’: 50.0}   | {'k': 10, 'Ax': 0.1, ‘Ay’: 1.0, 'Az': 2.0}    |
| _Equalized Odds Postprocessor (EOP)_     | Apply (no parameters)                       | Apply (no parameters)                         | Apply (no parameters)                         | Apply (no parameters)                         |
| _Reject Option Classification (ROC)_     | Apply with default settings                 | Apply with default settings                   | Apply with default settings                   | Apply with default settings                   |
| _Exponentiated Gradient Reduction (EGR)_ | constraints = 'DemographicParity'           | constraints = 'DemographicParity'             | constraints = 'DemographicParity'             | constraints = 'DemographicParity'             |
| _Adversarial Debiasing (ADB)_            | <p>num\_epochs = 200,</p><p>debias = True</p>      | <p>num\_epochs = 200,</p><p>debias = True</p> | <p>num\_epochs = 200,</p><p>debias = True</p> | <p>num\_epochs = 200,</p><p>debias = True</p> |
