import numpy as np
from virny.datasets import ACSIncomeDataset
from aif360.metrics import BinaryLabelDatasetMetric

from source.preprocessing import optimized_preprocessing, get_distortion_acs_income
from configs.constants import EXPERIMENT_SEEDS, TEST_SET_FRACTION


exp_iter_num = 1
experiment_seed = EXPERIMENT_SEEDS[exp_iter_num - 1]
sensitive_attr_for_intervention = 'RACE'
privileged_groups = [{sensitive_attr_for_intervention: 1}]
unprivileged_groups = [{sensitive_attr_for_intervention: 0}]
intervention_options = {
    'distortion_fun': get_distortion_acs_income,
    'epsilon': .05,
    'clist': [0.99, 1.99, 2.99],
    'dlist': [.1, .05, 0]
}

data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                               subsample_size=500, subsample_seed=42)


if __name__ == '__main__':
    print(data_loader.X_data.head())

    data_loader.categorical_columns = [col for col in data_loader.categorical_columns if col not in ('SEX', 'RAC1P')]
    data_loader.X_data[sensitive_attr_for_intervention] = data_loader.X_data['RAC1P'].apply(lambda x: 1 if x == '1' else 0)
    data_loader.full_df = data_loader.full_df.drop(['SEX', 'RAC1P'], axis=1)
    data_loader.X_data = data_loader.X_data.drop(['SEX', 'RAC1P'], axis=1)

    # Fair preprocessing
    train_trans_df, test_trans_df, train_binary_dataset, test_binary_dataset = \
        optimized_preprocessing(data_loader,
                                opt_preproc_options=intervention_options,
                                sensitive_attribute=sensitive_attr_for_intervention,
                                test_set_fraction=TEST_SET_FRACTION,
                                dataset_split_seed=experiment_seed)

    metric_origin_train = BinaryLabelDatasetMetric(train_binary_dataset,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
    metric_trans_train = BinaryLabelDatasetMetric(train_trans_df,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    print('train', np.abs(metric_origin_train.mean_difference()), np.abs(metric_trans_train.mean_difference()))

    metric_origin_test = BinaryLabelDatasetMetric(test_binary_dataset,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    metric_trans_test = BinaryLabelDatasetMetric(test_trans_df,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    print('test', np.abs(metric_origin_test.mean_difference()), np.abs(metric_trans_test.mean_difference()))
