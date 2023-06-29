import altair as alt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


class ExperimentsVisualizer:
    def __init__(self, exp_subgroup_metrics_dct: dict, exp_avg_runs_group_metrics_dct: dict,
                 dataset_name: str, model_names: list, sensitive_attrs: list):
        sns.set_theme(style="whitegrid")

        self.exp_subgroup_metrics_dct = exp_subgroup_metrics_dct
        self.exp_avg_runs_group_metrics_dct = exp_avg_runs_group_metrics_dct
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.sensitive_attrs = sensitive_attrs

        # Technical attributes
        self.all_error_subgroup_metrics = [
            'TPR',
            'TNR',
            'FNR',
            'FPR',
            'PPV',
            'Accuracy',
            'F1',
            'Positive-Rate',
            'Selection-Rate',
        ]
        self.all_variance_subgroup_metrics = [
            # 'Mean',
            'Std',
            'IQR',
            # 'Entropy',
            'Jitter',
            # 'Per_Sample_Accuracy',
            'Label_Stability',
        ]
        self.all_group_fairness_metrics_lst = [
            'Accuracy_Parity',
            'Equalized_Odds_TPR',
            'Equalized_Odds_FPR',
            'Equalized_Odds_FNR',
            'Disparate_Impact',
            'Statistical_Parity_Difference',
        ]
        self.all_group_variance_metrics_lst = [
            'IQR_Parity',
            'Label_Stability_Ratio',
            'Std_Parity',
            'Std_Ratio',
            'Jitter_Parity',
        ]

        # Create exp_avg_runs_subgroup_metrics_dct
        exp_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.exp_subgroup_metrics_dct.keys():
            for exp_iter in self.exp_subgroup_metrics_dct[model_name].keys():
                for percentage in self.exp_subgroup_metrics_dct[model_name][exp_iter].keys():
                    multiple_runs_subgroup_metrics_df = self.exp_subgroup_metrics_dct[model_name][exp_iter][percentage]
                    columns_to_group = [col for col in multiple_runs_subgroup_metrics_df.columns
                                        if col not in ('Bootstrap_Model_Seed', 'Run_Number', 'Record_Create_Date_Time')]
                    exp_avg_runs_subgroup_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {})[percentage] = multiple_runs_subgroup_metrics_df[columns_to_group].groupby(
                        ['Metric', 'Model_Name']
                    ).mean().reset_index()

        self.exp_avg_runs_subgroup_metrics_dct = exp_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_runs_subgroup_metrics_dct
        melted_exp_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.exp_avg_runs_subgroup_metrics_dct.keys():
            for exp_iter in self.exp_avg_runs_subgroup_metrics_dct[model_name].keys():
                for percentage in self.exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter].keys():
                    model_subgroup_metrics_df = self.exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter][percentage]
                    subgroup_names = [col for col in model_subgroup_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
                    melted_model_subgroup_metrics_df = model_subgroup_metrics_df.melt(
                        id_vars=[col for col in model_subgroup_metrics_df.columns if col not in subgroup_names],
                        value_vars=subgroup_names,
                        var_name="Subgroup",
                        value_name="Metric_Value"
                    )
                    melted_exp_avg_runs_subgroup_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {})[percentage] = melted_model_subgroup_metrics_df

        self.melted_exp_avg_runs_subgroup_metrics_dct = melted_exp_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_runs_group_metrics_dct
        melted_exp_avg_runs_group_metrics_dct = dict()
        for model_name in self.exp_avg_runs_group_metrics_dct.keys():
            for exp_iter in self.exp_avg_runs_group_metrics_dct[model_name].keys():
                for percentage in self.exp_avg_runs_group_metrics_dct[model_name][exp_iter].keys():
                    model_group_metrics_df = self.exp_avg_runs_group_metrics_dct[model_name][exp_iter][percentage]
                    group_names = [col for col in model_group_metrics_df.columns if col not in ('Metric', 'Model_Name')]
                    melted_model_group_metrics_df = model_group_metrics_df.melt(
                        id_vars=[col for col in model_group_metrics_df.columns if col not in group_names],
                        value_vars=group_names,
                        var_name="Group",
                        value_name="Metric_Value"
                    )
                    melted_exp_avg_runs_group_metrics_dct.setdefault(model_name, {}) \
                        .setdefault(exp_iter, {})[percentage] = melted_model_group_metrics_df

        self.melted_exp_avg_runs_group_metrics_dct = melted_exp_avg_runs_group_metrics_dct

        # Create melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct
        melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct = dict()
        for model_name in self.melted_exp_avg_runs_subgroup_metrics_dct.keys():
            first_exp_iter = list(self.melted_exp_avg_runs_subgroup_metrics_dct[model_name].keys())[0]

            for percentage in self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][first_exp_iter].keys():
                multiple_pct_exp_iters_subgroup_metrics_df = pd.DataFrame()

                for exp_iter in self.melted_exp_avg_runs_subgroup_metrics_dct[model_name].keys():
                    multiple_runs_subgroup_metrics_df = self.melted_exp_avg_runs_subgroup_metrics_dct[model_name][exp_iter][percentage]
                    multiple_pct_exp_iters_subgroup_metrics_df = pd.concat([multiple_pct_exp_iters_subgroup_metrics_df, multiple_runs_subgroup_metrics_df])

                columns_to_group = [col for col in multiple_pct_exp_iters_subgroup_metrics_df.columns
                                    if col not in ('Bootstrap_Model_Seed', 'Run_Number', 'Record_Create_Date_Time',
                                                   'Dataset_Split_Seed', 'Experiment_Iteration', 'Model_Init_Seed',
                                                   'Model_Params')]
                melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct.setdefault(model_name, {})[percentage] = \
                    multiple_pct_exp_iters_subgroup_metrics_df[columns_to_group].groupby(
                        ['Model_Name', 'Test_Set_Index', 'Metric', 'Subgroup']
                    ).mean().reset_index()

        self.melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct = melted_exp_avg_exp_iters_avg_runs_subgroup_metrics_dct

        # Create melted_exp_avg_exp_iters_avg_runs_group_metrics_dct
        melted_exp_avg_exp_iters_avg_runs_group_metrics_dct = dict()
        for model_name in self.melted_exp_avg_runs_group_metrics_dct.keys():
            first_exp_iter = list(self.melted_exp_avg_runs_group_metrics_dct[model_name].keys())[0]

            for percentage in self.melted_exp_avg_runs_group_metrics_dct[model_name][first_exp_iter].keys():
                multiple_pct_exp_iters_group_metrics_df = pd.DataFrame()

                for exp_iter in self.melted_exp_avg_runs_group_metrics_dct[model_name].keys():
                    multiple_runs_group_metrics_df = self.melted_exp_avg_runs_group_metrics_dct[model_name][exp_iter][percentage]
                    multiple_pct_exp_iters_group_metrics_df = pd.concat([multiple_pct_exp_iters_group_metrics_df, multiple_runs_group_metrics_df])

                columns_to_group = [col for col in multiple_pct_exp_iters_group_metrics_df.columns
                                    if col not in ('Bootstrap_Model_Seed', 'Run_Number', 'Record_Create_Date_Time',
                                                   'Dataset_Split_Seed', 'Experiment_Iteration', 'Model_Init_Seed',
                                                   'Model_Params')]
                melted_exp_avg_exp_iters_avg_runs_group_metrics_dct.setdefault(model_name, {})[percentage] = \
                    multiple_pct_exp_iters_group_metrics_df[columns_to_group].groupby(
                        ['Model_Name', 'Metric', 'Group']
                    ).mean().reset_index()

        self.melted_exp_avg_exp_iters_avg_runs_group_metrics_dct = melted_exp_avg_exp_iters_avg_runs_group_metrics_dct

    def create_subgroup_metrics_box_plot_for_multiple_exp_iters(self, target_percentage: float,
                                                                target_preprocessing_technique: str,
                                                                subgroup_metrics: list = None,
                                                                subgroup_metrics_type: str = None):
        if subgroup_metrics_type is not None and not SubgroupMetricsType.has_value(subgroup_metrics_type):
            raise ValueError(f'subgroup_metrics_type must be in {tuple(SubgroupMetricsType._value2member_map_.keys())}')

        if subgroup_metrics is None:
            if subgroup_metrics_type is None:
                subgroup_metrics = self.all_error_subgroup_metrics + self.all_variance_subgroup_metrics
            else:
                subgroup_metrics = self.all_error_subgroup_metrics if subgroup_metrics_type == SubgroupMetricsType.ERROR.value \
                    else self.all_variance_subgroup_metrics

        subgroup = 'overall'
        all_models_pct_subgroup_metrics_df = pd.DataFrame()
        for model_name in self.exp_avg_runs_subgroup_metrics_dct.keys():
            for exp_iter in self.exp_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique].keys():
                all_models_pct_subgroup_metrics_df = pd.concat([
                    all_models_pct_subgroup_metrics_df,
                    self.exp_avg_runs_subgroup_metrics_dct[model_name][target_preprocessing_technique][exp_iter][target_percentage]
                ])
        all_models_pct_subgroup_metrics_df = all_models_pct_subgroup_metrics_df.reset_index(drop=True)

        to_plot = all_models_pct_subgroup_metrics_df[all_models_pct_subgroup_metrics_df['Metric'].isin(subgroup_metrics)]
        plt.figure(figsize=(15, 7))
        ax = sns.boxplot(x=to_plot['Metric'],
                         y=to_plot[subgroup],
                         hue=to_plot['Model_Name'])

        plt.legend(loc='upper left',
                   ncol=2,
                   fancybox=True,
                   shadow=True,
                   fontsize=14)
        plt.xlabel("Metric name", fontsize=16)
        plt.ylabel("Metric value", fontsize=16)
        ax.tick_params(labelsize=14)
        fig = ax.get_figure()
        fig.tight_layout()
