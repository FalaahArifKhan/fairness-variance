import pandas as pd


class MetricsComposer:
    def __init__(self, sensitive_attributes, model_metrics_df):
        self.sensitive_attributes = sensitive_attributes
        self.model_metrics_df = model_metrics_df

    def compose_metrics(self):
        groups_metrics_dct = dict()
        for protected_group in self.sensitive_attributes:
            dis_group = protected_group + '_dis'
            priv_group = protected_group + '_priv'
            cfm = self.model_metrics_df
            cfm = cfm.set_index('Metric')

            groups_metrics_dct[protected_group] = {
                # Bias metrics
                'Equalized_Odds_TPR': cfm[dis_group]['TPR'] - cfm[priv_group]['TPR'],
                'Equalized_Odds_FPR': cfm[dis_group]['FPR'] - cfm[priv_group]['FPR'],
                'Disparate_Impact': cfm[dis_group]['Positive-Rate'] / cfm[priv_group]['Positive-Rate'],
                'Statistical_Parity_Difference': cfm[dis_group]['Positive-Rate'] - cfm[priv_group]['Positive-Rate'],
                'Accuracy_Parity': cfm[dis_group]['Accuracy'] - cfm[priv_group]['Accuracy'],
                # Variance metrics
                'Label_Stability_Ratio': cfm[dis_group]['Label_Stability'] / cfm[priv_group]['Label_Stability'],
                'IQR_Parity': cfm[dis_group]['IQR'] - cfm[priv_group]['IQR'],
                'Std_Parity': cfm[dis_group]['Std'] - cfm[priv_group]['Std'],
                'Std_Ratio': cfm[dis_group]['Std'] / cfm[priv_group]['Std'],
                'Jitter_Parity': cfm[dis_group]['Jitter'] - cfm[priv_group]['Jitter'],
            }

        model_composed_metrics_df = pd.DataFrame(groups_metrics_dct).reset_index()
        model_composed_metrics_df = model_composed_metrics_df.rename(columns={"index": "Metric"})

        return model_composed_metrics_df
