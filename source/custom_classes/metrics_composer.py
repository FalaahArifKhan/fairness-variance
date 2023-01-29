import pandas as pd


class MetricsComposer:
    """
    Composer class that combines different metrics to create new ones such as 'Disparate_Impact' or 'Accuracy_Parity'

    Parameters
    ----------
    sensitive_attributes_dct
        A dictionary where keys are sensitive attribute names (including attributes intersections),
         and values are privilege values for these attributes
    model_metrics_df
        Dataframe of subgroups metrics for one model
    """
    def __init__(self, sensitive_attributes_dct: dict, model_metrics_df: pd.DataFrame):
        self.sensitive_attributes_dct = sensitive_attributes_dct
        self.model_metrics_df = model_metrics_df

    def compose_metrics(self):
        """
        Compose subgroups metrics from self.model_metrics_df.

        Return a dictionary of composed metrics.
        """
        groups_metrics_dct = dict()
        for sensitive_attr in self.sensitive_attributes_dct.keys():
            dis_group = sensitive_attr + '_dis'
            priv_group = sensitive_attr + '_priv'
            cfm = self.model_metrics_df
            cfm = cfm.set_index('Metric')

            groups_metrics_dct[sensitive_attr] = {
                # Group statistical Bias metrics
                'Equalized_Odds_TPR': cfm[dis_group]['TPR'] - cfm[priv_group]['TPR'],
                'Equalized_Odds_FPR': cfm[dis_group]['FPR'] - cfm[priv_group]['FPR'],
                'Disparate_Impact': cfm[dis_group]['Positive-Rate'] / cfm[priv_group]['Positive-Rate'],
                'Statistical_Parity_Difference': cfm[dis_group]['Positive-Rate'] - cfm[priv_group]['Positive-Rate'],
                'Accuracy_Parity': cfm[dis_group]['Accuracy'] - cfm[priv_group]['Accuracy'],
                # Group variance metrics
                'Label_Stability_Ratio': cfm[dis_group]['Label_Stability'] / cfm[priv_group]['Label_Stability'],
                'IQR_Parity': cfm[dis_group]['IQR'] - cfm[priv_group]['IQR'],
                'Std_Parity': cfm[dis_group]['Std'] - cfm[priv_group]['Std'],
                'Std_Ratio': cfm[dis_group]['Std'] / cfm[priv_group]['Std'],
                'Jitter_Parity': cfm[dis_group]['Jitter'] - cfm[priv_group]['Jitter'],
            }

        model_composed_metrics_df = pd.DataFrame(groups_metrics_dct).reset_index()
        model_composed_metrics_df = model_composed_metrics_df.rename(columns={"index": "Metric"})

        return model_composed_metrics_df
