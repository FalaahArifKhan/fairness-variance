import math
import itertools
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import entropy


def compute_label_stability(predicted_labels):
    """
    Label stability is defined as the absolute difference between the number of times the sample is classified as 0 and 1
    If the absolute difference is large, the label is more stable
    If the difference is exactly zero then it's extremely unstable --- equally likely to be classified as 0 or 1
    """
    count_pos = sum(predicted_labels)
    count_neg = len(predicted_labels) - count_pos
    return np.abs(count_pos - count_neg)/len(predicted_labels)


def compute_churn(predicted_labels_1, predicted_labels_2):
    """
    Pairwise stability metric for two model predictions
    """
    return np.sum([int(predicted_labels_1[i] != predicted_labels_2[i])
                   for i in range(len(predicted_labels_1))]) / len(predicted_labels_1)


def compute_jitter(models_prediction_labels):
    """
    Jitter is a stability metric that shows how the base model predictions fluctuate.
    Values closer to 0 -- perfect stability, values closer to 1 -- extremely bad stability.
    """
    n_models = len(models_prediction_labels)
    models_idx_lst = [i for i in range(n_models)]
    churns_sum = 0
    for i, j in itertools.combinations(models_idx_lst, 2):
        churns_sum += compute_churn(models_prediction_labels[i], models_prediction_labels[j])

    return churns_sum / (n_models * (n_models - 1) * 0.5)


def compute_entropy(labels):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    # Compute entropy
    ent = 0.
    base = math.e
    for i in probs:
        ent -= i * math.log(i, base)

    return ent


def compute_conf_interval(labels):
    """ Create 95% confidence interval for population mean weight """
    return sp.stats.norm.interval(alpha=0.95, loc=np.mean(labels), scale=sp.stats.sem(labels))


def compute_stability_metrics(results):
    means_lst = results.mean().values
    stds_lst = results.std().values
    iqr_lst = sp.stats.iqr(results, axis=0)
    conf_interval_df = pd.DataFrame(np.apply_along_axis(compute_conf_interval, 1, results.transpose().values),
                                    columns=['lower_bound', 'upper_bound'])

    return means_lst, stds_lst, iqr_lst, conf_interval_df


def compute_per_sample_accuracy(y_test, results):
    """

    :param y_test: y test dataset
    :param results: results variable from count_prediction_stats()
    :return: per_sample_accuracy and label_stability (refer to https://www.osti.gov/servlets/purl/1527311)
    """
    per_sample_predictions = {}
    label_stability = []
    per_sample_accuracy = []
    acc = None
    for sample in range(len(y_test)):
        per_sample_predictions[sample] =  [int(x<0.5) for x in results[sample].values]
        label_stability.append(compute_label_stability(per_sample_predictions[sample]))

        if y_test[sample] == 1:
            acc = np.mean(per_sample_predictions[sample])
        elif y_test[sample] == 0:
            acc = 1 - np.mean(per_sample_predictions[sample])
        if acc is not None:
            per_sample_accuracy.append(acc)

    return per_sample_accuracy, label_stability
