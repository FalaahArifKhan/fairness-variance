import math
import itertools
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from os import listdir
from scipy.stats import entropy
from os.path import isfile, join
from matplotlib import pyplot as plt

from configs.constants import CountPredictionStatsResponse
from utils.data_viz_utils import set_size


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


def count_prediction_stats(y_test, uq_results):
    """
    Compute means, stds, iqr, accuracy, jitter and transform predictions to pd df

    :param y_test: true labels
    :param uq_results: predicted labels
    """
    if isinstance(uq_results, np.ndarray):
        results = pd.DataFrame(uq_results)
    else:
        results = pd.DataFrame(uq_results).transpose()

    means_lst, stds_lst, iqr_lst, conf_interval_df = compute_stability_metrics(results)

    # Convert predict proba results of each model to correspondent labels
    uq_labels = results.applymap(lambda x: int(x<0.5))
    entropy_lst = np.apply_along_axis(compute_entropy, 1, uq_labels.transpose().values)
    jitter = compute_jitter(uq_labels.values)

    y_preds = np.array([int(x<0.5) for x in results.mean().values])
    accuracy = np.mean(np.array([y_preds[i] == int(y_test[i]) for i in range(len(y_test))]))

    per_sample_accuracy_lst, label_stability_lst = get_per_sample_accuracy(y_test, results)
    prediction_stats = CountPredictionStatsResponse(accuracy, jitter, means_lst, stds_lst, iqr_lst, entropy_lst,
                                                    per_sample_accuracy_lst, label_stability_lst)

    return y_preds, uq_labels, prediction_stats


def get_per_sample_accuracy(y_test, results):
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


def generate_bootstrap(features, labels, boostrap_size, with_replacement=True):
    bootstrap_index = np.random.choice(features.shape[0], size=boostrap_size, replace=with_replacement)
    bootstrap_features = pd.DataFrame(features).iloc[bootstrap_index].values
    bootstrap_labels = pd.DataFrame(labels).iloc[bootstrap_index].values
    if len(bootstrap_features) == boostrap_size:
        return bootstrap_features, bootstrap_labels
    else:
        raise ValueError('Bootstrap samples are not of the size requested')


def display_result_plots(results_dir):
    sns.set_style("darkgrid")
    results = dict()
    filenames = [f for f in listdir(results_dir) if isfile(join(results_dir, f))]

    for filename in filenames:
        results_df = pd.read_csv(results_dir + filename)
        results[f'{results_df.iloc[0]["Base_Model_Name"]}_{results_df.iloc[0]["N_Estimators"]}_estimators'] = results_df

    y_metrics = ['SPD_Race', 'SPD_Sex', 'SPD_Race_Sex', 'EO_Race', 'EO_Sex', 'EO_Race_Sex']
    x_metrics = ['Label_Stability', 'General_Ensemble_Accuracy', 'Std']
    for x_metric in x_metrics:
        for y_metric in y_metrics:
            x_lim = 0.3 if x_metric == 'SD' else 1.0
            display_uncertainty_plot(results, x_metric, y_metric, x_lim)


def display_uncertainty_plot(results, x_metric, y_metric, x_lim):
    fig, ax = plt.subplots()
    set_size(15, 8, ax)

    # List of all markers -- https://matplotlib.org/stable/api/markers_api.html
    markers = ['.', 'o', '+', '*', '|', '<', '>', '^', 'v', '1', 's', 'x', 'D', 'P', 'H']
    techniques = results.keys()
    shapes = []
    for idx, technique in enumerate(techniques):
        a = ax.scatter(results[technique][x_metric], results[technique][y_metric], marker=markers[idx], s=100)
        shapes.append(a)

    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.xlim(0, x_lim)
    plt.title(f'{x_metric} [{y_metric}]', fontsize=20)
    ax.legend(shapes, techniques, fontsize=12, title='Markers')

    plt.show()
