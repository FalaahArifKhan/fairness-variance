import pandas as pd
import seaborn as sns

from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


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
