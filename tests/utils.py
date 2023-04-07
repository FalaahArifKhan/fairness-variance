import numpy as np


def detect_outliers_std(old_df, new_df, column):
    mean = np.mean(old_df[column])
    std = np.std(old_df[column])
    # Flag values as outliers if they are more than 3 stddevs away from the mean
    outliers = new_df[(new_df[column] > (mean + 3 * std)) | (new_df[column] < (mean - 3 * std))]

    return outliers
