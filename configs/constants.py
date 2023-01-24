from enum import Enum
from dataclasses import dataclass


@dataclass
class CountPredictionStatsResponse:
    accuracy: float
    jitter: float
    means_lst: list
    stds_lst: list
    iqr_lst: list
    entropy_lst: list
    per_sample_accuracy_lst: list
    label_stability_lst: list


class ModelSetting(Enum):
    INCREMENTAL = "incremental"
    BATCH = "batch"


INTERSECTION_SIGN = '&'
