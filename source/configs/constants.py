from enum import Enum


class SubgroupMetricsType(Enum):
    ERROR = "error"
    VARIANCE = "variance"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class GroupMetricsType(Enum):
    FAIRNESS = "fairness"
    VARIANCE = "variance"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


NULL_PREDICTOR_SEED = 42
