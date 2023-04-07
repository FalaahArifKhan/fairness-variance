import pandas as pd
from abc import ABCMeta, abstractmethod


class AbstractGenerator(metaclass=ABCMeta):
    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def fit(self, df, target_column: str = None):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, target_column: str = None):
        pass

    @abstractmethod
    def fit_transform(self, df, target_column: str = None):
        pass
