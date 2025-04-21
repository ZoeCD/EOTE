from pandas import DataFrame, Series
from typing import Protocol, Any


class DecisionTree(Protocol):
    def create_and_fit(self, x: DataFrame, y: Series):
        ...

