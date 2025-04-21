from typing import Protocol, Union
from pandas import DataFrame, Series


class FeatureTree(Protocol):

    def set_feature(self, feature) -> None:
        ...

    def fit(self, x: DataFrame, y: DataFrame) -> None:
        ...

    def calculate_score_normal(self, instance: Series, actual_value: Union[str, float]) -> float:
        ...

    def calculate_score_anomaly(self, instance: Series, actual_value: Union[str, float]) -> float:
        ...