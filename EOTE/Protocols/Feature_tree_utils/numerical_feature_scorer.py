from typing import Protocol


class NumericalFeatureAnomalyScorer(Protocol):

    def calculate_score_normal(self, real_value: float, predicted_value: float) -> float:
        ...

    def calculate_score_anomaly(self, real_value: float, predicted_value: float) -> float:
        ...
