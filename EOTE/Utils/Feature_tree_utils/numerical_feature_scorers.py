from EOTE.Protocols import NumericalFeatureAnomalyScorer
from pandas import DataFrame
import numpy as np


class AnomalyDomainScorer(NumericalFeatureAnomalyScorer):

    def __init__(self):
        self.domain = None #
        self.anomaly_score = None

    def set_domain(self, data):
        self.domain = (data.min(), data.max()) #

    def calculate_score_normal(self, real_value: float, predicted_value: float) -> float:
        if real_value - predicted_value == 0:
            return 1.0
        score = abs(real_value - predicted_value) / abs(self.domain[1] - self.domain[0])
        return 1.0 - score

    def calculate_score_anomaly(self, real_value: float, predicted_value: float) -> float:
        if real_value - predicted_value == 0:
            return 0
        score = abs(real_value - predicted_value) / abs(self.domain[1] - self.domain[0])
        return score


class AnomalyBoxplotScorer(NumericalFeatureAnomalyScorer):

    def __init__(self):
        self.domain = None
        self.anomaly_score = None

    def set_domain(self, data: DataFrame) -> None:
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)
        self.domain = (lower_fence, upper_fence)

    def calculate_score_normal(self, real_value: float, predicted_value: float) -> float:
        score = abs(real_value - predicted_value) / abs(self.domain[1] - self.domain[0])
        return 1.0 - score

    def calculate_score_anomaly(self, real_value: float, predicted_value: float) -> float:
        score = abs(real_value - predicted_value) / abs(self.domain[1] - self.domain[0])
        return score
