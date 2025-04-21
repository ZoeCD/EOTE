from typing import Protocol, List, Dict
from numpy import ndarray


class FeatureWeightCalculatorInterface(Protocol):
    def calculate_weight(self, confusion_matrix: ndarray) -> float:
        ...


class CategoricalFeatureValueWeightsCalculatorInterface(Protocol):
    def calculate_weight(self, confusion_matrix: ndarray, feature_values: List[str]) -> Dict[str, float]:
        ...
