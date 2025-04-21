from EOTE.Protocols import FeatureTree
from EOTE.Utils import Feature
import numpy as np
import pandas as pd


class NumericalFeatureTree(FeatureTree):
    def __init__(self):
        self.feature = None
        self.regressor = None
        self.anomaly_scorer = None
        self.leaves_and_instances = None

    def set_feature(self, feature: Feature) -> None:
        self.feature = feature

    def fit(self, x: pd.DataFrame, y:pd.DataFrame) -> None:
        self.regressor = self.regressor.create_and_fit(x, y)
        self.__create_node_instance_dict(x, y)
        self.anomaly_scorer.set_domain(y)

    def calculate_score_normal(self, instance: pd.Series, actual_value: float) -> float:
        predicted_value = self.regressor.predict(instance.to_frame().T)[0]
        score = self.anomaly_scorer.calculate_score_normal(actual_value, predicted_value)
        if isinstance(score, np.ndarray):
            return score[0]
        return score

    def calculate_score_anomaly(self, instance: pd.Series, actual_value: float) -> float:
        predicted_value = self.regressor.predict(instance.to_frame().T)[0]
        score = self.anomaly_scorer.calculate_score_anomaly(actual_value, predicted_value)
        if isinstance(score, np.ndarray):
            return score[0]
        return score

    def __create_node_instance_dict(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        leaves_instances = self.regressor.apply(x)
        leaves_and_instances = {
            id_leaf: np.array([]) for id_leaf in np.unique(leaves_instances)
        }
        for id_leaf, instance_index in zip(leaves_instances, enumerate(x.values)):
            value = y.values[instance_index[0]]
            leaves_and_instances[id_leaf] = np.append(leaves_and_instances[id_leaf], [value], axis=0)
        self.leaves_and_instances = leaves_and_instances

