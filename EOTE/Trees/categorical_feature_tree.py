from EOTE.Protocols import FeatureTree
from EOTE.Utils import Feature
import pandas as pd
from typing import List
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


class CategoricalFeatureTree(FeatureTree):
    def __init__(self):
        self.feature = None
        self.feature_weight = None
        self.value_weights = None
        self.classifier = None

        self.feature_weight_calculator = None
        self.value_weights_calculator = None
        self.tree_creator = None
        self.confusion_matrix = None

    def set_feature(self, feature: Feature) -> None:
        self.feature = feature

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.classifier = self.tree_creator.create_and_fit(x, y)
        self.__get_confusion_matrix(x, y, feature_values=self.feature.values(), classifier=self.classifier)
        self.feature_weight = self.feature_weight_calculator.calculate_weight(self.confusion_matrix)
        self.value_weights = self.value_weights_calculator.calculate_values_weight(self.confusion_matrix,
                                                                                   self.feature.values())

    def __get_confusion_matrix(self, x: pd.DataFrame, y: pd.DataFrame, feature_values: List[str], classifier) -> None:
        y_pred = cross_val_predict(classifier, x, y, cv=5)
        self.confusion_matrix = confusion_matrix(y, y_pred, labels=feature_values)

    def calculate_score_normal(self, instance: pd.Series, actual_value: str) -> float:
        if self.__check_if_value_in_feature_values(actual_value):
            classifier_probabilities = self.__get_classifier_probabilities(instance)
            actual_value_weight = self.value_weights[actual_value]
            return self.feature_weight * actual_value_weight * classifier_probabilities[actual_value] # por qué lo calculamos diferente?
        return 0.0

    def __check_if_value_in_feature_values(self, actual_value: str) -> bool:
        if actual_value in self.feature.values():
            return True

    def __get_classifier_probabilities(self, instance: pd.Series) -> dict:
        classifier_probabilities = self.classifier.predict_proba(instance.to_frame().T)[0]
        return dict(zip(self.classifier.classes_, classifier_probabilities))

    def calculate_score_anomaly(self, instance: pd.Series, actual_value: str) -> float:
        if self.__check_if_value_in_feature_values(actual_value):
            classifier_probabilities = self.__get_classifier_probabilities(instance)
            return self.__sum_anomaly(classifier_probabilities, actual_value)
        return 0.0

    def __sum_anomaly(self, classifier_probabilities: dict, actual_value: str) -> float:
        current_sum_anomaly = 0
        count_anomaly_votes = 0

        for class_label, probability in classifier_probabilities.items():
            if class_label != actual_value and probability > 0:
                current_sum_anomaly += self.value_weights[class_label] * probability
                count_anomaly_votes += 1

        return self.feature_weight * current_sum_anomaly / count_anomaly_votes if count_anomaly_votes > 0 else 0

