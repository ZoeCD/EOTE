import numpy as np
from EOTE.Protocols import FeatureWeightCalculatorInterface, CategoricalFeatureValueWeightsCalculatorInterface
from typing import List, Dict


class FeatureWeightByAUC(FeatureWeightCalculatorInterface):

    def calculate_weight(self, confusion_matrix: np.ndarray) -> float:
        auc = self.__obtain_auc_multiclass(confusion_matrix)
        return auc

    def __obtain_auc_multiclass(self, confusion_matrix: np.ndarray) -> float:
        sum_val = 0.0
        for i in range(len(confusion_matrix)):
            tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
            tp += confusion_matrix[i][i]
            for j in range(len(confusion_matrix)):
                if i != j:
                    fn += confusion_matrix[i][j]
                    fp += confusion_matrix[j][i]
                    tn += confusion_matrix[j][j]
            sum_val += self.__obtain_auc_binary(tp, tn, fp, fn)

        avg = sum_val / len(confusion_matrix)
        return avg

    def __obtain_auc_binary(self, tp: float, tn: float, fp: float, fn: float) -> float:
        n_pos = tp + fn
        n_neg = tn + fp

        recall = tp / n_pos if n_pos > 0.0 else 0.0
        specificity = tn / n_neg if n_neg > 0.0 else 0.0

        return (recall + specificity) / 2.0


class CategoricalFeatureValuesWeightsByRowSum(CategoricalFeatureValueWeightsCalculatorInterface):

    def calculate_values_weight(self, confusion_matrix: np.ndarray,  feature_values: List[str]) -> Dict[str, float]:
        weight_by_feature_value = [sum(confusion_matrix[i]) for i in range(len(confusion_matrix))]
        total_sum_chat = sum(weight_by_feature_value)
        weight_by_feature_value = [weight / total_sum_chat for weight in weight_by_feature_value]
        return self.__transform_to_dict(weight_by_feature_value, feature_values)


    def __transform_to_dict(self, weights: List[float], feature_values: List[str]) -> Dict[str, float]:
        return {feature_value: weight for feature_value, weight in zip(feature_values, weights)}

