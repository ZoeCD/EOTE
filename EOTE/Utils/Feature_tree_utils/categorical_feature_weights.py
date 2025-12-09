import numpy as np
from EOTE.Protocols import FeatureWeightCalculatorInterface, CategoricalFeatureValueWeightsCalculatorInterface
from typing import List, Dict


class FeatureWeightByAUC(FeatureWeightCalculatorInterface):
    """Calculate categorical feature importance using multiclass AUC.

    Computes a feature weight based on the Area Under the ROC Curve (AUC) derived
    from a confusion matrix. Uses a one-vs-rest approach to extend binary AUC to
    multiclass classification.

    The multiclass AUC is calculated as:
    1. For each class, compute binary AUC (one vs all others)
    2. Average the binary AUCs across all classes

    Binary AUC is computed as: (Recall + Specificity) / 2
    - Recall = TP / (TP + FN)
    - Specificity = TN / (TN + FP)

    Higher AUC values (closer to 1.0) indicate the feature is more predictive.
    """

    def calculate_weight(self, confusion_matrix: np.ndarray) -> float:
        """Calculate feature weight from confusion matrix.

        Args:
            confusion_matrix: Square confusion matrix of shape (n_classes, n_classes)
                            where element [i, j] is count of instances with true
                            class i predicted as class j.

        Returns:
            Feature weight between 0.0 and 1.0, representing average multiclass AUC.
        """
        auc = self.__obtain_auc_multiclass(confusion_matrix)
        return auc

    def __obtain_auc_multiclass(self, confusion_matrix: np.ndarray) -> float:
        """Compute multiclass AUC using one-vs-rest approach.

        Args:
            confusion_matrix: Multiclass confusion matrix.

        Returns:
            Average AUC across all classes.
        """
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
        """Compute binary AUC from confusion matrix components.

        AUC is calculated as the average of recall and specificity.

        Args:
            tp: True positives
            tn: True negatives
            fp: False positives
            fn: False negatives

        Returns:
            Binary AUC between 0.0 and 1.0.
        """
        n_pos = tp + fn
        n_neg = tn + fp

        recall = tp / n_pos if n_pos > 0.0 else 0.0
        specificity = tn / n_neg if n_neg > 0.0 else 0.0

        return (recall + specificity) / 2.0


class CategoricalFeatureValuesWeightsByRowSum(CategoricalFeatureValueWeightsCalculatorInterface):
    """Calculate weights for each categorical value based on confusion matrix row sums.

    Computes a weight for each categorical value based on how frequently it appears
    in the confusion matrix. Row sums are normalized to create probability weights.

    The weight for value i is calculated as:
        weight[i] = row_sum[i] / total_sum

    where row_sum[i] is the sum of all values in row i of the confusion matrix.

    Higher weights indicate values that appear more frequently in predictions.
    """

    def calculate_values_weight(self, confusion_matrix: np.ndarray,  feature_values: List[str]) -> Dict[str, float]:
        """Calculate normalized weights for each categorical value.

        Args:
            confusion_matrix: Square confusion matrix of shape (n_classes, n_classes)
                            where rows correspond to feature values.
            feature_values: List of categorical value names corresponding to matrix rows.

        Returns:
            Dictionary mapping each feature value to its normalized weight (0.0 to 1.0).
            Weights sum to 1.0 across all values.

        Example:
            >>> confusion_matrix = np.array([[10, 2], [3, 15]])
            >>> feature_values = ["red", "blue"]
            >>> weights = calculator.calculate_values_weight(confusion_matrix, feature_values)
            >>> weights
            {'red': 0.4, 'blue': 0.6}  # row sums [12, 18] normalized by total 30
        """
        weight_by_feature_value = [sum(confusion_matrix[i]) for i in range(len(confusion_matrix))]
        total_sum = sum(weight_by_feature_value)
        weight_by_feature_value = [weight / total_sum for weight in weight_by_feature_value]
        return self.__transform_to_dict(weight_by_feature_value, feature_values)


    def __transform_to_dict(self, weights: List[float], feature_values: List[str]) -> Dict[str, float]:
        """Convert parallel lists of weights and values into a dictionary.

        Args:
            weights: List of normalized weight values.
            feature_values: List of categorical value names.

        Returns:
            Dictionary mapping value names to weights.
        """
        return {feature_value: weight for feature_value, weight in zip(feature_values, weights)}

