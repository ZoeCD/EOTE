import sys
sys.path.append(".")
import unittest
import numpy as np
from EOTE.Utils import FeatureWeightByAUC, CategoricalFeatureValuesWeightsByRowSum


class TestObtainAUCMulticlass(unittest.TestCase):

    def setUp(self):
        self.confusion_matrix = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        self.expected_auc = 1.0

    def test_obtain_auc_multiclass(self):
        auc = FeatureWeightByAUC().calculate_weight(self.confusion_matrix)
        self.assertEqual(auc, self.expected_auc)

    def test_obtain_auc_multiclass_with_zero_values(self):
        zero_matrix = np.zeros((3, 3))
        result = FeatureWeightByAUC().calculate_weight(zero_matrix)
        self.assertEqual(result, 0.0)

    def test_obtain_auc_multiclass_with_only_one_class(self):
        one_matrix = np.ones((1, 1))
        result = FeatureWeightByAUC().calculate_weight(one_matrix)
        self.assertEqual(result, 0.5)


class CategoricalFeatureWeightsTest(unittest.TestCase):
    def test_sum_rows_feature_value_weights_passing_all_zeros(self):
        confusion_matrix = np.array(
            [[22, 156, 51, 9, 7, 0, 8],
             [164, 34, 111, 47, 4, 0, 6],
             [154, 129, 50, 73, 21, 0, 0],
             [106, 99, 76, 22, 40, 5, 0],
             [34, 28, 42, 33, 43, 26, 0],
             [2, 3, 10, 9, 33, 12, 0],
             [44, 10, 2, 2, 0, 0, 6]]
        )
        feature_values = ["blue", "pink", "red", "black", "purple", "grey", "orange"]
        func = CategoricalFeatureValuesWeightsByRowSum().calculate_values_weight(confusion_matrix, feature_values)
        expected_values = [0.14598961338719, 0.211194460473168, 0.246393537218696, 0.200807847663012,
                           0.118869013271783, 0.0398153491055972, 0.036930178880554]
        for actual, expected in zip(func.values(), expected_values):
            self.assertAlmostEqual(actual, expected)

    def test_sum_rows_feature_value_weights(self):
        confusion_matrix = np.array([
            [237, 120, 21, 6],
            [135, 339, 174, 41],
            [46, 201, 69, 78],
            [21, 125, 109, 11],
        ])
        feature_values = ["blue", "pink", "red", "black"]
        func = CategoricalFeatureValuesWeightsByRowSum().calculate_values_weight(confusion_matrix, feature_values)
        expected_values = {"blue":0.221581073283324, "pink":0.397576457010964, "red":0.22735141373341, "black":0.153491055972302}
        for actual, expected in zip(func.values(), expected_values.values()):
            self.assertAlmostEqual(actual, expected)

