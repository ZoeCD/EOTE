from EOTE.Trees import CategoricalFeatureTree
from EOTE.Utils import Feature
import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class MockFeature:
    def __init__(self, values):
        self._values = values

    def values(self):
        return self._values
    
class SimpleTreeCreator:
    def create_and_fit(self, X, y):
        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)
        return classifier
    
@pytest.fixture
def mock_feature():
    return MockFeature(values=["Value1", "Value2"])

def test_set_feature_correctly(mock_feature):
    cft = CategoricalFeatureTree()
    cft.set_feature(mock_feature)
    assert cft.feature == mock_feature, "The feature was not set correctly"

@pytest.fixture
def setup_cft_with_mocks(mock_feature):
    cft = CategoricalFeatureTree()
    cft.set_feature(mock_feature)
    
   
    cft.tree_creator = SimpleTreeCreator()

    # Mock the external components
    cft.feature_weight_calculator = MagicMock()
    cft.value_weights_calculator = MagicMock()
    
    # Mock the confusion matrix to be used by the weight calculators
    cft.confusion_matrix = np.array([[5, 2], [1, 4]])
    
    # Setup return values for calculators
    cft.feature_weight_calculator.calculate_weight.return_value = 0.5
    cft.value_weights_calculator.calculate_values_weight.return_value = {"Value1": 0.7, "Value2": 0.3}
    
    return cft, mock_feature

def test_fit_method_integrations(setup_cft_with_mocks):
    cft, mock_feature = setup_cft_with_mocks
    
    # Create dummy X and y dataframes for fitting
    X = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
    y = pd.DataFrame(np.random.choice(["Value1", "Value2"], size=10), columns=['Target'])
    
    cft.fit(X, y['Target'])
        
    # Verify the feature_weight_calculator and value_weights_calculator were called with the confusion matrix
    cft.feature_weight_calculator.calculate_weight.assert_called_once_with(cft.confusion_matrix)
    cft.value_weights_calculator.calculate_values_weight.assert_called_once_with(cft.confusion_matrix, ["Value1", "Value2"])
    
    # Verify that the classifier, feature weight, and value weights are set
    assert cft.classifier is not None, "Classifier was not set"
    assert cft.feature_weight == 0.5, "Feature weight was not calculated correctly"
    assert cft.value_weights == {"Value1": 0.7, "Value2": 0.3}, "Value weights were not calculated correctly"


@pytest.fixture
def prepared_cft_instance():
    cft = CategoricalFeatureTree()
    mock_feature = MockFeature(values=["Value1", "Value2"])
    cft.set_feature(mock_feature)

    # Mock the classifier to control the predict_proba output
    mock_classifier = MagicMock()
    mock_classifier.classes_ = np.array(["Value1", "Value2"])
    mock_classifier.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2]]))
    cft.classifier = mock_classifier

    # Set the feature_weight and value_weights directly
    cft.feature_weight = 0.5
    cft.value_weights = {"Value1": 0.7, "Value2": 0.3}

    return cft


def test_calculate_score_normal(prepared_cft_instance):
    cft = prepared_cft_instance
    
    # Assuming the instance is a pandas Series of the correct shape
    instance = pd.Series(np.random.rand(4), index=['A', 'B', 'C', 'D'])
    actual_value = "Value1"
    
    # Calculate the score for a normal case
    score = cft.calculate_score_normal(instance, actual_value)
    
    # Expected score calculation based on mocked setup
    # Note: This calculation should match how `calculate_score_normal` is implemented
    expected_score = 0.5 * 0.7 * 0.8  # feature_weight * value_weight for Value1 * classifier probability for Value1
    
    assert score == expected_score, f"Expected score {expected_score}, but got {score}"

@pytest.fixture
def prepared_cft_for_anomaly():
    cft = CategoricalFeatureTree()
    # Assuming MockFeature and other setup from previous examples
    mock_feature = MockFeature(values=["Value1", "Value2", "Value3"])
    cft.set_feature(mock_feature)

    # Mock the classifier with controlled predict_proba output
    mock_classifier = MagicMock()
    mock_classifier.classes_ = np.array(["Value1", "Value2", "Value3"])
    # Example: for a given instance, classifier probabilities are 70% for Value1, 20% for Value2, and 10% for Value3
    mock_classifier.predict_proba = MagicMock(return_value=np.array([[0.7, 0.2, 0.1]]))
    cft.classifier = mock_classifier

    # Set the feature_weight and value_weights directly
    cft.feature_weight = 0.5
    cft.value_weights = {"Value1": 0.7, "Value2": 0.2, "Value3": 0.1}

    return cft

def test_calculate_score_anomaly(prepared_cft_for_anomaly):
    cft = prepared_cft_for_anomaly
    
    instance = pd.Series(np.random.rand(4), index=['A', 'B', 'C', 'D'])
    actual_value = "Value1"  # Assuming the actual value is "Value1"
    
    # Calculate the score for an anomaly case
    anomaly_score = cft.calculate_score_anomaly(instance, actual_value)
    
    # Expected anomaly score calculation based on the mocked setup
    # Note: This calculation should match how `calculate_score_anomaly` is implemented, focusing on values other than the actual one
    expected_anomaly_score = 0.5 * ((0.2 * 0.2) + (0.1 * 0.1)) / 2  # feature_weight * sum of (value_weights * classifier probabilities for other values) / count of other values
    
    assert anomaly_score == expected_anomaly_score, f"Expected anomaly score {expected_anomaly_score}, but got {anomaly_score}"


@pytest.fixture
def prepared_cft_for_edge_case():
    cft = CategoricalFeatureTree()
    # Assuming MockFeature and other setup from previous examples
    mock_feature = MockFeature(values=["Value1", "Value2"])
    cft.set_feature(mock_feature)

    # Mock the classifier with controlled predict_proba output
    mock_classifier = MagicMock()
    mock_classifier.classes_ = np.array(["Value1", "Value2"])
    mock_classifier.predict_proba = MagicMock(return_value=np.array([[0.5, 0.5]]))
    cft.classifier = mock_classifier

    # Set the feature_weight and value_weights directly
    cft.feature_weight = 0.5
    cft.value_weights = {"Value1": 0.5, "Value2": 0.5}

    return cft

def test_calculate_score_normal_with_unexpected_value(prepared_cft_for_edge_case):
    cft = prepared_cft_for_edge_case
    
    instance = pd.Series(np.random.rand(4), index=['A', 'B', 'C', 'D'])
    actual_value_not_in_feature = "Value3"  # Value not in the feature's possible values
    
    # Calculate the normal score with an actual value not present in the feature's values
    score = cft.calculate_score_normal(instance, actual_value_not_in_feature)
    
    assert score == 0.0, "Expected score to be 0.0 for an actual value not in the feature's possible values"

def test_calculate_score_anomaly_with_unexpected_value(prepared_cft_for_edge_case):
    cft = prepared_cft_for_edge_case
    
    instance = pd.Series(np.random.rand(4), index=['A', 'B', 'C', 'D'])
    actual_value_not_in_feature = "Value3"  # Value not in the feature's possible values
    
    # Calculate the anomaly score with an actual value not present in the feature's values
    anomaly_score = cft.calculate_score_anomaly(instance, actual_value_not_in_feature)
    
    assert anomaly_score == 0.0, "Expected anomaly score to be 0.0 for an actual value not in the feature's possible values"
