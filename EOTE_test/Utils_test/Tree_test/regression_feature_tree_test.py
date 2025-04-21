from EOTE.Trees import NumericalFeatureTree
from EOTE.Utils import Feature
import pytest
from unittest.mock import MagicMock, Mock
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class MockFeature(Feature):
    def __init__(self, name):
        self.name = name

def test_set_feature():
    tree = NumericalFeatureTree()
    test_feature = MockFeature(name="test_feature")
    
    # Use the set_feature method
    tree.set_feature(test_feature)
    
    # Now assert that the feature has been correctly set
    assert tree.feature == test_feature, "The feature was not correctly set."

class MockAnomalyScorer():
    def set_domain(self, y):
        self.domain = [min(y), max(y)]

class SimpleTreeCreator:
    def create_and_fit(self, X, y):
        classifier = DecisionTreeRegressor()
        classifier.fit(X, y)
        return classifier

def test_fit_method():
    # Initialize your NumericalFeatureTree
    tree = NumericalFeatureTree()

    # Mock the regressor and anomaly scorer within the tree
    tree.regressor = SimpleTreeCreator()
    tree.anomaly_scorer = MockAnomalyScorer()

    # Create dummy X and y data
    X = pd.DataFrame({'feature1': np.arange(10), 'feature2': np.arange(10) * 2})
    y = pd.DataFrame({'target': np.random.rand(10)})

    # Fit the tree with the data
    tree.fit(X, y['target'])

    # Assertions
    # Check if the regressor has been fitted
    assert hasattr(tree.regressor, 'tree_'), "The regressor has not been fitted."

    # Check if leaves_and_instances dictionary is properly initialized
    assert isinstance(tree.leaves_and_instances, dict), "leaves_and_instances is not initialized correctly."
    assert len(tree.leaves_and_instances) > 0, "leaves_and_instances should have at least one leaf."

    # Check if the anomaly scorer's domain is set
    assert hasattr(tree.anomaly_scorer, 'domain'), "The anomaly scorer's domain is not set."
    assert len(tree.anomaly_scorer.domain) == 2, "The anomaly scorer's domain should have two elements."
    assert tree.anomaly_scorer.domain[0] == y['target'].min() and tree.anomaly_scorer.domain[1] == y['target'].max(), "The anomaly scorer's domain is not set correctly."

@pytest.fixture
def setup_numerical_feature_tree():
    tree = NumericalFeatureTree()
    
    # Mock the regressor with a predict method
    tree.regressor = Mock()
    tree.regressor.predict = Mock(return_value=np.array([0.5]))  # Mocked predicted value
    
    # Mock the anomaly scorer with a calculate_score_normal method
    tree.anomaly_scorer = Mock()
    tree.anomaly_scorer.calculate_score_normal = Mock(return_value=0.1)  # Mocked anomaly score
    
    return tree

def test_calculate_score_normal(setup_numerical_feature_tree):
    tree = setup_numerical_feature_tree
    
    # Create a dummy instance (as a pandas Series)
    instance = pd.Series({'feature1': 0.5, 'feature2': 1.0})
    actual_value = 0.6  # Dummy actual value
    
    # Calculate the anomaly score
    score = tree.calculate_score_normal(instance, actual_value)
    
    # Assertions
    # Ensure the regressor's predict method was called correctly
    tree.regressor.predict.assert_called_once()
    
    # Ensure the anomaly scorer's calculate_score_normal method was called with the mocked predicted and actual values
    tree.anomaly_scorer.calculate_score_normal.assert_called_once_with(actual_value, 0.5)
    
    # Check that the calculated score matches the mocked return value from calculate_score_normal
    assert score == 0.1, "The calculated anomaly score is incorrect."

@pytest.fixture
def setup_numerical_feature_tree_for_anomaly():
    tree = NumericalFeatureTree()
    
    # Mock the regressor with a predict method
    tree.regressor = Mock()
    tree.regressor.predict = Mock(return_value=np.array([0.5]))  # Mocked predicted value
    
    # Mock the anomaly scorer with a calculate_score_anomaly method
    tree.anomaly_scorer = Mock()
    tree.anomaly_scorer.calculate_score_anomaly = Mock(return_value=0.2)  # Mocked anomaly score
    
    return tree

def test_calculate_score_anomaly(setup_numerical_feature_tree_for_anomaly):
    tree = setup_numerical_feature_tree_for_anomaly
    
    # Create a dummy instance (as a pandas Series)
    instance = pd.Series({'feature1': 0.5, 'feature2': 1.0})
    actual_value = 0.6  # Dummy actual value
    
    # Calculate the anomaly score
    score = tree.calculate_score_anomaly(instance, actual_value)
    
    # Assertions
    # Ensure the regressor's predict method was called correctly
    tree.regressor.predict.assert_called_once()
    
    # Ensure the anomaly scorer's calculate_score_anomaly method was called with the mocked predicted and actual values
    tree.anomaly_scorer.calculate_score_anomaly.assert_called_once_with(actual_value, 0.5)
    
    # Check that the calculated score matches the mocked return value from calculate_score_anomaly
    assert score == 0.2, "The calculated anomaly score for an anomaly is incorrect."

def test_calculate_score_normal_with_array_return(setup_numerical_feature_tree):
    tree = setup_numerical_feature_tree

    # Adjust the mock to return a numpy array
    tree.anomaly_scorer.calculate_score_normal.return_value = np.array([0.3])

    instance = pd.Series({'feature1': 0.5, 'feature2': 1.0})
    actual_value = 0.6

    score = tree.calculate_score_normal(instance, actual_value)

    # Check that the method correctly handles a numpy array and returns the first element
    assert score == 0.3, "The method did not return the first element of the numpy array as expected."

def test_calculate_score_anomaly_with_array_return(setup_numerical_feature_tree_for_anomaly):
    tree = setup_numerical_feature_tree_for_anomaly

    # Adjust the mock to return a numpy array
    tree.anomaly_scorer.calculate_score_anomaly.return_value = np.array([0.4])

    instance = pd.Series({'feature1': 0.5, 'feature2': 1.0})
    actual_value = 0.6

    score = tree.calculate_score_anomaly(instance, actual_value)

    # Check that the method correctly handles a numpy array and returns the first element
    assert score == 0.4, "The method did not return the first element of the numpy array as expected."
