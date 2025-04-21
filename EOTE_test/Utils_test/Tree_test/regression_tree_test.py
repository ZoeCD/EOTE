import sys
sys.path.append(".")
import unittest
import pandas as pd
from sklearn.datasets import make_classification
from EOTE.Trees import RegressionTree
from sklearn.tree import DecisionTreeRegressor
import pytest
from sklearn.datasets import fetch_california_housing
import pandas as pd


def test_set_alphas_correctly():
    alphas = [0.1, 0.2, 0.3]
    rt = RegressionTree(alphas=alphas)
    assert rt.alphas == alphas, "Alphas were not set correctly"

def test_change_alphas_before_create_and_fit():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    
    rt = RegressionTree([0.1])
    
    new_alphas = [0.01, 0.02]
    rt.set_alphas(new_alphas)
    classifier = rt.create_and_fit(X, y)
    
    assert rt.alphas == new_alphas, "Alphas were not updated correctly"


def test_create_and_fit_runs_without_alphas():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    
    rt = RegressionTree()
    classifier = rt.create_and_fit(X, y)
    
    assert classifier, "create_and_fit failed to return a classifier without alphas set"

def test_tree_depth_larger_than_1_with_alphas():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    
    rt = RegressionTree()
    rt.set_alphas([0.1, 0.01, 0.001])
    classifier = rt.create_and_fit(X, y)
    
    assert classifier.get_depth() > 1, "Tree depth was not greater than 1 when alphas were set"


def test_alphas_impact_classifier():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    
    rt = RegressionTree()
    rt.set_alphas([0.0])  # Minimal pruning
    classifier_minimal = rt.create_and_fit(X, y)
    depth_minimal = classifier_minimal.get_depth()
    
    rt.set_alphas([0.1])  # Increased pruning
    classifier_increased = rt.create_and_fit(X, y)
    depth_increased = classifier_increased.get_depth()
    
    assert depth_increased < depth_minimal, "Increased alphas did not result in more pruning (shallower tree)"
