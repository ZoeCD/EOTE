import sys
sys.path.append(".")
import unittest
import pandas as pd
from sklearn.datasets import make_classification
from EOTE.Trees import ClassificationTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pytest


def test_set_alphas_correctly():
    alphas = [0.1, 0.2, 0.3]
    ct = ClassificationTree(alphas=alphas)
    assert ct.alphas == alphas, "Alphas were not set correctly"

def test_change_alphas_before_create_and_fit():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    ct = ClassificationTree([0.1])
    
    new_alphas = [0.01, 0.02]
    ct.set_alphas(new_alphas)
    classifier = ct.create_and_fit(X, y)
    
    assert ct.alphas == new_alphas, "Alphas were not updated correctly"


def test_create_and_fit_runs_without_alphas():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    ct = ClassificationTree()
    classifier = ct.create_and_fit(X, y)
    
    assert classifier, "create_and_fit failed to return a classifier without alphas set"

def test_tree_depth_larger_than_1_with_alphas():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    tc = ClassificationTree()
    tc.set_alphas([0.1, 0.01, 0.001])
    classifier = tc.create_and_fit(X, y)
    
    assert classifier.get_depth() > 1, "Tree depth was not greater than 1 when alphas were set"


def test_alphas_impact_classifier():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    ct = ClassificationTree()
    ct.set_alphas([0.0])  # Minimal pruning
    classifier_minimal = ct.create_and_fit(X, y)
    depth_minimal = classifier_minimal.get_depth()
    
    ct.set_alphas([0.1])  # Increased pruning
    classifier_increased = ct.create_and_fit(X, y)
    depth_increased = classifier_increased.get_depth()
    
    assert depth_increased < depth_minimal, "Increased alphas did not result in more pruning (shallower tree)"
