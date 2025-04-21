import sys
sys.path.append(".")
import unittest
from EOTE.Trees import PathShortenerMixedData
import pytest

@pytest.fixture
def path_shortener():
    return PathShortenerMixedData()

def test_simple_case_no_reduction(path_shortener):
    path = "If (gender_F ≤ 0.5) AND (age > 50) then {'negative': '100.0%', 'positive': '0.0%'}"
    feature_names = ["gender", "age"]
    features = ["gender_M", "gender_F", "age"]
    answer = "If (gender ≠ F) AND (age > 50) then {'negative': '100.0%', 'positive': '0.0%'}"
    new_path = path_shortener.shorten_path(feature_names, features, path)
    assert answer == new_path

def test_two_features_no_reduction(path_shortener):
    path = "If (gender_M > 0.5) AND (color_yellow > 0.5) then {'negative': '100.0%', 'positive': '0.0%'}"
    feature_names = ["gender", "age", "color"]
    features = ["gender_M", "gender_F", "age", "color_yellow", "color_purple"]
    answer = "If (gender is M) AND (color is yellow) then {'negative': '100.0%', 'positive': '0.0%'}"
    new_path = path_shortener.shorten_path(feature_names, features, path)
    assert answer == new_path

def test_multiple_features_reduction(path_shortener):
    path = "If (Whole_weight > 0.863999992609024) AND (Diameter > 0.5024999976158142) AND " \
            "(Diameter ≤ 0.5425000190734863) AND (Whole_weight ≤ 1.3729999661445618) AND " \
            "AND (Sex_M > 0.5) AND (Size_small ≤ 0.5) AND (Size_medium ≤ 0.5) AND then (Length = 0.63)"

    feature_names = ["Whole_weight", "Diameter", "Sex", "Size"]
    features = ["Whole_weight", "Diameter", "Sex_M", "Sex_F", "Size_small", "Size_medium"]
    answer = "If (0.863999992609024 < Whole_weight ≤ 1.3729999661445618) AND " \
                "(0.5024999976158142 < Diameter ≤ 0.5425000190734863) AND (Sex is M)" \
                " AND (Size ≠ small, medium) then (Length = 0.63)"
    new_path = path_shortener.shorten_path(feature_names, features, path)
    assert answer == new_path

def test_regression_threshold_selection(path_shortener):
    path = "If (Whole_weight > 0.863999992609024) AND (Whole_weight > 0.963999992609024) AND " \
            "(Whole_weight ≤ 1.0) AND (Whole_weight ≤ 1.3729999661445618) AND " \
            "then (Length = 0.63)"

    feature_names = ["Whole_weight", "Diameter", "Sex", "Size"]
    features = ["Whole_weight", "Diameter", "Sex_M", "Sex_F", "Size_small", "Size_medium"]
    answer = "If (0.963999992609024 < Whole_weight ≤ 1.0) then (Length = 0.63)"
    new_path = path_shortener.shorten_path(feature_names, features, path)
    assert answer == new_path

def test_path_without_then_clause(path_shortener):
    all_features = ['Feature1']
    current_features = ['Feature1']
    path = "Feature1 Only has one node."  
    assert path_shortener.shorten_path(all_features, current_features, path) == path

def test_numerical_feature_with_max_only(path_shortener):
    all_features = ['Feature1']
    current_features = ['Feature1']
    path = "If (Feature1 ≤ 20) then outcome is positive"
    expected_output = "If (Feature1 ≤ 20) then outcome is positive" 
    assert path_shortener.shorten_path(all_features, current_features, path) == expected_output