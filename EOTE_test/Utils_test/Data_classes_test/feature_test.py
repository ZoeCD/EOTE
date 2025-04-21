import sys
sys.path.append(".")
import pytest
from EOTE.Utils.Data_classes import Feature
import numpy as np

@pytest.fixture
def nominal_feature():
    return Feature("nominal_feature", values=np.array(["A", "B", "C"]))

@pytest.fixture
def numerical_feature():
    return Feature("numerical_feature", values=np.array([1, 2, 3]))

@pytest.fixture
def empty_feature():
    return Feature("empty_feature")

def test_init_with_valid_name_and_no_values():
    feature = Feature("test_feature")
    assert feature.name() == "test_feature"
    assert feature.values() is None
    assert feature.type() == None  

def test_init_with_valid_name_and_empty_list():
    feature = Feature("test_feature", [])
    assert feature.name() == "test_feature"
    assert feature.values() == []
    assert feature.type() == None 

def test_init_with_values(numerical_feature, nominal_feature):
    assert numerical_feature.type() == "Numerical"
    assert nominal_feature.type() == "Nominal"

def test_name(nominal_feature):
    assert nominal_feature.name() == "nominal_feature"

def test_values_setter_and_getter(nominal_feature):
    nominal_feature.set_values(np.array(["D", "E", "F"]))
    np.testing.assert_array_equal(nominal_feature.values(), np.array(["D", "E", "F"]))

def test_missing_value_objects_setter_and_getter(nominal_feature):
    nominal_feature.set_missing_value_objects([0, 2])
    assert nominal_feature.missing_value_objects == [0, 2]

def test_type_after_setting_new_values(numerical_feature):
    numerical_feature.set_values(np.array(["X", "Y", "Z"]))
    assert numerical_feature.type() == "Nominal"

