import sys
sys.path.append(".")
import pandas as pd
from EOTE.Utils.Data_processing.data_cleaning import InsufficientCategoricalValuesAttributeRemover
import pytest
from EOTE.Utils import UniqueClassVerifier
import numpy as np

def test_AttributeRemover_all_valid_data():
        data = [["yellow", "small", "dip", "child"],
                ["yellow", "small", "dip", "child"],
                ["yellow", "small", "dip", "child"],
                ["yellow", "small", "dip", "child"],
                ["yellow", "small", "dip", "child"],
                ["yellow", "small", "dip", "child"],
                ["purple", "medium", "stretch", "adult"],
                ["purple", "medium", "stretch", "adult"],
                ["purple", "medium", "stretch", "adult"],
                ["purple", "medium", "stretch", "adult"],
                ["purple", "medium", "stretch", "adult"],
                ["purple", "medium", "stretch", "adult"]]
        dataset = pd.DataFrame(data, columns=["color", "size", "act", "age"])
        remover = InsufficientCategoricalValuesAttributeRemover(min_values=2, min_instances=3)
        result = remover.remove_invalid_attributes(dataset)
        pd.testing.assert_frame_equal(dataset, result)

def test_AttributeRemover_insufficient_instances_per_value():
    data = [["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "adult"],
            ["purple", "medium", "stretch", "adult"]]
    dataset = pd.DataFrame(data, columns=["color", "size", "act", "age"])
    expected_data = [["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"]]
    expected_dataset = pd.DataFrame(expected_data, columns=["color", "size", "act"],)
    remover = InsufficientCategoricalValuesAttributeRemover(min_values=2, min_instances=3)
    result = remover.remove_invalid_attributes(dataset)
    pd.testing.assert_frame_equal(expected_dataset, result)

def test_AttributeRemover_min_instances_limit():
    data = [["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "adult"],
            ["purple", "medium", "stretch", "adult"],
            ["purple", "medium", "stretch", "adult"]]
    dataset = pd.DataFrame(data, columns=["color", "size", "act", "age"])
    remover = InsufficientCategoricalValuesAttributeRemover(min_values=2, min_instances=3)
    result = remover.remove_invalid_attributes(dataset)
    pd.testing.assert_frame_equal(dataset, result)

def test_AttributeRemover_insufficient_values_per_attribute():
    data = [["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"],
            ["purple", "medium", "stretch", "child"]]
    dataset = pd.DataFrame(data, columns=["color", "size", "act", "age"])
    expected_data = [["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["yellow", "small", "dip"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"],
            ["purple", "medium", "stretch"]]
    expected_dataset = pd.DataFrame(expected_data, columns=["color", "size", "act"])
    remover = InsufficientCategoricalValuesAttributeRemover(min_values=2, min_instances=3)
    result = remover.remove_invalid_attributes(dataset)
    pd.testing.assert_frame_equal(expected_dataset, result)

def test_AttributeRemover_with_missing_data():
    data = [["yellow", "small", "dip", np.nan],
            ["yellow", "small", "dip", np.nan],
            ["yellow", "small", "dip", "child"],
            ["yellow", "small", "dip", np.nan],
            ["yellow", "small", "dip", np.nan],
            ["yellow", "small", "dip", np.nan],
            ["purple", "medium", "stretch", np.nan],
            ["purple", "medium", "stretch", np.nan],
            ["purple", "medium", "stretch", np.nan],
            ["purple", "medium", "stretch", np.nan],
            ["purple", "medium", "stretch", np.nan],
            ["purple", "medium", "stretch", np.nan]]
    dataset = pd.DataFrame(data, columns=["color", "size", "act", "age"])
    expected_data = [["yellow", "small", "dip"],
                        ["yellow", "small", "dip"],
                        ["yellow", "small", "dip"],
                        ["yellow", "small", "dip"],
                        ["yellow", "small", "dip"],
                        ["yellow", "small", "dip"],
                        ["purple", "medium", "stretch"],
                        ["purple", "medium", "stretch"],
                        ["purple", "medium", "stretch"],
                        ["purple", "medium", "stretch"],
                        ["purple", "medium", "stretch"],
                        ["purple", "medium", "stretch"]]
    expected_dataset = pd.DataFrame(expected_data, columns=["color", "size", "act"])
    remover = InsufficientCategoricalValuesAttributeRemover(min_values=2, min_instances=3)
    result = remover.remove_invalid_attributes(dataset)
    pd.testing.assert_frame_equal(expected_dataset, result)

def test_AttributeRemover_mixed_data():
    data = [["yellow", 12.4, "dip", np.nan],
            ["yellow", 503.2, "dip", np.nan],
            ["yellow", 203.121, "dip", "child"],
            ["yellow", 402.211, "dip", np.nan],
            ["yellow", 39.12, "dip", np.nan],
            ["yellow", 192.1, "dip", np.nan],
            ["purple", 92.120, "stretch", np.nan],
            ["purple", 743.23, "stretch", np.nan],
            ["purple", 473.2, "stretch", np.nan],
            ["purple", 48.23, "stretch", np.nan],
            ["purple", 32.12, "stretch", np.nan],
            ["purple", 49.21, "stretch", np.nan]]
    dataset = pd.DataFrame(data, columns=["color", "size", "act", "age"])
    expected_data = [["yellow", 12.4, "dip"],
                        ["yellow", 503.2, "dip"],
                        ["yellow", 203.121, "dip"],
                        ["yellow", 402.211, "dip"],
                        ["yellow", 39.12, "dip"],
                        ["yellow", 192.1, "dip"],
                        ["purple", 92.120, "stretch"],
                        ["purple", 743.23, "stretch"],
                        ["purple", 473.2, "stretch"],
                        ["purple", 48.23, "stretch"],
                        ["purple", 32.12, "stretch"],
                        ["purple", 49.21, "stretch"]]
    expected_dataset = pd.DataFrame(expected_data, columns=["color", "size", "act"])
    remover = InsufficientCategoricalValuesAttributeRemover(min_values=2, min_instances=3)
    result = remover.remove_invalid_attributes(dataset)
    pd.testing.assert_frame_equal(expected_dataset, result)

def test_class_verifier_not_unique_class_data():
        class_data = pd.DataFrame({"class": ["genuine",
                                "genuine",
                                "genuine",
                                "genuine",
                                "genuine",
                                "anomaly",
                                "anomaly",
                                "anomaly"]})

        verifier = UniqueClassVerifier()
        with pytest.raises(ValueError):
            verifier.verify_if_valid(class_data), "Class verifier failed to raise an exception for invalid class dataframe!"

def test_class_verifier_empty_dataframe():
        class_data = pd.DataFrame()

        verifier = UniqueClassVerifier()
        with pytest.raises(ValueError):
            verifier.verify_if_valid(class_data), "Class verifier failed to raise an exception for empty dataframe!"

def test_class_verifier_unique_class_data():
    class_data = pd.DataFrame({"class": ["genuine",
                                "genuine",
                                "genuine",
                                "genuine",
                                "genuine"]})
    verifier = UniqueClassVerifier()
    result = verifier.verify_if_valid(class_data)
    assert result == True, "Class verifier failed to verify a valid class dataframe!"


